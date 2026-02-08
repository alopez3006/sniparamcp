"""RLM Engine - Documentation query engine implementation.

This module implements the RLM (REPL Language Model) tools that provide
context-efficient documentation queries. It processes markdown documentation
and provides various query tools.
"""

import asyncio
import hashlib
import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .db import get_db
from .models import (
    ContextQueryResult,
    ContextSection,
    ContextSectionRef,
    DecomposeResult,
    DecomposeStrategy,
    DeleteSummaryResult,
    DocumentCategoryEnum,
    GetChunkResult,
    GetSummariesResult,
    GetTemplateResult,
    ListTemplatesResult,
    MultiQueryResult,
    MultiQueryResultItem,
    Plan,
    PlanResult,
    PlanStep,
    PlanStrategy,
    PromptTemplateInfo,
    RequestAccessResult,
    SearchMode,
    SectionInfo,
    SettingsResult,
    SharedContextResult,
    SharedDocumentInfo,
    StoreSummaryResult,
    SubQuery,
    SummaryInfo,
    SummaryType,
    SyncDocumentsResult,
    ToolName,
    UploadDocumentResult,
)
from .services.agent_limits import (
    check_memory_limits,
    validate_agents_access,
)
from .services.agent_memory import (
    delete_memories,
    list_memories,
    semantic_recall,
    store_memory,
)
from .services.chunker import get_chunker
from .services.embeddings import get_light_embeddings_service
from .services.query_router import route_query
from .services.shared_context import (
    DocumentCategory,
    allocate_shared_context_budget,
    compute_context_hash,
    create_shared_document,
    get_shared_prompt_templates,
    list_shared_collections,
    load_project_shared_context,
    merge_shared_context_with_project_docs,
)
from .services.swarm_coordinator import (
    acquire_claim,
    claim_task,
    complete_task,
    create_swarm,
    create_task,
    get_state,
    join_swarm,
    release_claim,
    set_state,
)
from .services.swarm_events import (
    broadcast_event,
)

# ---------------------------------------------------------------------------
# Stop words — excluded from keyword scoring to prevent false title matches.
# Without this, "what are prices?" ranks "What Happens When Limits Are Exceeded"
# above actual pricing content because "what" and "are" get 5x title weight.
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset({
    # Articles, auxiliaries, modals
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need",
    # Prepositions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    # Adverbs and conjunctions
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if",
    # Pronouns and determiners
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "my", "your", "his", "her", "our", "their", "about",
    "up", "also", "any", "many", "much",
    # Generic nouns that appear in many queries but aren't distinctive
    "value", "proposition", "core", "main", "key", "primary",
    "work", "works", "working", "feature", "features",
    "thing", "things", "something", "everything",
    # Common verbs that don't add search value
    "use", "used", "using", "get", "gets", "getting",
    "make", "makes", "making", "see", "sees", "seeing",
    "know", "knows", "knowing", "think", "thinks",
    "want", "wants", "wanting", "like", "likes",
})


def _stem_keyword(word: str) -> str:
    """Basic suffix stemmer to improve keyword matching.

    Strips common English suffixes to produce an approximate stem used for
    substring matching.  A shorter stem naturally matches more morphological
    variants, e.g. ``_stem_keyword("prices")`` → ``"pric"`` which is a
    substring of "pricing", "price", "priced", etc.

    Minimum-length guards ensure short words like "doing" (5 chars) are not
    over-stripped into meaningless 2-char stems.
    """
    # Longer suffixes first — order matters.
    if len(word) > 7 and word.endswith("tion"):
        return word[:-4]
    if len(word) > 7 and word.endswith("ment"):
        return word[:-4]
    if len(word) > 7 and word.endswith("ness"):
        return word[:-4]
    if len(word) > 7 and word.endswith("ible"):
        return word[:-4]
    if len(word) > 7 and word.endswith("able"):
        return word[:-4]
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 6 and word.endswith("ies"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed") and not word.endswith("eed"):
        return word[:-2]
    if len(word) > 5 and word.endswith("er"):
        return word[:-2]
    if len(word) > 5 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 5 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if len(word) > 4 and word.endswith("e") and not word.endswith("ee"):
        return word[:-1]
    return word


# ---------------------------------------------------------------------------
# Adaptive Hybrid Search: weight profiles & RRF constant
# ---------------------------------------------------------------------------
# When keyword ranking has high confidence (exact title match, specific terms),
# boost keyword weight to prevent semantic noise from diluting precise results.
# When query is conceptual (how/why/explain), boost semantic weight.
_HYBRID_KEYWORD_HEAVY = (0.60, 0.40)  # factual / title-match queries (was 0.70/0.30)
_HYBRID_BALANCED = (0.40, 0.60)  # default - favor semantic for better recall (was 0.50/0.50)
_HYBRID_SEMANTIC_HEAVY = (0.25, 0.75)  # conceptual / how-why queries

# Reciprocal Rank Fusion constant (k=60 is the standard from Cormack+ 2009).
# Lower k gives more weight to top-ranked results, improving precision.
# Higher k smooths rankings, improving recall.
_RRF_K = 45  # Reduced from 60 for better precision while keeping good recall

# Generic title terms — these get reduced title weight (1.5x instead of 5x)
# because they appear in many unrelated sections and cause false matches.
# E.g., "Snipara tools not available" ranks highly for "What tools does Snipara expose?"
# because "snipara" + "tools" match the title even though it's a debugging section.
_GENERIC_TITLE_TERMS = frozenset({
    "snipara", "rlm", "mcp",  # Project-specific but ubiquitous
    "tools", "tool", "guide", "reference", "overview", "docs",  # Generic doc terms
    "how", "what", "when", "where", "why",  # Question words (shouldn't boost titles)
    "using", "use", "get", "set", "run", "make",  # Common verbs
    "available", "not", "error", "issue", "troubleshoot",  # Debugging terms
})

# Query terms that signal structured/factual content (keyword-friendly)
# These trigger keyword-heavy weights (60/40) for better precision
_SPECIFIC_QUERY_TERMS = frozenset(
    {
        # Technical/infrastructure
        "pricing", "price", "cost", "tier", "plan",
        "stack", "version", "model", "schema", "table",
        "endpoint", "api", "command", "config", "database",
        "deploy", "deployment", "auth", "authentication",
        # Business/product terms - these need keyword matching
        "value", "proposition", "feature", "benefit", "overview",
        "architecture", "workflow", "integration", "limit", "rate",
        # Search-specific terms
        "hybrid", "semantic", "keyword", "search", "query",
        "token", "context", "chunk", "section", "document",
    }
)

# Conceptual query prefixes (semantic-friendly)
# These trigger semantic-heavy weights (25/75) for better conceptual matching
_CONCEPTUAL_PREFIXES = (
    # How/Why questions
    "how does",
    "how do",
    "how is",
    "how are",
    "how can",
    "why does",
    "why do",
    "why is",
    "why are",
    # What questions (conceptual, not factual lookups)
    "what is",
    "what are",
    "what does",
    "what do",
    # Explanation requests
    "explain",
    "describe",
    "compare",
    "tell me about",
    "overview of",
    # Specific conceptual patterns
    "what happens when",
    "what is the difference",
    "what are the tradeoffs",
    "value proposition",
    "core value",
    "main purpose",
    "key features",
)

# ---------------------------------------------------------------------------
# List/Enumeration Query Detection
# ---------------------------------------------------------------------------
# Queries asking for lists of items (articles, tasks, features) should boost
# sections with numbered patterns like "### Article #1", "1. First item", etc.
# This improves ranking for "what are the next articles to write" type queries.
_LIST_QUERY_PATTERNS = frozenset({
    "what are the",
    "list the",
    "list all",
    "which",
    "what to write",
    "what to do",
    "next articles",
    "next tasks",
    "next steps",
    "upcoming",
    "planned",
    "todo",
    "to-do",
    "roadmap",
})

# Patterns in section titles/content that indicate enumerated list items
# These get boosted when a list query is detected
_NUMBERED_SECTION_PATTERNS = (
    r"^#+\s*(?:article|task|step|item|feature|issue|bug|story)\s*#?\d+",  # ### Article #1
    r"^#+\s*\d+[\.\):]",  # ## 1. or ## 1) or ## 1:
    r"^\d+[\.\)]",  # 1. or 1) at start
    r"#\d+\b",  # #1, #2, etc.
)

# Terms indicating planned/unpublished/future content
# Boost sections containing these when query asks about "next" or "planned" items
_PLANNED_CONTENT_MARKERS = frozenset({
    "📝", "unpublished", "planned", "draft", "todo", "upcoming",
    "next:", "status:", "wip", "in progress", "pending",
})

# ---------------------------------------------------------------------------
# Query Expansion: Abstract terms → concrete keywords for better search recall
# ---------------------------------------------------------------------------
# Abstract queries like "architecture" miss specific sections because they don't
# contain the actual component names. Expand abstract terms with concrete keywords.
_QUERY_EXPANSIONS: dict[str, list[str]] = {
    # Architecture queries need component names
    "architecture": [
        "snipara-mcp", "FastAPI", "Railway", "Vercel", "Neon",
        "component", "three-component", "PostgreSQL", "Redis",
    ],
    "three-component": [
        "snipara-mcp", "FastAPI", "Vercel", "Railway", "PostgreSQL",
    ],
    "components": [
        "snipara-mcp", "FastAPI", "Vercel", "web app", "MCP server",
    ],
    # MCP tools queries need tool names
    "mcp tools": [
        "rlm_context_query", "rlm_ask", "rlm_search", "rlm_decompose",
        "rlm_multi_query", "rlm_plan", "rlm_remember", "rlm_recall",
    ],
    "tools": [
        "rlm_context_query", "rlm_ask", "rlm_search", "rlm_decompose",
    ],
    # Value proposition needs business terms
    "value proposition": [
        "context optimization", "token reduction", "90%", "LLM-agnostic",
        "high margins", "no vendor lock-in",
    ],
    # Shared context needs budget allocation terms
    "shared context": [
        "budget allocation", "MANDATORY", "BEST_PRACTICES", "GUIDELINES",
        "REFERENCE", "40%", "30%", "20%", "10%",
    ],
    "budget allocation": [
        "MANDATORY", "BEST_PRACTICES", "GUIDELINES", "REFERENCE",
        "40%", "30%", "20%", "10%", "shared context",
    ],
    # Pricing/limits need concrete values
    "pricing": [
        "FREE", "PRO", "TEAM", "ENTERPRISE", "$19", "$49", "$499",
        "queries/mo", "100", "5000", "20000",
    ],
    "limits": [
        "rate limit", "monthly", "429", "exceeded", "reset_at",
    ],
    # Deployment needs infrastructure terms
    "deployment": [
        "Railway", "Vercel", "Neon", "snipara-fastapi", "snipara-server",
        "main branch", "dev branch", "auto-deploy",
    ],
    # Memory/agent features
    "memory": [
        "rlm_remember", "rlm_recall", "rlm_memories", "rlm_forget",
        "ttl_days", "agent", "session", "decision", "learning",
    ],
    "agent": [
        "memory", "swarm", "rlm_remember", "rlm_recall", "coordination",
    ],
}

# Minimum sections to return for abstract/conceptual queries
# Abstract queries need more context to prevent hallucination
ABSTRACT_QUERY_MIN_SECTIONS = 5


def _expand_query(query: str) -> str:
    """Expand abstract query terms with concrete keywords.

    For queries containing abstract terms like "architecture", appends
    concrete keywords that should match documentation sections.

    Args:
        query: Original query string

    Returns:
        Expanded query with additional keywords, or original if no expansion
    """
    query_lower = query.lower()
    expansions: list[str] = []

    for term, keywords in _QUERY_EXPANSIONS.items():
        if term in query_lower:
            expansions.extend(keywords)

    if expansions:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_expansions: list[str] = []
        for kw in expansions:
            kw_lower = kw.lower()
            if kw_lower not in seen and kw_lower not in query_lower:
                seen.add(kw_lower)
                unique_expansions.append(kw)

        if unique_expansions:
            # Append keywords to original query
            expanded = f"{query} {' '.join(unique_expansions)}"
            logger.info(f"Query expansion: '{query}' → '{expanded}'")
            return expanded

    return query


def _is_abstract_query(query: str) -> bool:
    """Check if query is abstract and needs more context sections.

    Abstract queries use broad terms that may match few sections but require
    comprehensive answers. We boost minimum sections to reduce hallucination.
    """
    query_lower = query.lower()
    # Check if query contains any expansion terms
    for term in _QUERY_EXPANSIONS.keys():
        if term in query_lower:
            return True
    # Also check for conceptual prefixes
    return any(query_lower.startswith(p) for p in _CONCEPTUAL_PREFIXES)


def _is_list_query(query: str) -> bool:
    """Check if query is asking for a list/enumeration of items.

    List queries like "what are the next articles to write" should boost
    sections with numbered patterns (Article #1, 1. First item, etc.)
    over prose/template sections that happen to contain matching keywords.
    """
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in _LIST_QUERY_PATTERNS)


def _is_numbered_section(title: str, content: str) -> bool:
    """Check if section title matches numbered/enumerated patterns.

    Returns True for sections like:
    - "### Article #1: Title"
    - "## 1. First Item"
    - "### Step 3: Implementation"
    """
    title_lower = title.lower()
    for pattern in _NUMBERED_SECTION_PATTERNS:
        if re.search(pattern, title_lower, re.IGNORECASE):
            return True
    return False


def _has_planned_content_markers(content: str) -> bool:
    """Check if content contains markers indicating planned/unpublished status.

    Matches sections with 📝, "Status: Unpublished", "Draft", etc.
    """
    content_lower = content.lower()
    return any(marker in content_lower for marker in _PLANNED_CONTENT_MARKERS)


# Plans that have access to semantic search features
SEMANTIC_SEARCH_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to recursive context features
RECURSIVE_CONTEXT_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to advanced planning features
PLAN_FEATURE_PLANS = {Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to query caching
CACHE_ENABLED_PLANS = {Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to summary storage features
SUMMARY_STORAGE_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to shared context features
SHARED_CONTEXT_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to raw document loading (rlm_load_document)
RAW_DOCUMENT_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to REPL context bridge (rlm_repl_context)
REPL_CONTEXT_PLANS = {Plan.PRO, Plan.TEAM, Plan.ENTERPRISE}

# Plans that have access to orchestration features (rlm_load_project, rlm_orchestrate)
ORCHESTRATION_PLANS = {Plan.TEAM, Plan.ENTERPRISE}

# Tool access level categories for per-project access control
# READ_TOOLS: Available to VIEWER and above
READ_TOOLS = {
    ToolName.RLM_ASK,
    ToolName.RLM_SEARCH,
    ToolName.RLM_CONTEXT,
    ToolName.RLM_STATS,
    ToolName.RLM_SECTIONS,
    ToolName.RLM_READ,
    ToolName.RLM_CONTEXT_QUERY,
    ToolName.RLM_DECOMPOSE,
    ToolName.RLM_MULTI_QUERY,
    ToolName.RLM_PLAN,
    ToolName.RLM_GET_SUMMARIES,
    ToolName.RLM_SHARED_CONTEXT,
    ToolName.RLM_LIST_TEMPLATES,
    ToolName.RLM_GET_TEMPLATE,
    ToolName.RLM_RECALL,
    ToolName.RLM_MEMORIES,
    ToolName.RLM_STATE_GET,
    ToolName.RLM_TASK_CLAIM,
    ToolName.RLM_SETTINGS,
    # Phase 12: RLM Orchestration Tools
    ToolName.RLM_LOAD_DOCUMENT,
    ToolName.RLM_LOAD_PROJECT,
    ToolName.RLM_ORCHESTRATE,
    # Phase 13: REPL Context Bridge
    ToolName.RLM_REPL_CONTEXT,
    # Phase 14: Pass-by-Reference
    ToolName.RLM_GET_CHUNK,
}

# WRITE_TOOLS: Available to EDITOR and above
WRITE_TOOLS = {
    ToolName.RLM_INJECT,
    ToolName.RLM_CLEAR_CONTEXT,
    ToolName.RLM_STORE_SUMMARY,
    ToolName.RLM_DELETE_SUMMARY,
    ToolName.RLM_REMEMBER,
    ToolName.RLM_FORGET,
    ToolName.RLM_UPLOAD_DOCUMENT,
    ToolName.RLM_SYNC_DOCUMENTS,
    ToolName.RLM_STATE_SET,
    ToolName.RLM_BROADCAST,
    ToolName.RLM_TASK_COMPLETE,
}

# ADMIN_TOOLS: Available to ADMIN only
ADMIN_TOOLS = {
    ToolName.RLM_SWARM_CREATE,
    ToolName.RLM_SWARM_JOIN,
    ToolName.RLM_CLAIM,
    ToolName.RLM_RELEASE,
    ToolName.RLM_TASK_CREATE,
    ToolName.RLM_MULTI_PROJECT_QUERY,
}

# NONE_ALLOWED: Tools that can be used even with NONE access level
NONE_ALLOWED_TOOLS = {
    ToolName.RLM_REQUEST_ACCESS,
}

# AGENT_TOOLS: Tools that require Agents subscription and Context plan validation
# These tools require Agents TEAM/ENTERPRISE users to have Context TEAM/ENTERPRISE
AGENT_TOOLS = {
    # Memory tools
    ToolName.RLM_REMEMBER,
    ToolName.RLM_RECALL,
    ToolName.RLM_MEMORIES,
    ToolName.RLM_FORGET,
    # Swarm tools
    ToolName.RLM_SWARM_CREATE,
    ToolName.RLM_SWARM_JOIN,
    ToolName.RLM_CLAIM,
    ToolName.RLM_RELEASE,
    ToolName.RLM_STATE_GET,
    ToolName.RLM_STATE_SET,
    ToolName.RLM_BROADCAST,
    ToolName.RLM_TASK_CREATE,
    ToolName.RLM_TASK_CLAIM,
    ToolName.RLM_TASK_COMPLETE,
}

# Default system instructions injected into every query response
# This ensures customers use Snipara tools effectively
DEFAULT_SYSTEM_INSTRUCTIONS = """## Snipara Context Guidelines

You have access to Snipara MCP tools for optimized documentation queries.
**ALWAYS use these tools when searching for information about this project:**

- `rlm_context_query` - Primary tool for finding relevant documentation (use FIRST)
- `rlm_search` - Search for regex patterns across docs
- `rlm_decompose` - Break complex questions into sub-queries
- `rlm_shared_context` - Get team coding standards and best practices

**Workflow:** Query Snipara → Get optimized context → Answer based on results.
Do NOT read files directly when Snipara can provide the context more efficiently.

---
"""

# Grounding instructions - ALWAYS included to prevent hallucination
# This is a core promise: "Anti-Hallucination by Design"
GROUNDING_INSTRUCTIONS = """
## Source Guidelines

Base your answer on the provided documentation context:

1. **Synthesize freely**: Combine and explain information from the provided sections.
2. **Cite specifics**: When mentioning specific facts (numbers, commands, API names), note the source section.
3. **Acknowledge gaps**: If the context doesn't fully cover something, say "The documentation doesn't specifically address X" but still provide helpful context.

---
"""

# Enhanced grounding for low-confidence results
STRICT_GROUNDING_INSTRUCTIONS = """
## Low-Confidence Context Warning

The search returned limited results. Please:

1. **Be conservative**: Stick closely to what's explicitly stated in the sections.
2. **Cite sources**: Reference section titles when making claims.
3. **Note limitations**: If the context seems incomplete, mention this and suggest the user try a more specific query.

---
"""

# Minimum token budget for conceptual queries (to reduce hallucination)
CONCEPTUAL_QUERY_MIN_TOKENS = 5000
# Minimum token budget for multi-hop queries (require combining info from multiple sections)
MULTI_HOP_QUERY_MIN_TOKENS = 6000
# Score threshold below which we consider results "low confidence"
LOW_CONFIDENCE_SCORE_THRESHOLD = 15.0
# Token budget expansion factor when confidence is low
LOW_CONFIDENCE_TOKEN_MULTIPLIER = 1.5

# Multi-hop query patterns - queries that require combining info from multiple sections
_MULTI_HOP_PATTERNS = (
    "trace the",
    "full flow",
    "and back",
    "end to end",
    "end-to-end",
    "step by step",
    "step-by-step",
    "all the",
    "how do they differ",
    "how do they relate",
    "what are all",
    "compare all",
    "difference between",
    "from start to",
    "from beginning to",
    "complete flow",
    "entire flow",
    "chain of",
    "sequence of",
)

# Internal/debug path patterns - sections from these paths get score penalties
# because they contain project internals, debug logs, and session data that
# shouldn't rank above actual documentation.
_INTERNAL_PATH_PATTERNS = (
    ".claude/",       # Claude session files, commands, debug logs
    ".github/",       # GitHub-specific config files
    ".git/",          # Git internals
    ".vscode/",       # VS Code settings
    "node_modules/",  # Dependencies
    "__pycache__/",   # Python cache
    "debug",          # Files containing "debug" in path
    "session",        # Session logs
)
# Score multiplier for sections from internal paths (0.1 = 90% penalty)
_INTERNAL_PATH_PENALTY = 0.1


def _is_internal_path(file_path: str) -> bool:
    """Check if a file path matches internal/debug patterns that should be deprioritized.

    Internal files like .claude/commands/debug.md contain session logs and debugging
    info that can pollute search results when they match common query terms.
    """
    if not file_path:
        return False
    path_lower = file_path.lower()
    return any(pattern in path_lower for pattern in _INTERNAL_PATH_PATTERNS)


# First-query tool tips - injected only on the first query of a session
# This helps users understand all available tools without wasting tokens on every query
# Tips are plan-filtered: users only see tools available to their plan


def get_first_query_tips(plan: "Plan") -> str:
    """Generate plan-filtered tool tips for first query.

    Args:
        plan: User's current plan (FREE, PRO, TEAM, ENTERPRISE)

    Returns:
        Tool tips string with only tools available to this plan
    """
    tips = ["## Snipara Tool Guide (First Query Tips)", ""]

    # Primary tools - available to all plans
    tips.append("**Primary Tools:**")
    tips.append("- `rlm_context_query` - Full documentation query with token budgeting")
    tips.append("- `rlm_ask` - Quick, simple query (~2500 tokens, no config needed)")
    tips.append("- `rlm_search` - Regex pattern search across documentation")
    tips.append("")

    # Pro+ tools - semantic search, decompose, multi-query
    if plan in SEMANTIC_SEARCH_PLANS:
        tips.append("**Power User Tools (Pro+):**")
        tips.append("- `rlm_multi_query` - Batch multiple queries in parallel")
        tips.append("- `rlm_decompose` - Break complex queries into sub-queries")
        tips.append("- `rlm_shared_context` - Get team coding standards/best practices")
        tips.append("- `rlm_load_document` - Load raw document content by file path")
        tips.append("")

    # Team+ tools - multi-project, plan, templates, orchestration
    if plan in PLAN_FEATURE_PLANS:
        tips.append("**Team Tools (Team+):**")
        tips.append("- `rlm_multi_project_query` - Search across ALL your projects")
        tips.append("- `rlm_plan` - Generate execution plan for complex questions")
        tips.append("- `rlm_list_templates` / `rlm_get_template` - Use prompt templates")
        tips.append("- `rlm_load_project` - Load full project structure with content")
        tips.append("- `rlm_orchestrate` - Multi-round context exploration (search + raw load)")
        tips.append("")

    # Utility tools - available to all
    tips.append("**Utility Tools:**")
    tips.append("- `rlm_inject` / `rlm_context` / `rlm_clear_context` - Session context")
    tips.append("- `rlm_stats` / `rlm_sections` - Browse documentation structure")
    tips.append("")

    tips.append("**Tip:** Use `rlm_ask` for quick answers, `rlm_context_query` for full control.")
    tips.append("")
    tips.append("---")

    return "\n".join(tips)


logger = logging.getLogger(__name__)

# Initialize tiktoken encoder (using cl100k_base for GPT-4/Claude compatibility)
_encoding: tiktoken.Encoding | None = None


def get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder (lazy initialization)."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(get_encoder().encode(text))


@dataclass
class Section:
    """A documentation section."""

    id: str
    title: str
    content: str
    start_line: int
    end_line: int
    level: int  # Header level (1-6)


@dataclass
class DocumentationIndex:
    """Index of loaded documentation."""

    files: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    total_chars: int = 0
    # File boundary tracking: maps file path → (start_line_0indexed, end_line_0indexed_exclusive)
    file_boundaries: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Auto-detected ubiquitous keywords (terms appearing in >70% of section titles)
    # These are excluded from distinctive keyword matching to avoid false relevance
    ubiquitous_keywords: set[str] = field(default_factory=set)


@dataclass
class ToolResult:
    """Result from executing an RLM tool."""

    data: Any
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProjectSettings:
    """Project automation settings from dashboard."""

    max_tokens_per_query: int = 4000
    search_mode: str = "hybrid"
    include_summaries: bool = True
    enrich_prompts: bool = False
    auto_inject_context: bool = False
    system_instructions: str | None = None  # Custom instructions to prepend to responses


class RLMEngine:
    """RLM documentation query engine."""

    def __init__(
        self,
        project_id: str,
        plan: Plan = Plan.FREE,
        settings: dict | None = None,
        user_id: str | None = None,
        access_level: str = "EDITOR",
    ):
        """Initialize the engine for a project.

        Args:
            project_id: The project ID
            plan: The user's subscription plan (affects feature access)
            settings: Project automation settings from dashboard (optional)
            user_id: The user ID for access requests (optional)
            access_level: The user's access level for this project (NONE, VIEWER, EDITOR, ADMIN)
        """
        self.project_id = project_id
        self.plan = plan
        self.user_id = user_id
        self.access_level = access_level
        self.index: DocumentationIndex | None = None
        self.session_context: str = ""
        self._chunks_available: bool | None = None  # Cache for chunk availability check
        self._tips_shown_this_session: bool = False  # Track if first-query tips were shown
        # Pass-by-reference chunk cache: chunk_id -> (section, file_path, score)
        # This enables rlm_get_chunk to retrieve full content by ID
        self._chunk_cache: dict[str, tuple[Section, str, float, float, float]] = {}

        # Load settings from dashboard config or use defaults
        if settings:
            self.settings = ProjectSettings(
                max_tokens_per_query=settings.get("max_tokens_per_query", 4000),
                search_mode=settings.get("search_mode", "hybrid"),
                include_summaries=settings.get("include_summaries", True),
                enrich_prompts=settings.get("enrich_prompts", False),
                auto_inject_context=settings.get("auto_inject_context", False),
            )
        else:
            self.settings = ProjectSettings()

    def _generate_chunk_id(self, section_id: str) -> str:
        """Generate a unique chunk ID for pass-by-reference retrieval.

        The chunk ID is a hash of project_id + section_id, ensuring uniqueness
        across projects while being deterministic for the same section.
        """
        key = f"{self.project_id}:{section_id}"
        return f"chunk_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    def _cache_chunk(
        self,
        section: "Section",
        file_path: str,
        score: float,
        keyword_score: float = 0.0,
        semantic_score: float = 0.0,
    ) -> str:
        """Cache a section for later retrieval via rlm_get_chunk.

        Returns the chunk_id for referencing this section.
        """
        chunk_id = self._generate_chunk_id(section.id)
        self._chunk_cache[chunk_id] = (section, file_path, score, keyword_score, semantic_score)
        return chunk_id

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a cached chunk by ID for rlm_get_chunk tool.

        Returns None if the chunk is not found (e.g., from a different session).
        """
        cached = self._chunk_cache.get(chunk_id)
        if not cached:
            return None

        section, file_path, score, kw_score, sem_score = cached
        return {
            "chunk_id": chunk_id,
            "title": section.title,
            "content": section.content,
            "file": file_path,
            "lines": (section.start_line, section.end_line),
            "token_count": count_tokens(section.content),
            "relevance_score": min(score / 100.0, 1.0),
            "keyword_score": kw_score,
            "semantic_score": sem_score,
        }

    async def load_documents(self) -> None:
        """Load and index project documents from database."""
        db = await get_db()

        # Get project info for slug (used for ubiquitous keyword detection)
        project = await db.project.find_unique(where={"id": self.project_id})
        project_slug = project.slug if project else None

        documents = await db.document.find_many(
            where={"projectId": self.project_id},
            order={"path": "asc"},
        )

        self.index = DocumentationIndex()

        for doc in documents:
            self.index.files.append(doc.path)
            doc_lines = doc.content.split("\n")

            # Track line offset for this file
            line_offset = len(self.index.lines)
            self.index.lines.extend(doc_lines)
            self.index.total_chars += len(doc.content)

            # Track file boundaries for rlm_load_document
            self.index.file_boundaries[doc.path] = (line_offset, line_offset + len(doc_lines))

            # Parse sections from this document
            self._parse_sections(doc_lines, line_offset, doc.path)

        # Compute ubiquitous keywords after all sections are indexed
        self._compute_ubiquitous_keywords(project_slug)

    def _parse_sections(self, lines: list[str], offset: int, file_path: str) -> None:
        """Parse markdown sections from lines.

        Orphan sections (headings with < 30 tokens of body content) are merged
        into the previous section to avoid polluting the index with stubs that
        waste context budget slots during retrieval.
        """
        if self.index is None:
            return

        min_section_tokens = 30  # Sections smaller than this get merged upward
        current_section: Section | None = None
        section_content: list[str] = []
        in_code_block = False

        for i, line in enumerate(lines):
            # Track fenced code blocks to avoid parsing comments as headers
            if line.startswith("```") or line.startswith("~~~"):
                in_code_block = not in_code_block

            # Check for markdown headers (only outside code blocks)
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code_block else None

            if header_match:
                # Save previous section (merge orphans into prior section)
                if current_section:
                    current_section.content = "\n".join(section_content)
                    current_section.end_line = offset + i - 1
                    content_tokens = count_tokens(current_section.content)

                    if content_tokens < min_section_tokens and self.index.sections:
                        # Orphan heading — merge into previous section
                        prev = self.index.sections[-1]
                        prev.content += "\n\n" + current_section.content
                        prev.end_line = current_section.end_line
                    else:
                        self.index.sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                section_id = self._generate_section_id(title)

                current_section = Section(
                    id=section_id,
                    title=title,
                    content="",
                    start_line=offset + i + 1,  # 1-indexed
                    end_line=0,
                    level=level,
                )
                section_content = [line]
            elif current_section:
                section_content.append(line)

        # Save last section (also merge if orphan)
        if current_section:
            current_section.content = "\n".join(section_content)
            current_section.end_line = offset + len(lines)
            content_tokens = count_tokens(current_section.content)

            if content_tokens < min_section_tokens and self.index.sections:
                prev = self.index.sections[-1]
                prev.content += "\n\n" + current_section.content
                prev.end_line = current_section.end_line
            else:
                self.index.sections.append(current_section)

    def _generate_section_id(self, title: str) -> str:
        """Generate a unique section ID from title."""
        # Clean the title for use as an ID
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", title)
        clean = re.sub(r"\s+", "_", clean.strip())
        clean = clean.upper()[:20]
        return f"[{clean}]"

    def _compute_ubiquitous_keywords(self, project_slug: str | None = None) -> None:
        """Compute ubiquitous keywords from section titles.

        Ubiquitous keywords are terms that appear in >40% of section titles,
        plus the project name/slug itself (which is always ubiquitous).
        These are excluded from distinctive keyword matching to avoid false relevance.

        Example: In a project called "snipara", the term "snipara" might appear
        in many titles, so it shouldn't count toward relevance.
        """
        if self.index is None or not self.index.sections:
            return

        # Common stop words to always exclude
        stop_words = {
            "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are",
            "with", "from", "by", "on", "at", "as", "be", "this", "that", "it",
            "not", "but", "what", "all", "were", "we", "when", "your", "can",
            "had", "have", "was", "one", "our", "out", "you", "her", "has",
            "how", "use", "using", "used", "overview", "introduction", "guide",
        }

        # Count keyword occurrences across all section titles
        keyword_section_count: dict[str, int] = {}
        total_sections = len(self.index.sections)

        for section in self.index.sections:
            # Extract keywords from title (lowercase, alphanumeric only)
            title_words = re.sub(r"[^a-zA-Z0-9\s]", " ", section.title.lower()).split()
            # Filter short words and stop words
            title_keywords = {w for w in title_words if len(w) > 2 and w not in stop_words}

            for kw in title_keywords:
                keyword_section_count[kw] = keyword_section_count.get(kw, 0) + 1

        # Mark keywords appearing in >40% of sections as ubiquitous
        # Lower threshold (was 70%) because project names often appear in 30-60% of titles
        threshold = 0.40
        ubiquitous = set()
        for kw, count in keyword_section_count.items():
            frequency = count / total_sections
            if frequency >= threshold:
                ubiquitous.add(kw)
                logger.debug(
                    f"Ubiquitous keyword detected: '{kw}' appears in {count}/{total_sections} "
                    f"({frequency:.1%}) section titles"
                )

        # Always add the project slug as ubiquitous (project name appears everywhere)
        if project_slug:
            # Split slug on common separators to get individual words
            slug_words = re.sub(r"[-_]", " ", project_slug.lower()).split()
            for word in slug_words:
                if len(word) > 2 and word not in stop_words:
                    if word not in ubiquitous:
                        logger.debug(f"Adding project slug word as ubiquitous: '{word}'")
                    ubiquitous.add(word)

        self.index.ubiquitous_keywords = ubiquitous

        if ubiquitous:
            logger.info(
                f"Auto-detected {len(ubiquitous)} ubiquitous keywords: {sorted(ubiquitous)}"
            )

    async def load_session_context(self) -> None:
        """Load persisted session context from database."""
        db = await get_db()

        context_entries = await db.sessioncontext.find_many(
            where={
                "projectId": self.project_id,
            },
            order={"createdAt": "asc"},
        )

        if context_entries:
            self.session_context = "\n\n".join(entry.value for entry in context_entries)

    async def _has_precomputed_chunks(self) -> bool:
        """Check if this project has pre-computed chunks for semantic search.

        Uses pgvector-indexed chunks for fast similarity search instead of
        generating embeddings on-the-fly. Result is cached for the engine lifetime.

        Returns:
            True if chunks exist and can be used for semantic search.
        """
        if self._chunks_available is not None:
            return self._chunks_available

        db = await get_db()
        try:
            count = await db.query_raw(
                """
                SELECT COUNT(*) as count FROM document_chunks dc
                JOIN documents d ON dc."documentId" = d.id
                WHERE d."projectId" = $1
                LIMIT 1
                """,
                self.project_id,
            )
            self._chunks_available = count[0]["count"] > 0 if count else False
        except Exception as e:
            logger.warning(f"Failed to check for precomputed chunks: {e}")
            self._chunks_available = False

        return self._chunks_available

    async def _calculate_semantic_scores_from_chunks(
        self, query: str, limit: int = 50, min_similarity: float = 0.3
    ) -> dict[str, float]:
        """Calculate semantic scores using pre-computed chunk embeddings via pgvector.

        This is the fast path for semantic search - uses embeddings that were
        pre-computed during document indexing rather than generating them on-the-fly.

        Args:
            query: The search query string.
            limit: Maximum number of chunks to retrieve.
            min_similarity: Minimum cosine similarity threshold (0-1).

        Returns:
            Dictionary mapping section IDs to their semantic similarity scores (0-1).
        """
        from .services.indexer import DocumentIndexer

        if not self.index:
            return {}

        db = await get_db()
        indexer = DocumentIndexer(db)

        try:
            result = await indexer.search_similar(
                project_id=self.project_id,
                query=query,
                limit=limit,
                min_similarity=min_similarity,
            )

            # Map chunk results back to section IDs by line overlap
            scores: dict[str, float] = {}
            for chunk in result.get("results", []):
                chunk_start = chunk.get("start_line", 0)
                chunk_end = chunk.get("end_line", 0)
                chunk_similarity = chunk.get("similarity", 0.0)

                for section in self.index.sections:
                    # Check if chunk overlaps with section (by line range)
                    if chunk_start <= section.end_line and chunk_end >= section.start_line:
                        # Use max score if section appears in multiple chunks
                        current_score = scores.get(section.id, 0.0)
                        scores[section.id] = max(current_score, chunk_similarity)

            logger.info(
                f"Chunk-based semantic search: {len(result.get('results', []))} chunks, "
                f"{len(scores)} sections scored"
            )
            return scores

        except Exception as e:
            logger.warning(f"Chunk-based semantic search failed: {e}")
            return {}

    async def execute(self, tool: ToolName, params: dict[str, Any]) -> ToolResult:
        """
        Execute an RLM tool.

        Args:
            tool: The tool to execute
            params: Tool parameters

        Returns:
            ToolResult with data and token counts
        """
        # Ensure documents are loaded
        if self.index is None:
            await self.load_documents()
            await self.load_session_context()

        # Route to appropriate handler
        handlers = {
            ToolName.RLM_ASK: self._handle_ask,
            ToolName.RLM_SEARCH: self._handle_search,
            ToolName.RLM_INJECT: self._handle_inject,
            ToolName.RLM_CONTEXT: self._handle_context,
            ToolName.RLM_CLEAR_CONTEXT: self._handle_clear_context,
            ToolName.RLM_STATS: self._handle_stats,
            ToolName.RLM_SECTIONS: self._handle_sections,
            ToolName.RLM_READ: self._handle_read,
            ToolName.RLM_CONTEXT_QUERY: self._handle_context_query,
            # Phase 4.5: Recursive Context Tools
            ToolName.RLM_DECOMPOSE: self._handle_decompose,
            ToolName.RLM_MULTI_QUERY: self._handle_multi_query,
            ToolName.RLM_PLAN: self._handle_plan,
            ToolName.RLM_MULTI_PROJECT_QUERY: self._handle_multi_project_query,
            # Phase 4.6: Summary Storage Tools
            ToolName.RLM_STORE_SUMMARY: self._handle_store_summary,
            ToolName.RLM_GET_SUMMARIES: self._handle_get_summaries,
            ToolName.RLM_DELETE_SUMMARY: self._handle_delete_summary,
            # Phase 7: Shared Context Tools
            ToolName.RLM_SHARED_CONTEXT: self._handle_shared_context,
            ToolName.RLM_LIST_TEMPLATES: self._handle_list_templates,
            ToolName.RLM_GET_TEMPLATE: self._handle_get_template,
            ToolName.RLM_LIST_COLLECTIONS: self._handle_list_collections,
            ToolName.RLM_UPLOAD_SHARED_DOCUMENT: self._handle_upload_shared_document,
            # Phase 8.2: Agent Memory Tools
            ToolName.RLM_REMEMBER: self._handle_remember,
            ToolName.RLM_RECALL: self._handle_recall,
            ToolName.RLM_MEMORIES: self._handle_memories,
            ToolName.RLM_FORGET: self._handle_forget,
            # Phase 9.1: Multi-Agent Swarm Tools
            ToolName.RLM_SWARM_CREATE: self._handle_swarm_create,
            ToolName.RLM_SWARM_JOIN: self._handle_swarm_join,
            ToolName.RLM_CLAIM: self._handle_claim,
            ToolName.RLM_RELEASE: self._handle_release,
            ToolName.RLM_STATE_GET: self._handle_state_get,
            ToolName.RLM_STATE_SET: self._handle_state_set,
            ToolName.RLM_BROADCAST: self._handle_broadcast,
            ToolName.RLM_TASK_CREATE: self._handle_task_create,
            ToolName.RLM_TASK_CLAIM: self._handle_task_claim,
            ToolName.RLM_TASK_COMPLETE: self._handle_task_complete,
            # Phase 10: Document Sync Tools
            ToolName.RLM_UPLOAD_DOCUMENT: self._handle_upload_document,
            ToolName.RLM_SYNC_DOCUMENTS: self._handle_sync_documents,
            ToolName.RLM_SETTINGS: self._handle_settings,
            # Phase 11: Access Control Tools
            ToolName.RLM_REQUEST_ACCESS: self._handle_request_access,
            # Phase 12: RLM Orchestration Tools
            ToolName.RLM_LOAD_DOCUMENT: self._handle_load_document,
            ToolName.RLM_LOAD_PROJECT: self._handle_load_project,
            ToolName.RLM_ORCHESTRATE: self._handle_orchestrate,
            # Phase 13: REPL Context Bridge
            ToolName.RLM_REPL_CONTEXT: self._handle_repl_context,
            # Phase 14: Pass-by-Reference
            ToolName.RLM_GET_CHUNK: self._handle_get_chunk,
        }

        handler = handlers.get(tool)
        if not handler:
            raise ValueError(f"Unknown tool: {tool}")

        # Check tool access level permissions
        access_error = self._check_tool_access(tool)
        if access_error:
            return access_error

        # Check Agents Context plan requirement for agent tools
        if tool in AGENT_TOOLS:
            agents_access_result = await self._check_agents_access(tool)
            if agents_access_result:
                return agents_access_result

        return await handler(params)

    def _check_tool_access(self, tool: ToolName) -> ToolResult | None:
        """
        Check if the user has access to execute the tool based on their access level.

        Returns:
            ToolResult with error if access denied, None if access allowed
        """
        access_level = self.access_level

        # ADMIN has access to all tools
        if access_level == "ADMIN":
            return None

        # EDITOR has access to READ + WRITE tools
        if access_level == "EDITOR":
            if tool in READ_TOOLS or tool in WRITE_TOOLS:
                return None
            return ToolResult(
                data={
                    "error": f"Access denied: {tool.value} requires ADMIN access",
                    "access_level": access_level,
                    "required_level": "ADMIN",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # VIEWER has access to READ tools only
        if access_level == "VIEWER":
            if tool in READ_TOOLS:
                return None
            required = "EDITOR" if tool in WRITE_TOOLS else "ADMIN"
            return ToolResult(
                data={
                    "error": f"Access denied: {tool.value} requires {required} access",
                    "access_level": access_level,
                    "required_level": required,
                },
                input_tokens=0,
                output_tokens=0,
            )

        # NONE access level - only allowed tools
        if access_level == "NONE":
            if tool in NONE_ALLOWED_TOOLS:
                return None
            return ToolResult(
                data={
                    "error": "Access denied to this project. Use rlm_request_access to request access.",
                    "access_level": access_level,
                    "tool": "rlm_request_access",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Unknown access level - default to EDITOR for backward compatibility
        return None

    async def _check_agents_access(self, tool: ToolName) -> ToolResult | None:
        """
        Check if user has proper Agents subscription and Context plan requirement.

        For Agents TEAM/ENTERPRISE plans, the team must also have Context TEAM/ENTERPRISE.
        Personal Agents plans (STARTER/PRO) don't require a Context plan.

        Returns:
            ToolResult with error if access denied, None if access allowed
        """
        try:
            allowed, error, warning = await validate_agents_access(self.project_id)

            if not allowed:
                return ToolResult(
                    data={
                        "error": error or "Agents access denied",
                        "upgrade_required": True,
                        "tool": tool.value,
                        "help": (
                            "Agents TEAM/ENTERPRISE plans require a matching Context plan. "
                            "Personal plans (STARTER/PRO) can be used without a Context subscription. "
                            "Visit your dashboard to upgrade your plan."
                        ),
                    },
                    input_tokens=0,
                    output_tokens=0,
                )

            # If there's a warning (grace period), log it but allow access
            if warning:
                logging.getLogger(__name__).warning(
                    f"Agents access warning for project {self.project_id}: {warning}"
                )

            return None

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error checking agents access for project {self.project_id}: {e}"
            )
            # Fail open - if we can't check, allow access
            return None

    async def _handle_ask(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_ask - query documentation with natural language."""
        question = params.get("question", "")

        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        # Search for relevant sections based on question keywords
        keywords = re.findall(r"\w+", question.lower())
        relevant_sections: list[tuple[Section, int]] = []

        for section in self.index.sections:
            score = 0
            section_text = (section.title + " " + section.content).lower()
            for keyword in keywords:
                if keyword in section_text:
                    score += section_text.count(keyword)
            if score > 0:
                relevant_sections.append((section, score))

        # Sort by relevance and take top results
        relevant_sections.sort(key=lambda x: x[1], reverse=True)
        top_sections = relevant_sections[:5]

        # Build response
        if not top_sections:
            response = f"No relevant documentation found for: {question}"
        else:
            response_parts = [f"**Relevant Documentation for:** {question}\n"]

            if self.session_context:
                response_parts.append(f"**Session Context:**\n{self.session_context}\n")

            for section, score in top_sections:
                response_parts.append(
                    f"\n## {section.title} (lines {section.start_line}-{section.end_line})"
                )
                # Truncate content if too long
                content = (
                    section.content[:2000] + "..."
                    if len(section.content) > 2000
                    else section.content
                )
                response_parts.append(content)

            response = "\n".join(response_parts)

        # Estimate tokens (rough: 4 chars per token)
        input_tokens = len(question) // 4
        output_tokens = len(response) // 4

        return ToolResult(data=response, input_tokens=input_tokens, output_tokens=output_tokens)

    async def _handle_search(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_search - search for patterns."""
        from .config import settings

        pattern = params.get("pattern", "")
        max_results = params.get("max_results", 20)

        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        # Security: Validate pattern length to prevent ReDoS
        if len(pattern) > settings.max_regex_pattern_length:
            return ToolResult(
                data=f"Invalid regex pattern: Pattern too long (max {settings.max_regex_pattern_length} characters)",
                input_tokens=0,
                output_tokens=0,
            )

        # Security: Check for potentially dangerous regex patterns
        dangerous_patterns = [
            r"(.+)+",  # Nested quantifiers
            r"(.*)*",
            r"(.+)*",
            r"(.*)+",
            r"([a-zA-Z]+)*",  # Repeated groups with quantifiers
            r"(a+)+",
            r"(a*)*",
        ]
        pattern_lower = pattern.lower()
        for dangerous in dangerous_patterns:
            if dangerous in pattern_lower or dangerous.replace("a", "[a-z]") in pattern_lower:
                return ToolResult(
                    data="Invalid regex pattern: Pattern contains potentially unsafe constructs (nested quantifiers)",
                    input_tokens=0,
                    output_tokens=0,
                )

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(data=f"Invalid regex pattern: {e}", input_tokens=0, output_tokens=0)

        results: list[dict[str, Any]] = []

        # Execute search with timeout protection using thread pool
        def search_sync():
            """Synchronous search function to run in thread pool."""
            for i, line in enumerate(self.index.lines, start=1):
                # Limit line length to prevent ReDoS on very long lines
                search_line = line[:10000] if len(line) > 10000 else line
                try:
                    if regex.search(search_line):
                        results.append(
                            {
                                "line_number": i,
                                "content": line.strip()[:500],  # Limit content length in results
                            }
                        )
                        if len(results) >= max_results:
                            break
                except Exception:
                    # Skip lines that cause regex issues
                    continue

        try:
            await asyncio.wait_for(
                asyncio.to_thread(search_sync),
                timeout=settings.regex_timeout,
            )
        except TimeoutError:
            logger.warning(f"Regex search timed out for pattern: {pattern[:50]}...")
            return ToolResult(
                data="Search timed out. Try a simpler pattern.",
                input_tokens=len(pattern) // 4,
                output_tokens=10,
            )

        response = {
            "pattern": pattern,
            "total_matches": len(results),
            "results": results,
        }

        input_tokens = len(pattern) // 4
        output_tokens = sum(len(r["content"]) for r in results) // 4

        return ToolResult(data=response, input_tokens=input_tokens, output_tokens=output_tokens)

    async def _handle_inject(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_inject - inject session context."""
        context = params.get("context", "")
        append = params.get("append", False)

        db = await get_db()

        if append:
            # Append to existing context
            self.session_context = f"{self.session_context}\n\n{context}".strip()
        else:
            # Replace context
            self.session_context = context

        # Persist to database
        await db.sessioncontext.upsert(
            where={
                "projectId_key": {
                    "projectId": self.project_id,
                    "key": "session_context",
                }
            },
            data={
                "create": {
                    "projectId": self.project_id,
                    "key": "session_context",
                    "value": self.session_context,
                },
                "update": {
                    "value": self.session_context,
                },
            },
        )

        response = f"Context {'appended' if append else 'set'} successfully ({len(context)} chars)"

        return ToolResult(data=response, input_tokens=len(context) // 4, output_tokens=10)

    async def _handle_context(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_context - show current session context."""
        if not self.session_context:
            response = "No session context set."
        else:
            response = f"**Current Session Context:**\n\n{self.session_context}"

        return ToolResult(data=response, input_tokens=0, output_tokens=len(response) // 4)

    async def _handle_clear_context(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_clear_context - clear session context."""
        db = await get_db()

        # Clear from memory
        self.session_context = ""

        # Clear from database
        await db.sessioncontext.delete_many(where={"projectId": self.project_id})

        return ToolResult(data="Session context cleared.", input_tokens=0, output_tokens=5)

    async def _handle_stats(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_stats - show documentation statistics."""
        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        response = {
            "files_loaded": len(self.index.files),
            "total_lines": len(self.index.lines),
            "total_characters": self.index.total_chars,
            "sections": len(self.index.sections),
            "files": self.index.files,
            "project_id": self.project_id,
        }

        return ToolResult(data=response, input_tokens=0, output_tokens=50)

    async def _handle_sections(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_sections - list documentation sections with pagination."""
        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        # Pagination params
        limit = min(params.get("limit", 50), 500)  # Default 50, max 500
        offset = params.get("offset", 0)
        title_filter = params.get("filter", "").lower()

        # Filter sections by title if filter provided
        all_sections = self.index.sections
        if title_filter:
            all_sections = [s for s in all_sections if title_filter in s.title.lower()]

        total_count = len(all_sections)

        # Apply pagination
        paginated = all_sections[offset : offset + limit]

        sections = [
            SectionInfo(
                id=s.id,
                title=s.title,
                start_line=s.start_line,
                end_line=s.end_line,
            ).model_dump()
            for s in paginated
        ]

        response = {
            "total_sections": total_count,
            "returned": len(sections),
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total_count,
            "sections": sections,
        }

        return ToolResult(data=response, input_tokens=0, output_tokens=len(sections) * 20)

    async def _handle_read(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_read - read specific line range."""
        start_line = params.get("start_line", 1)
        end_line = params.get("end_line", start_line + 50)

        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        # Validate line numbers
        max_line = len(self.index.lines)
        start_line = max(1, min(start_line, max_line))
        end_line = max(start_line, min(end_line, max_line))

        # Get lines (convert to 0-indexed)
        lines = self.index.lines[start_line - 1 : end_line]

        # Format with line numbers
        formatted_lines = [f"{start_line + i:5d}| {line}" for i, line in enumerate(lines)]
        content = "\n".join(formatted_lines)

        response = {
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": len(lines),
            "content": content,
        }

        return ToolResult(data=response, input_tokens=0, output_tokens=len(content) // 4)

    async def _handle_context_query(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_context_query - the main context optimization tool.

        This tool returns the most relevant documentation sections that fit within
        the client's token budget. It uses keyword-based search (with semantic and
        hybrid modes planned for the future).

        **Query Order:** Shared best practices are queried FIRST, then local project
        scope is queried with the remaining token budget.

        Args:
            params: Dict containing:
                - query: The question/query string
                - max_tokens: Token budget (default from dashboard settings or 4000)
                - search_mode: keyword, semantic, or hybrid (default from dashboard settings)
                - include_metadata: Include file/line info (default True)
                - prefer_summaries: Use stored summaries instead of full content (default from dashboard settings)
                - include_shared_context: Include shared best practices (default True for Pro+)
                - shared_context_budget_percent: Percent of budget for shared context (default 30)

        Returns:
            ToolResult with ContextQueryResult containing ranked sections
        """
        original_query = params.get("query", "")
        # Expand abstract queries with concrete keywords for better search recall
        query = _expand_query(original_query)

        # Use dashboard settings as defaults, allow request params to override
        max_tokens = params.get("max_tokens", self.settings.max_tokens_per_query)
        search_mode_str = params.get("search_mode", self.settings.search_mode)
        prefer_summaries = params.get("prefer_summaries", self.settings.include_summaries)
        include_shared_context = params.get("include_shared_context", True)
        shared_context_budget_percent = params.get("shared_context_budget_percent", 30)
        # Pass-by-reference mode: return chunk IDs + previews instead of full content
        # This reduces hallucination by maintaining clear source attribution
        return_references = params.get("return_references", False)

        # Detect complex queries and boost token budget to reduce hallucination
        query_lower = query.lower()

        # Multi-hop queries need the most context (combining info from multiple sections)
        is_multi_hop_query = any(p in query_lower for p in _MULTI_HOP_PATTERNS)
        if is_multi_hop_query and max_tokens < MULTI_HOP_QUERY_MIN_TOKENS:
            logger.info(
                f"Multi-hop query detected, boosting token budget: "
                f"{max_tokens} → {MULTI_HOP_QUERY_MIN_TOKENS}"
            )
            max_tokens = MULTI_HOP_QUERY_MIN_TOKENS

        # Conceptual queries need more context than default
        is_conceptual_query = any(query_lower.startswith(p) for p in _CONCEPTUAL_PREFIXES)
        if is_conceptual_query and max_tokens < CONCEPTUAL_QUERY_MIN_TOKENS:
            logger.info(
                f"Conceptual query detected, boosting token budget: "
                f"{max_tokens} → {CONCEPTUAL_QUERY_MIN_TOKENS}"
            )
            max_tokens = CONCEPTUAL_QUERY_MIN_TOKENS

        # Parse search mode
        try:
            search_mode = SearchMode(search_mode_str)
        except ValueError:
            search_mode = SearchMode.KEYWORD

        # Plan gating: Free users can only use keyword search
        original_search_mode = search_mode
        if search_mode != SearchMode.KEYWORD and self.plan not in SEMANTIC_SEARCH_PLANS:
            logger.info(
                f"Downgrading search mode from {search_mode.value} to keyword "
                f"(plan: {self.plan.value})"
            )
            search_mode = SearchMode.KEYWORD

        # Plan gating: prefer_summaries requires Pro+ plan
        if prefer_summaries and self.plan not in SUMMARY_STORAGE_PLANS:
            prefer_summaries = False

        # Plan gating: shared context requires Pro+ plan
        if include_shared_context and self.plan not in SHARED_CONTEXT_PLANS:
            include_shared_context = False

        # First-query tool tips injection
        # Inject tips only on the first query of this session to help users
        # discover available tools without wasting tokens on every query
        is_first_query = not self._tips_shown_this_session
        if is_first_query:
            self._tips_shown_this_session = True

        # Build effective session context (includes plan-filtered tips on first query only)
        if is_first_query:
            tips = get_first_query_tips(self.plan)
            if self.session_context:
                effective_session_context = f"{tips}\n{self.session_context}"
            else:
                effective_session_context = tips
        else:
            effective_session_context = self.session_context

        if not self.index:
            return ToolResult(
                data=ContextQueryResult(
                    sections=[],
                    total_tokens=0,
                    max_tokens=max_tokens,
                    query=query,
                    search_mode=search_mode,
                ).model_dump(),
                input_tokens=count_tokens(query),
                output_tokens=0,
            )

        # Load summaries if prefer_summaries is enabled
        summaries_by_path: dict[str, dict[str, str]] = {}
        if prefer_summaries:
            summaries_by_path = await self._load_summaries_for_project()

        # Account for session context tokens if present (includes tips on first query)
        session_context_tokens = 0
        session_context_included = False
        remaining_budget = max_tokens

        if effective_session_context:
            session_context_tokens = count_tokens(effective_session_context)
            if session_context_tokens < remaining_budget * 0.2:  # Max 20% for context
                remaining_budget -= session_context_tokens
                session_context_included = True

        # ============ SHARED CONTEXT FIRST ============
        # Query shared best practices FIRST, then local project scope
        shared_context_sections: list[ContextSection] = []
        shared_context_tokens = 0

        if include_shared_context:
            # Allocate budget for shared context (default 30%)
            shared_budget = int(remaining_budget * shared_context_budget_percent / 100)

            try:
                shared_ctx = await load_project_shared_context(self.project_id)

                if shared_ctx.documents:
                    # Allocate shared documents within budget
                    allocated_docs = allocate_shared_context_budget(shared_ctx, shared_budget)

                    # Query-aware filtering: only include shared docs that have
                    # some relevance to the query. MANDATORY docs always included.
                    # Uses title + first 200 chars (intro) to avoid false positives
                    # from large docs that mention common keywords in code examples.
                    # Stop words excluded so "what"/"are" don't create false matches.
                    query_keywords = {
                        w for w in re.findall(r"\w+", query.lower())
                        if len(w) >= 3 and w not in _STOP_WORDS
                    }

                    # Convert shared documents to ContextSection format
                    for doc in allocated_docs:
                        try:
                            cat_enum = DocumentCategoryEnum(doc.category.value)
                        except ValueError:
                            cat_enum = DocumentCategoryEnum.BEST_PRACTICES

                        # MANDATORY docs always included; others need TITLE relevance.
                        # Only checking the title (not body) prevents false positives
                        # from generic keywords ("snipara") appearing in all team docs.
                        if cat_enum != DocumentCategoryEnum.MANDATORY and query_keywords:
                            title_lower = doc.title.lower()
                            hits = sum(
                                1 for kw in query_keywords
                                if kw in title_lower
                                or (_stem_keyword(kw) != kw and _stem_keyword(kw) in title_lower)
                            )
                            # Require 2+ keyword matches for queries with 3+ keywords.
                            # This prevents single-word false positives like "Language Model Tools"
                            # matching any query containing "model" or "tools".
                            # For short queries (1-2 keywords), allow single match to preserve
                            # queries like "authentication" matching "Authentication Standards".
                            min_required = 2 if len(query_keywords) > 2 else 1
                            if hits < min_required:
                                logger.debug(
                                    f"Skipping shared doc '{doc.title}' "
                                    f"({hits} keyword hits < {min_required} required)"
                                )
                                continue

                        shared_context_sections.append(
                            ContextSection(
                                title=f"[{cat_enum.value}] {doc.title}",
                                content=doc.content,
                                file=f"shared:{doc.collection_name}",
                                lines=(0, 0),  # Shared docs don't have line numbers
                                relevance_score=1.0,  # Shared context always high priority
                                token_count=doc.token_count,
                                truncated=False,
                            )
                        )
                        shared_context_tokens += doc.token_count

                    # Update remaining budget for local project scope
                    remaining_budget -= shared_context_tokens
                    logger.info(
                        f"Loaded {len(shared_context_sections)} shared docs "
                        f"({shared_context_tokens} tokens), {remaining_budget} tokens remaining"
                    )
            except Exception as e:
                logger.warning(f"Failed to load shared context: {e}")

        # ============ LOCAL PROJECT SCOPE ============
        # Score and rank sections from local project
        scored_sections = await self._score_sections(query, search_mode)

        # ============ CONFIDENCE-BASED EXPANSION ============
        # If top results have low scores, expand token budget and add grounding.
        # This reduces hallucination by providing more context when uncertain.
        low_confidence = False
        if scored_sections:
            top_score = scored_sections[0][1] if scored_sections else 0
            if top_score < LOW_CONFIDENCE_SCORE_THRESHOLD:
                low_confidence = True
                # Expand budget to get more context
                expanded_budget = int(max_tokens * LOW_CONFIDENCE_TOKEN_MULTIPLIER)
                # Also expand remaining_budget proportionally
                expansion_ratio = expanded_budget / max_tokens
                remaining_budget = int(remaining_budget * expansion_ratio)
                logger.info(
                    f"Low confidence detected (top_score={top_score:.1f} < {LOW_CONFIDENCE_SCORE_THRESHOLD}), "
                    f"expanding token budget: {max_tokens} → {expanded_budget}"
                )
                max_tokens = expanded_budget

        # Filter out shallow sections (just headings with < 30 tokens of content).
        # These waste context budget slots without contributing useful information.
        min_useful_tokens = 30
        scored_sections = [
            (section, score)
            for section, score in scored_sections
            if count_tokens(section.content) >= min_useful_tokens
        ]

        # ---- Minimum relevance score filter ----
        # Filter out low-relevance sections to improve precision.
        # Only include sections with score >= 15% of top score (minimum 5.0 absolute).
        # This prevents irrelevant sections from diluting the context.
        MIN_RELEVANCE_RATIO = 0.15  # 15% of top score
        MIN_ABSOLUTE_SCORE = 5.0    # Minimum absolute score
        if scored_sections:
            top_score = scored_sections[0][1]
            min_score = max(top_score * MIN_RELEVANCE_RATIO, MIN_ABSOLUTE_SCORE)
            before_count = len(scored_sections)
            scored_sections = [
                (s, sc) for s, sc in scored_sections if sc >= min_score
            ]
            if len(scored_sections) < before_count:
                logger.info(
                    f"Relevance filter: removed {before_count - len(scored_sections)} "
                    f"low-scoring sections (min_score={min_score:.1f})"
                )

        # ---- Query specificity filter ----
        # Broad queries (e.g. "architecture") match many sections.  Compute
        # a simple document-frequency ratio: matched / total.  When > 40 %
        # of sections match, raise the score threshold to keep only the
        # highest-signal results instead of flooding the context budget.
        total_sections = len(self.index.sections) if self.index else 1
        match_ratio = len(scored_sections) / max(total_sections, 1)
        if match_ratio > 0.40 and scored_sections:
            # Broad query detected — keep only top 30 % of scored sections
            # by raising the minimum score to the 70th-percentile score.
            cutoff_idx = max(1, int(len(scored_sections) * 0.30))
            cutoff_score = scored_sections[cutoff_idx - 1][1] if cutoff_idx <= len(scored_sections) else 0
            before_count = len(scored_sections)
            scored_sections = [
                (s, sc) for s, sc in scored_sections if sc >= cutoff_score
            ]
            logger.info(
                f"Broad query filter: {match_ratio:.0%} sections matched, "
                f"trimmed {before_count} → {len(scored_sections)} "
                f"(cutoff_score={cutoff_score:.1f})"
            )

        # Cap results to top-N sections.  Returning too many dilutes signal-to-noise
        # for the downstream LLM (e.g., 35 sections for a simple definition query).
        max_sections = 15
        scored_sections = scored_sections[:max_sections]

        # ---- Minimum sections for abstract queries ----
        # Abstract queries need more context to prevent hallucination. If we have
        # fewer sections than the minimum, boost the token budget to ensure we can
        # include at least ABSTRACT_QUERY_MIN_SECTIONS.
        is_abstract = _is_abstract_query(original_query)
        if is_abstract and len(scored_sections) >= ABSTRACT_QUERY_MIN_SECTIONS:
            # Estimate tokens needed for minimum sections
            avg_tokens_per_section = 400  # Conservative estimate
            min_budget_needed = ABSTRACT_QUERY_MIN_SECTIONS * avg_tokens_per_section
            if max_tokens < min_budget_needed:
                logger.info(
                    f"Abstract query detected, boosting budget for min {ABSTRACT_QUERY_MIN_SECTIONS} sections: "
                    f"{max_tokens} → {min_budget_needed}"
                )
                max_tokens = min_budget_needed
                remaining_budget = max(remaining_budget, min_budget_needed - shared_context_tokens)

        # ---- Title + content dedup ----
        # Two-pass deduplication:
        # 1. Title-word overlap (fast, catches near-identical headings)
        # 2. Content-word overlap (catches sections with different titles
        #    but substantially overlapping body text — common for broad
        #    queries like "architecture" where subsections repeat parent
        #    content)
        def _title_key(title: str) -> frozenset[str]:
            """Normalise a section title to a bag of lowercase alpha words."""
            return frozenset(re.findall(r"[a-z]+", title.lower()))

        def _content_shingle(content: str) -> frozenset[str]:
            """Extract a word-set 'fingerprint' from the first 500 chars."""
            words = re.findall(r"[a-z]{3,}", content[:500].lower())
            return frozenset(words)

        seen_title_keys: dict[frozenset[str], float] = {}
        seen_content_keys: list[frozenset[str]] = []
        deduped: list[tuple[Section, float]] = []
        for section, score in scored_sections:
            tkey = _title_key(section.title)
            # Pass 1: title similarity (>= 85 % word overlap)
            title_dup = False
            for existing_key in seen_title_keys:
                if not tkey or not existing_key:
                    continue
                overlap = len(tkey & existing_key) / min(len(tkey), len(existing_key))
                if overlap >= 0.85:
                    title_dup = True
                    break

            # Pass 2: content similarity (>= 60 % word overlap)
            content_dup = False
            if not title_dup:
                ckey = _content_shingle(section.content)
                if ckey and len(ckey) >= 5:  # Only check substantial sections
                    for existing_ckey in seen_content_keys:
                        if not existing_ckey:
                            continue
                        c_overlap = len(ckey & existing_ckey) / min(len(ckey), len(existing_ckey))
                        if c_overlap >= 0.60:
                            content_dup = True
                            logger.debug(
                                f"Content dedup: '{section.title}' overlaps "
                                f"with existing section ({c_overlap:.0%})"
                            )
                            break

            if not title_dup and not content_dup:
                deduped.append((section, score))
                seen_title_keys[tkey] = score
                seen_content_keys.append(_content_shingle(section.content))
        scored_sections = deduped

        # Greedy selection: add sections until budget is exceeded
        selected_sections: list[ContextSection] = []
        suggestions: list[str] = []
        total_tokens = session_context_tokens if session_context_included else 0
        total_tokens += shared_context_tokens  # Include shared context tokens
        summaries_used = 0

        for section, score in scored_sections:
            file_path = self._find_file_for_section(section)

            # Check if we should use a summary instead
            content_to_use = section.content
            used_summary = False

            if prefer_summaries and file_path in summaries_by_path:
                # Try to find a matching summary for this section
                section_summaries = summaries_by_path[file_path]
                # Prefer concise summary, then detailed
                for summary_type in ["concise", "detailed", "technical"]:
                    if summary_type in section_summaries:
                        summary_content = section_summaries[summary_type]
                        summary_tokens = count_tokens(summary_content)
                        section_tokens = count_tokens(section.content)
                        # Use summary if it's significantly smaller
                        if summary_tokens < section_tokens * 0.5:
                            content_to_use = f"[Summary ({summary_type})]\n{summary_content}"
                            used_summary = True
                            break

            section_tokens = count_tokens(content_to_use)

            if total_tokens + section_tokens <= remaining_budget:
                # Section fits - add it fully
                selected_sections.append(
                    ContextSection(
                        title=section.title,
                        content=content_to_use,
                        file=file_path,
                        lines=(section.start_line, section.end_line),
                        relevance_score=min(score / 100.0, 1.0),  # Normalize score
                        token_count=section_tokens,
                        truncated=False,
                    )
                )
                total_tokens += section_tokens
                if used_summary:
                    summaries_used += 1
            elif total_tokens < remaining_budget:
                # Partial fit - truncate to fit remaining budget
                remaining = remaining_budget - total_tokens
                truncated_content = self._smart_truncate(content_to_use, remaining)
                truncated_tokens = count_tokens(truncated_content)

                if truncated_tokens >= 50:  # Only include if meaningful
                    selected_sections.append(
                        ContextSection(
                            title=section.title,
                            content=truncated_content,
                            file=file_path,
                            lines=(section.start_line, section.end_line),
                            relevance_score=min(score / 100.0, 1.0),
                            token_count=truncated_tokens,
                            truncated=True,
                        )
                    )
                    total_tokens += truncated_tokens
                    if used_summary:
                        summaries_used += 1
                break
            else:
                # No more budget - add to suggestions
                if len(suggestions) < 5:
                    suggestions.append(f"{section.title} (score: {score:.1f})")

        # ---- Hard cap: enforce max_tokens guarantee ----
        # The greedy loop above *should* stay within budget, but shared
        # context allocation, session context, and rounding during truncation
        # can push total_tokens slightly over.  Drop the lowest-scoring local
        # sections until we're within budget.
        if total_tokens > max_tokens and selected_sections:
            logger.warning(
                f"Token budget exceeded ({total_tokens}/{max_tokens}), "
                f"trimming lowest-scoring local sections"
            )
            while total_tokens > max_tokens and selected_sections:
                removed = selected_sections.pop()
                total_tokens -= removed.token_count

        # Build result - shared context sections come FIRST
        all_sections = shared_context_sections + selected_sections
        search_mode_downgraded = original_search_mode != search_mode

        # Include system instructions (custom from project or default)
        instructions = self.settings.system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS

        # ALWAYS add grounding instructions - core promise: "Anti-Hallucination by Design"
        # Use strict mode for low confidence, standard mode otherwise
        # Only add grounding instructions for low-confidence results
        # Adding grounding to all queries was found to increase hallucination
        # by making the LLM too cautious (refusing to synthesize from context)
        if low_confidence:
            instructions = instructions + STRICT_GROUNDING_INSTRUCTIONS
            logger.info("Added grounding instructions due to low confidence results")

        # Smart routing: analyze query and recommend execution mode
        routing_decision = route_query(query, context_tokens=total_tokens)

        # Pass-by-reference mode: convert sections to chunk references
        section_refs: list[ContextSectionRef] = []
        preview_tokens = 0

        if return_references:
            # Build chunk references instead of returning full content
            # This reduces hallucination by:
            # 1. Limiting what the LLM sees (previews only)
            # 2. Requiring explicit chunk retrieval for details
            # 3. Maintaining clear source attribution via chunk_id
            for section in all_sections:
                # Generate preview (first ~100 chars)
                preview = section.content[:100].strip()
                if len(section.content) > 100:
                    preview += "..."

                # Cache the full content for later retrieval
                chunk_id = self._cache_chunk(
                    section=Section(
                        id=hashlib.md5(f"{section.file}:{section.title}".encode()).hexdigest()[:12],
                        title=section.title,
                        content=section.content,
                        start_line=section.lines[0],
                        end_line=section.lines[1],
                        level=1,
                        parent_titles=[],
                    ),
                    file_path=section.file,
                    score=section.relevance_score * 100,
                    keyword_score=0.0,  # Not tracked in ContextSection
                    semantic_score=0.0,
                )

                section_refs.append(ContextSectionRef(
                    chunk_id=chunk_id,
                    title=section.title,
                    preview=preview,
                    file=section.file,
                    lines=section.lines,
                    relevance_score=section.relevance_score,
                    token_count=section.token_count,
                    keyword_score=0.0,
                    semantic_score=0.0,
                ))
                preview_tokens += count_tokens(preview)

            logger.info(
                f"Pass-by-reference mode: {len(section_refs)} chunks cached, "
                f"{preview_tokens} preview tokens (vs {total_tokens} full tokens)"
            )

        result = ContextQueryResult(
            sections=[] if return_references else all_sections,
            section_refs=section_refs if return_references else [],
            references_mode=return_references,
            total_tokens=preview_tokens if return_references else total_tokens,
            max_tokens=max_tokens,
            query=query,
            search_mode=search_mode,
            search_mode_downgraded=search_mode_downgraded,
            session_context_included=session_context_included,
            suggestions=suggestions,
            summaries_used=summaries_used,
            system_instructions=instructions,
            shared_context_included=len(shared_context_sections) > 0,
            shared_context_tokens=shared_context_tokens,
            first_query_tips_included=is_first_query and session_context_included,
            # Smart routing hints
            routing_recommendation=routing_decision.mode.value,
            routing_confidence=routing_decision.confidence,
            routing_reason=routing_decision.reason,
            query_complexity=routing_decision.complexity.value,
        )

        # Calculate actual token usage for billing
        input_tokens = count_tokens(query)
        output_tokens = total_tokens

        return ToolResult(
            data=result.model_dump(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _load_summaries_for_project(self) -> dict[str, dict[str, str]]:
        """
        Load all summaries for the project, organized by document path and type.

        Returns:
            Dict mapping document_path -> {summary_type -> summary_content}
        """
        db = await get_db()

        summaries = await db.documentsummary.find_many(
            where={"projectId": self.project_id},
            include={"document": True},
        )

        result: dict[str, dict[str, str]] = {}
        for s in summaries:
            if s.document:
                path = s.document.path
                if path not in result:
                    result[path] = {}
                result[path][s.summaryType] = s.summary

        return result

    async def _score_sections(
        self, query: str, search_mode: SearchMode
    ) -> list[tuple[Section, float]]:
        """
        Score sections by relevance to the query.

        Supports three search modes:
        - KEYWORD: Traditional keyword matching
        - SEMANTIC: Embedding-based similarity search (uses pre-computed chunks when available)
        - HYBRID: Combined keyword + semantic scoring

        Uses pre-computed chunks with pgvector for fast semantic search when available,
        falling back to on-the-fly embedding generation if chunks don't exist.
        """
        if not self.index:
            return []

        keywords = [w for w in re.findall(r"\w+", query.lower()) if w not in _STOP_WORDS]
        scored: list[tuple[Section, float]] = []

        # Detect list/enumeration queries for numbered section boosting
        is_list_query = _is_list_query(query)
        is_planned_query = any(
            term in query.lower() for term in ("next", "planned", "upcoming", "todo", "write")
        )
        if is_list_query:
            logger.info(f"List query detected: '{query[:60]}...' - boosting numbered sections")

        # Calculate keyword scores for all sections (always in-memory, fast)
        keyword_scores: dict[str, float] = {}
        for section in self.index.sections:
            base_score = self._calculate_keyword_score(section, keywords)

            # Boost numbered sections for list queries (Fix #1: Query-intent detection)
            # E.g., "Article #3" should rank higher than "Example Article Structure"
            # for queries like "what are the next articles to write"
            if is_list_query and base_score > 0:
                if _is_numbered_section(section.title, section.content):
                    base_score *= 2.0  # 2x boost for numbered sections
                    logger.debug(f"Numbered section boost: '{section.title}' → {base_score:.1f}")

                # Boost sections with planned/unpublished markers (Fix #3: Semantic grounding)
                # E.g., sections with "📝 Unpublished" should rank higher for "next articles"
                if is_planned_query and _has_planned_content_markers(section.content):
                    base_score *= 1.5  # 1.5x boost for planned content
                    logger.debug(f"Planned content boost: '{section.title}' → {base_score:.1f}")

            keyword_scores[section.id] = base_score

        # Handle different search modes
        if search_mode == SearchMode.KEYWORD:
            # Pure keyword search with relevance filtering for multi-keyword queries.
            # Problem: Project-specific terms appear everywhere, so irrelevant sections rank highly.
            # Solution: For 3+ keyword queries, require either:
            #   - 1+ NON-ubiquitous keyword in title (distinctive match), OR
            #   - High content score (>50) indicating strong relevance
            # Ubiquitous keywords (auto-detected from section titles) don't count for title matching.
            ubiquitous_keywords = self.index.ubiquitous_keywords
            logger.debug(
                f"Keyword search: ubiquitous_keywords={sorted(ubiquitous_keywords) if ubiquitous_keywords else '(empty)'}"
            )
            for section in self.index.sections:
                score = keyword_scores[section.id]
                if score <= 0:
                    continue

                # Apply title relevance filter for multi-keyword queries
                if len(keywords) >= 3:
                    title_lower = section.title.lower()
                    # Count only non-ubiquitous keywords in title
                    distinctive_title_hits = sum(
                        1 for kw in keywords
                        if kw not in ubiquitous_keywords and (
                            kw in title_lower
                            or (_stem_keyword(kw) != kw and _stem_keyword(kw) in title_lower)
                        )
                    )
                    if distinctive_title_hits >= 1:
                        pass  # Has distinctive keyword in title - keep
                    elif score > 50:
                        pass  # High content relevance - keep
                    else:
                        # No distinctive title match and low content score - skip
                        logger.debug(
                            f"Keyword search: Skipping '{section.title}' "
                            f"(distinctive_title_hits={distinctive_title_hits}, kw_score={score:.1f})"
                        )
                        continue

                scored.append((section, score))

        elif search_mode == SearchMode.SEMANTIC:
            # Pure semantic search - try pre-computed chunks first
            use_chunks = await self._has_precomputed_chunks()

            if use_chunks:
                semantic_scores = await self._calculate_semantic_scores_from_chunks(query)
                logger.info(
                    f"Using pre-computed chunks for semantic search (project: {self.project_id})"
                )
            else:
                semantic_scores = await self._calculate_semantic_scores(query)
                logger.info(
                    f"Using on-the-fly embedding (no chunks for project: {self.project_id})"
                )

            for section in self.index.sections:
                score = semantic_scores.get(section.id, 0.0) * 100  # Scale to 0-100
                if score > 10:  # Minimum similarity threshold
                    scored.append((section, score))

        elif search_mode == SearchMode.HYBRID:
            # Combined keyword + semantic search via Reciprocal Rank Fusion
            # with adaptive weights based on query characteristics.

            # Log ubiquitous keywords for debugging relevance filter
            ubiq_kws = self.index.ubiquitous_keywords
            logger.debug(
                f"Hybrid search: ubiquitous_keywords={sorted(ubiq_kws) if ubiq_kws else '(empty)'}, "
                f"query_keywords={keywords[:5]}"
            )

            use_chunks = await self._has_precomputed_chunks()

            if use_chunks:
                semantic_scores = await self._calculate_semantic_scores_from_chunks(query)
                logger.info(
                    f"Using pre-computed chunks for hybrid search (project: {self.project_id})"
                )
            else:
                # Pre-filter to top keyword candidates to avoid embedding all sections.
                # Two-stage filter: (1) score drop-off threshold eliminates noise from
                # broad queries like "search modes" where hundreds of sections match
                # weakly, (2) hard cap of 30 as safety net.
                sorted_kw = sorted(
                    ((sid, sc) for sid, sc in keyword_scores.items() if sc > 0),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if sorted_kw:
                    top_score = sorted_kw[0][1]
                    # Drop sections scoring < 10% of the top score (min threshold 2.0)
                    threshold = max(top_score * 0.10, 2.0)
                    top_keyword_ids = {
                        sid for sid, sc in sorted_kw[:30] if sc >= threshold
                    }
                else:
                    top_keyword_ids = set()

                semantic_scores = await self._calculate_semantic_scores(
                    query, candidate_ids=top_keyword_ids if top_keyword_ids else None
                )
                logger.info(
                    f"Using on-the-fly embedding for hybrid search "
                    f"({len(top_keyword_ids)} keyword candidates, "
                    f"threshold={threshold:.1f}, project: {self.project_id})"
                    if sorted_kw else
                    f"Using on-the-fly embedding for hybrid search "
                    f"(0 keyword candidates, project: {self.project_id})"
                )

            # Adaptive weights: classify query to pick keyword/semantic balance
            kw_weight, sem_weight = self._classify_query_weights(query, keyword_scores)
            logger.info(
                f"Hybrid adaptive weights: kw={kw_weight:.2f} sem={sem_weight:.2f} "
                f"for query: {query[:80]}"
            )

            # Reciprocal Rank Fusion (rank-based, immune to score-magnitude mismatch)
            rrf_results = self._rrf_fusion(
                keyword_scores=keyword_scores,
                semantic_scores=semantic_scores,
                keyword_weight=kw_weight,
                semantic_weight=sem_weight,
            )

            # Apply graded normalization for clear rank separation
            # Top = 100, then exponential decay (100, 88, 77, 68, 60...)
            # This creates clearer "ground truth" distinction vs similar raw scores
            graded_results = self._normalize_scores_graded(rrf_results)

            section_map = {s.id: s for s in self.index.sections}
            for sid, graded_score in graded_results:
                section = section_map.get(sid)
                if not section:
                    continue

                # Mild boost when both signals strongly agree
                final_score = graded_score
                kw = keyword_scores.get(sid, 0)
                sem = semantic_scores.get(sid, 0.0) * 100
                if kw > 5 and sem > 30:
                    final_score *= 1.10  # Smaller boost since graded already separates

                # Filter out sections with insufficient relevance for multi-keyword queries.
                # Problem: Project-specific terms appear everywhere, so sections like "VS Code Extension"
                # rank highly for queries like "What is Snipara's value proposition?"
                # Solution: For 3+ keyword queries, require either:
                #   - 1+ NON-ubiquitous keyword in title (distinctive match), OR
                #   - High keyword score (>50) OR high semantic score (>40)
                # Ubiquitous keywords are auto-detected from section titles at index time.
                ubiquitous_keywords = self.index.ubiquitous_keywords
                if len(keywords) >= 3:
                    title_lower = section.title.lower()
                    distinctive_title_hits = sum(
                        1 for kw_term in keywords
                        if kw_term not in ubiquitous_keywords and (
                            kw_term in title_lower
                            or (_stem_keyword(kw_term) != kw_term and _stem_keyword(kw_term) in title_lower)
                        )
                    )
                    if distinctive_title_hits >= 1:
                        pass  # Has distinctive keyword in title - keep
                    elif kw > 50:
                        pass  # High keyword relevance - keep
                    elif sem > 55 and kw > 3:
                        pass  # Strong semantic + some keyword presence - keep
                    else:
                        logger.debug(
                            f"Filtering out '{section.title}' (distinctive_hits={distinctive_title_hits}, "
                            f"kw={kw:.1f}, sem={sem:.1f})"
                        )
                        continue

                if final_score > 3:  # ~3 % of max score
                    scored.append((section, final_score))

        # Apply internal path penalty before sorting
        # Sections from internal/debug paths (like .claude/commands/debug.md)
        # get their scores reduced to prevent them from ranking above real docs.
        if self.index and scored:
            penalized: list[tuple[Section, float]] = []
            for section, score in scored:
                file_path = self._find_file_for_section(section)
                if _is_internal_path(file_path):
                    penalized_score = score * _INTERNAL_PATH_PENALTY
                    logger.debug(
                        f"Penalizing internal file section: '{section.title}' "
                        f"({file_path}) {score:.1f} → {penalized_score:.1f}"
                    )
                    penalized.append((section, penalized_score))
                else:
                    penalized.append((section, score))
            scored = penalized

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _calculate_semantic_scores(
        self,
        query: str,
        candidate_ids: set[str] | None = None,
        max_sections: int = 30,
    ) -> dict[str, float]:
        """
        Calculate semantic similarity scores for sections (on-the-fly fallback path).

        Uses the *light* embedding model (bge-small-en-v1.5, 384 dims) which is ~10x
        faster than bge-large on CPU. This path is only used when pre-computed pgvector
        chunks are not available. Both query and section embeddings are computed fresh
        so dimension mismatch with pgvector (1024 dims) is not an issue.

        Args:
            query: The search query string.
            candidate_ids: If provided, only embed these section IDs (e.g. top keyword hits).
            max_sections: Hard cap on sections to embed (default 30). bge-small-en-v1.5
                takes ~0.3s per text on Railway CPU; 30 sections ≈ 3-5s.
        """
        if not self.index or not self.index.sections:
            return {}

        try:
            embeddings_service = get_light_embeddings_service()

            # Filter to candidate sections if provided, otherwise use all
            if candidate_ids is not None:
                sections = [s for s in self.index.sections if s.id in candidate_ids]
            else:
                sections = list(self.index.sections)

            # Hard cap to prevent batch embedding timeout on large projects
            if len(sections) > max_sections:
                logger.info(
                    f"Capping on-the-fly embedding from {len(sections)} to {max_sections} sections"
                )
                sections = sections[:max_sections]

            # Generate query embedding (async to avoid blocking event loop)
            query_embedding = await embeddings_service.embed_text_async(query)

            # Generate section embeddings (title + truncated content)
            # Using 120 chars to reduce tokenization cost on CPU.
            # Title carries the primary semantic signal; content adds context.
            section_texts = [
                f"{s.title}\n{s.content[:120]}"
                for s in sections
            ]
            section_embeddings = await embeddings_service.embed_texts_async(section_texts)

            # Calculate similarities
            similarities = embeddings_service.cosine_similarity(query_embedding, section_embeddings)

            # Map to section IDs
            return {
                section.id: similarity
                for section, similarity in zip(sections, similarities)
            }
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to empty scores: {e}")
            return {}

    def _calculate_keyword_score(self, section: Section, keywords: list[str]) -> float:
        """
        Calculate keyword relevance score for a section.

        Scoring factors:
        - Title matches weighted 5x (not length-normalized)
        - Content matches weighted 1x (BM25-style length normalization)
        - Section level bonus (higher level = more important)
        - Title keyword coverage boost (multi-keyword title match)
        """
        score = 0.0
        title_lower = section.title.lower()
        content_lower = section.content.lower()

        # BM25-style length normalization for content scoring.
        # Prevents long sections from dominating via raw keyword counts.
        # avgdl ~2000 chars is a reasonable average section length.
        content_length = len(content_lower)
        length_norm = 1.0 / (1.0 + 0.75 * (content_length / 2000.0 - 1.0))
        length_norm = max(length_norm, 0.15)  # Floor to avoid near-zero

        title_keyword_hits = 0

        for keyword in keywords:
            if len(keyword) < 2:  # Skip very short words
                continue

            stem = _stem_keyword(keyword)

            # Title matches — reduced weight for generic terms
            title_count = title_lower.count(keyword)
            # Fall back to stem match for morphological variants
            # e.g. "prices" (stem "pric") matches title containing "pricing"
            if title_count == 0 and stem != keyword:
                title_count = title_lower.count(stem)
            if title_count > 0:
                title_keyword_hits += 1
                # Generic terms get 1.5x weight, specific terms get 5x
                # This prevents "Snipara tools not available" ranking above
                # actual tool documentation for "What tools does Snipara expose?"
                if keyword in _GENERIC_TITLE_TERMS or stem in _GENERIC_TITLE_TERMS:
                    score += title_count * 1.5
                else:
                    score += title_count * 5.0

            # Content matches (length-normalized)
            content_count = content_lower.count(keyword)
            if content_count == 0 and stem != keyword:
                content_count = content_lower.count(stem)
            score += content_count * length_norm

        # Bonus for higher-level sections (h1, h2 more important)
        level_bonus = max(0, 4 - section.level) * 0.5
        score += level_bonus if score > 0 else 0

        # Title keyword coverage boost: when multiple query keywords appear
        # in the section title, this section is likely a direct topical match.
        # Apply multiplicative boost proportional to the number of title hits.
        if title_keyword_hits >= 2:
            coverage_boost = 1.0 + title_keyword_hits * 2.0
            score *= coverage_boost

        # Exact phrase match bonus: if the entire query (or a significant portion)
        # appears verbatim in the title, this is very likely the right section.
        # This dramatically improves NDCG for specific queries.
        query_words = [w for w in keywords if len(w) >= 3]
        if len(query_words) >= 2:
            query_phrase = " ".join(query_words[:4])  # First 4 significant words
            if query_phrase.lower() in title_lower:
                score *= 3.0  # 3x bonus for exact phrase in title
                logger.debug(f"Exact phrase match in title: '{query_phrase}' → '{section.title}'")

        return score

    # ------------------------------------------------------------------
    # Adaptive Hybrid Search helpers
    # ------------------------------------------------------------------

    def _classify_query_weights(
        self,
        query: str,
        keyword_scores: dict[str, float],
    ) -> tuple[float, float]:
        """Return (keyword_weight, semantic_weight) adapted to the query.

        Heuristics (no LLM call, ~0 ms overhead):

        1. **Strong keyword signal** – the top keyword score is well above the
           median, meaning there's likely an exact or near-exact title match.
           Combined with specific technical terms ➜ keyword-heavy (70/30).

        2. **Conceptual query** – starts with "how does", "why", "explain" etc.
           and doesn't contain structured-data terms ➜ semantic-heavy (30/70).

        3. **Default** – balanced (50/50).
        """
        query_lower = query.lower()
        query_words = {w for w in re.findall(r"\w+", query_lower) if len(w) > 2}

        # Signal 1: strong keyword confidence
        kw_values = [v for v in keyword_scores.values() if v > 0]
        strong_keyword = False
        if kw_values:
            top_kw = max(kw_values)
            # Top score is at least 3× the median – very confident match
            median_kw = sorted(kw_values)[len(kw_values) // 2] if kw_values else 0
            strong_keyword = top_kw > 15 and (median_kw == 0 or top_kw / median_kw >= 3)

        # Signal 2: specific / structured-data terms in the query
        # Also check stemmed variants so "prices" matches "price" in the set
        stemmed_words = {_stem_keyword(w) for w in query_words}
        has_specific = bool(
            (query_words | stemmed_words) & _SPECIFIC_QUERY_TERMS
        )

        # Signal 3: conceptual query pattern
        is_conceptual = any(query_lower.startswith(p) for p in _CONCEPTUAL_PREFIXES)

        # Priority: specific keyword matches first, then conceptual patterns
        if strong_keyword and has_specific:
            return _HYBRID_KEYWORD_HEAVY
        if strong_keyword:
            return _HYBRID_BALANCED  # Don't go full keyword-heavy without specific terms
        if is_conceptual:
            return _HYBRID_SEMANTIC_HEAVY
        return _HYBRID_BALANCED

    @staticmethod
    def _rrf_fusion(
        keyword_scores: dict[str, float],
        semantic_scores: dict[str, float],
        keyword_weight: float,
        semantic_weight: float,
        k: int = _RRF_K,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of keyword and semantic rankings.

        RRF score for document *d*::

            rrf(d) = w_kw / (k + rank_kw(d)) + w_sem / (k + rank_sem(d))

        Unlike weighted-score averaging, RRF is **rank-based** so it is
        robust to score-magnitude mismatches between keyword and semantic
        signals (the root cause of the hybrid regression on ``tech_stack``
        and ``core_value_prop``).

        Sections absent from a ranking receive a pessimistic rank equal to
        ``len(ranking) + 1``.
        """
        # Build keyword ranking (descending score)
        kw_ranked = sorted(
            ((sid, sc) for sid, sc in keyword_scores.items() if sc > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        kw_rank: dict[str, int] = {sid: rank for rank, (sid, _) in enumerate(kw_ranked, start=1)}
        default_kw_rank = len(kw_ranked) + 1

        # Build semantic ranking (descending similarity)
        sem_ranked = sorted(
            ((sid, sc) for sid, sc in semantic_scores.items() if sc > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        sem_rank: dict[str, int] = {sid: rank for rank, (sid, _) in enumerate(sem_ranked, start=1)}
        default_sem_rank = len(sem_ranked) + 1

        # Union of all section IDs present in either ranking
        all_ids = set(kw_rank) | set(sem_rank)

        rrf_scores: list[tuple[str, float]] = []
        for sid in all_ids:
            rk = kw_rank.get(sid, default_kw_rank)
            rs = sem_rank.get(sid, default_sem_rank)
            score = keyword_weight / (k + rk) + semantic_weight / (k + rs)
            rrf_scores.append((sid, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores

    def _normalize_scores_graded(
        self, scores: list[tuple[str, float]], decay_factor: float = 0.94
    ) -> list[tuple[str, float]]:
        """Normalize scores to 0-100 with clear rank separation.

        Creates a graded scoring where:
        - Rank 1 (ground truth) = 100
        - Each subsequent rank decays by decay_factor
        - But also considers actual score gaps

        This produces clearer separation than raw score normalization:
        - Raw: [0.05, 0.048, 0.045, 0.042] → [100, 96, 90, 84] (too similar)
        - Graded: [0.05, 0.048, 0.045, 0.042] → [100, 94, 88, 83] (clear hierarchy)

        Args:
            scores: List of (id, raw_score) tuples, already sorted descending
            decay_factor: Base decay per rank (default 0.94 = ~6% drop per rank for better recall)

        Returns:
            List of (id, graded_score) with clear separation
        """
        if not scores:
            return []

        # Top score gets 100
        top_score = scores[0][1] if scores[0][1] > 0 else 1.0
        result = [(scores[0][0], 100.0)]

        for i, (sid, raw) in enumerate(scores[1:], start=1):
            # Combine rank-based decay with score-ratio decay
            # rank_factor: exponential decay based on position (0.94^rank)
            # score_factor: how close is this score to the top? (raw/top)
            rank_factor = decay_factor ** i
            score_factor = raw / top_score if top_score > 0 else 0

            # Weighted combination: 40% rank-based, 60% score-based (favor raw scores for better recall)
            # Higher score_weight keeps more relevant sections that may have lower ranks
            graded = 100 * (0.4 * rank_factor + 0.6 * score_factor)

            # Floor at 1 (never 0 unless truly irrelevant)
            result.append((sid, max(round(graded, 1), 1.0)))

        return result

    def _find_file_for_section(self, section: Section) -> str:
        """Find which file a section belongs to based on line numbers.

        Uses file_boundaries for precise lookup: section line numbers are matched
        against the (start, end) ranges tracked during indexing.
        """
        if not self.index:
            return "unknown"

        # Use file_boundaries for precise mapping (line numbers are 1-indexed in sections)
        for file_path, (start, end) in self.index.file_boundaries.items():
            # start/end are 0-indexed, section lines are 1-indexed
            if start + 1 <= section.start_line <= end:
                return file_path

        return self.index.files[-1] if self.index.files else "unknown"

    def _smart_truncate(self, content: str, max_tokens: int) -> str:
        """
        Truncate content to fit within token budget at sentence boundaries.

        Tries to cut at the end of a sentence to preserve meaning.
        """
        if count_tokens(content) <= max_tokens:
            return content

        # Binary search for the right length
        encoder = get_encoder()
        tokens = encoder.encode(content)

        if len(tokens) <= max_tokens:
            return content

        # Truncate tokens and decode
        truncated_tokens = tokens[:max_tokens]
        truncated = encoder.decode(truncated_tokens)

        # Try to end at a sentence boundary
        sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
        best_end = -1

        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > best_end and pos > len(truncated) * 0.5:  # At least 50% of content
                best_end = pos + len(ending)

        if best_end > 0:
            return truncated[:best_end].rstrip() + "..."

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > len(truncated) * 0.7:
            return truncated[:last_space].rstrip() + "..."

        return truncated.rstrip() + "..."

    # ============ PHASE 4.5: RECURSIVE CONTEXT HANDLERS ============

    async def _handle_decompose(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_decompose - decompose complex queries into sub-queries.

        This tool breaks down a complex question into smaller, focused sub-queries
        that can be executed independently. No LLM required - uses NLP techniques.

        Args:
            params: Dict containing:
                - query: The complex question to decompose
                - max_depth: Maximum recursion depth (default 2)
                - strategy: Decomposition strategy (default auto)
                - hints: Optional hints to guide decomposition

        Returns:
            ToolResult with DecomposeResult containing sub-queries and dependencies
        """
        query = params.get("query", "")
        _max_depth = params.get("max_depth", 2)  # noqa: F841 — reserved for future use
        strategy_str = params.get("strategy", "auto")
        hints = params.get("hints", [])

        # Plan gating
        if self.plan not in RECURSIVE_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_decompose requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=count_tokens(query),
                output_tokens=0,
            )

        # Parse strategy
        try:
            strategy = DecomposeStrategy(strategy_str)
        except ValueError:
            strategy = DecomposeStrategy.AUTO

        if not self.index:
            return ToolResult(
                data=DecomposeResult(
                    original_query=query,
                    sub_queries=[],
                    dependencies=[],
                    suggested_sequence=[],
                    total_estimated_tokens=0,
                    strategy_used=strategy,
                ).model_dump(),
                input_tokens=count_tokens(query),
                output_tokens=0,
            )

        # Extract key terms from the query
        chunker = get_chunker()
        key_terms = chunker.extract_key_terms(query)

        # Include hints as additional terms
        if hints:
            for hint in hints:
                hint_terms = chunker.extract_key_terms(hint)
                key_terms.extend(hint_terms)

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)

        # Find sections matching each term
        term_sections: dict[str, list[Section]] = {}
        for term in unique_terms[:10]:  # Limit to top 10 terms
            matching = self._find_sections_for_term(term)
            if matching:
                term_sections[term] = matching

        # Build sub-queries from terms with matching sections.
        # Use phrase-level queries (term + surrounding context from original query)
        # instead of bare single-word terms to avoid catastrophic keyword ambiguity.
        query_lower = query.lower()
        sub_queries: list[SubQuery] = []
        for i, (term, sections) in enumerate(term_sections.items(), start=1):
            # Estimate tokens based on section content
            estimated_tokens = sum(min(count_tokens(s.content), 1500) for s in sections[:3])

            # Build a phrase-level sub-query: include neighbouring terms from
            # the original query so the downstream search has enough context.
            phrase = self._build_phrase_query(term, query_lower, unique_terms)

            sub_queries.append(
                SubQuery(
                    id=i,
                    query=phrase,
                    priority=i,  # Earlier terms have higher priority
                    estimated_tokens=estimated_tokens,
                    key_terms=[term],
                )
            )

        # Analyze dependencies based on document links
        dependencies = self._analyze_document_links(term_sections)

        # Generate suggested sequence using topological sort
        suggested_sequence = self._topological_sort(len(sub_queries), dependencies)

        # Calculate total estimated tokens
        total_estimated = sum(sq.estimated_tokens for sq in sub_queries)

        result = DecomposeResult(
            original_query=query,
            sub_queries=sub_queries,
            dependencies=dependencies,
            suggested_sequence=suggested_sequence,
            total_estimated_tokens=total_estimated,
            strategy_used=strategy,
        )

        input_tokens = count_tokens(query)
        output_tokens = count_tokens(str(result.model_dump()))

        return ToolResult(
            data=result.model_dump(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _handle_multi_query(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_multi_query - execute multiple queries in one call.

        Distributes token budget across queries and executes them in parallel.

        Args:
            params: Dict containing:
                - queries: List of query items with optional per-query budgets
                - max_tokens: Total token budget (default 8000)
                - search_mode: Search mode for all queries (default hybrid)

        Returns:
            ToolResult with MultiQueryResult containing all results
        """
        queries_raw = params.get("queries", [])
        max_tokens = params.get("max_tokens", 8000)
        search_mode_str = params.get("search_mode", "hybrid")

        # Plan gating
        if self.plan not in RECURSIVE_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_multi_query requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Parse search mode
        try:
            search_mode = SearchMode(search_mode_str)
        except ValueError:
            search_mode = SearchMode.HYBRID

        # Apply plan gating on search mode
        if search_mode != SearchMode.KEYWORD and self.plan not in SEMANTIC_SEARCH_PLANS:
            search_mode = SearchMode.KEYWORD

        # Parse queries
        queries: list[dict[str, Any]] = []
        for q in queries_raw:
            if isinstance(q, str):
                queries.append({"query": q, "max_tokens": None})
            elif isinstance(q, dict):
                queries.append(
                    {
                        "query": q.get("query", ""),
                        "max_tokens": q.get("max_tokens"),
                    }
                )

        if not queries:
            return ToolResult(
                data=MultiQueryResult(
                    results=[],
                    total_tokens=0,
                    queries_executed=0,
                    queries_skipped=0,
                    search_mode=search_mode,
                ).model_dump(),
                input_tokens=0,
                output_tokens=0,
            )

        # Distribute budget across queries
        num_queries = len(queries)
        default_per_query = max_tokens // num_queries

        # Assign budgets
        for q in queries:
            if q["max_tokens"] is None:
                q["max_tokens"] = default_per_query

        # Execute queries in parallel
        async def execute_single_query(query_item: dict[str, Any]) -> MultiQueryResultItem:
            try:
                query_params = {
                    "query": query_item["query"],
                    "max_tokens": query_item["max_tokens"],
                    "search_mode": search_mode.value,
                    "include_metadata": True,
                }
                result = await self._handle_context_query(query_params)

                # Extract sections from result
                result_data = result.data
                sections = [
                    ContextSection(**s) if isinstance(s, dict) else s
                    for s in result_data.get("sections", [])
                ]

                return MultiQueryResultItem(
                    query=query_item["query"],
                    sections=sections,
                    tokens_used=result_data.get("total_tokens", 0),
                    success=True,
                )
            except Exception as e:
                logger.warning(f"Multi-query item failed: {e}")
                return MultiQueryResultItem(
                    query=query_item["query"],
                    sections=[],
                    tokens_used=0,
                    success=False,
                    error=str(e),
                )

        # Run all queries in parallel
        tasks = [execute_single_query(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        total_tokens = sum(r.tokens_used for r in results)
        queries_executed = sum(1 for r in results if r.success)
        queries_skipped = sum(1 for r in results if not r.success)

        multi_result = MultiQueryResult(
            results=list(results),
            total_tokens=total_tokens,
            queries_executed=queries_executed,
            queries_skipped=queries_skipped,
            search_mode=search_mode,
        )

        input_tokens = sum(count_tokens(q["query"]) for q in queries)
        output_tokens = total_tokens

        return ToolResult(
            data=multi_result.model_dump(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _handle_plan(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_plan - generate execution plan for complex queries.

        Creates a step-by-step plan for the client's LLM to execute, including
        decomposition, multi-query, and context retrieval steps.

        Args:
            params: Dict containing:
                - query: The complex question to plan for
                - strategy: Execution strategy (default relevance_first)
                - max_tokens: Total token budget (default 16000)

        Returns:
            ToolResult with PlanResult containing execution steps
        """
        query = params.get("query", "")
        strategy_str = params.get("strategy", "relevance_first")
        max_tokens = params.get("max_tokens", 16000)

        # Plan gating - rlm_plan requires Team+ plan
        if self.plan not in PLAN_FEATURE_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_plan requires Team plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=count_tokens(query),
                output_tokens=0,
            )

        # Parse strategy
        try:
            strategy = PlanStrategy(strategy_str)
        except ValueError:
            strategy = PlanStrategy.RELEVANCE_FIRST

        # Generate a unique plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Step 1: Always start with decomposition
        steps: list[PlanStep] = []

        steps.append(
            PlanStep(
                step=1,
                action="decompose",
                params={
                    "query": query,
                    "max_depth": 2,
                },
                depends_on=[],
                expected_output="sub_queries",
            )
        )

        # Step 2: Execute sub-queries based on strategy
        if strategy == PlanStrategy.BREADTH_FIRST:
            # Execute all sub-queries at same level first
            steps.append(
                PlanStep(
                    step=2,
                    action="multi_query",
                    params={
                        "queries": "$step1.sub_queries",
                        "max_tokens": max_tokens - 2000,  # Reserve some for synthesis
                        "search_mode": "hybrid",
                    },
                    depends_on=[1],
                    expected_output="sections",
                )
            )
        elif strategy == PlanStrategy.DEPTH_FIRST:
            # Execute sub-queries one at a time, depth first
            steps.append(
                PlanStep(
                    step=2,
                    action="context_query",
                    params={
                        "query": "$step1.sub_queries[0].query",
                        "max_tokens": max_tokens // 4,
                        "search_mode": "hybrid",
                    },
                    depends_on=[1],
                    expected_output="sections",
                )
            )
            steps.append(
                PlanStep(
                    step=3,
                    action="multi_query",
                    params={
                        "queries": "$step1.sub_queries[1:]",
                        "max_tokens": (max_tokens * 3) // 4 - 1000,
                        "search_mode": "hybrid",
                    },
                    depends_on=[1, 2],
                    expected_output="sections",
                )
            )
        else:  # RELEVANCE_FIRST
            # Execute most relevant sub-query first, then batch rest
            steps.append(
                PlanStep(
                    step=2,
                    action="context_query",
                    params={
                        "query": "$step1.sub_queries[0].query",
                        "max_tokens": max_tokens // 3,
                        "search_mode": "hybrid",
                    },
                    depends_on=[1],
                    expected_output="sections",
                )
            )
            steps.append(
                PlanStep(
                    step=3,
                    action="multi_query",
                    params={
                        "queries": "$step1.sub_queries[1:5]",  # Next 4 most relevant
                        "max_tokens": (max_tokens * 2) // 3 - 1000,
                        "search_mode": "hybrid",
                    },
                    depends_on=[1],
                    expected_output="sections",
                )
            )

        # Estimate tokens and queries
        estimated_tokens = max_tokens
        estimated_queries = len(steps)

        result = PlanResult(
            plan_id=plan_id,
            query=query,
            steps=steps,
            estimated_total_tokens=estimated_tokens,
            strategy=strategy,
            estimated_queries=estimated_queries,
        )

        input_tokens = count_tokens(query)
        output_tokens = count_tokens(str(result.model_dump()))

        return ToolResult(
            data=result.model_dump(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _handle_multi_project_query(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_multi_project_query - query across multiple team projects.

        This tool requires team-level access and must be called via the team endpoint:
        POST /v1/team/{team_slug}/mcp

        Returns an error when called via the project-scoped MCP endpoint.
        """
        query = params.get("query", "")
        return ToolResult(
            data={
                "error": "rlm_multi_project_query requires a team API key",
                "message": "This tool queries across all projects in a team. Use the team endpoint: POST /v1/team/{team_slug}/mcp",
                "example": {
                    "endpoint": "https://api.snipara.com/v1/team/{team_slug}/mcp",
                    "method": "POST",
                    "headers": {"X-API-Key": "your-team-api-key"},
                    "body": {
                        "tool": "rlm_multi_project_query",
                        "params": {"query": query or "your question here"},
                    },
                },
            },
            input_tokens=count_tokens(query),
            output_tokens=0,
        )

    # ============ HELPER METHODS FOR RECURSIVE CONTEXT ============

    @staticmethod
    def _build_phrase_query(term: str, query_lower: str, all_terms: list[str]) -> str:
        """Build a phrase-level sub-query from a term and its original context.

        Instead of returning bare single-word terms like ``"prometheus"`` which
        cause catastrophic keyword ambiguity, this returns phrases like
        ``"prometheus monitoring metrics"`` by including neighbouring terms
        from the original query.
        """
        # If the term is already a bigram+, use it directly
        if " " in term:
            return term

        # Find neighbouring terms from the original query to add context
        context_terms = [term]
        for other in all_terms:
            if other == term or other in term or term in other:
                continue
            # Include if the other term also appears in the original query
            if other in query_lower:
                context_terms.append(other)
            if len(context_terms) >= 3:
                break

        return " ".join(context_terms)

    def _find_sections_for_term(self, term: str) -> list[Section]:
        """Find sections that match a search term."""
        if not self.index:
            return []

        term_lower = term.lower()
        matching: list[tuple[Section, float]] = []

        for section in self.index.sections:
            section_text = (section.title + " " + section.content).lower()
            if term_lower in section_text:
                # Score by frequency
                score = section_text.count(term_lower)
                # Boost title matches
                if term_lower in section.title.lower():
                    score *= 3
                matching.append((section, score))

        # Sort by score and return top matches
        matching.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in matching[:5]]

    def _analyze_document_links(
        self, term_sections: dict[str, list[Section]]
    ) -> list[tuple[int, int]]:
        """
        Find dependencies between sub-queries based on markdown links.

        Returns list of (a, b) tuples meaning query a should be read before query b.
        """
        dependencies: list[tuple[int, int]] = []
        terms = list(term_sections.keys())

        for i, (term, sections) in enumerate(term_sections.items()):
            for section in sections:
                # Find markdown links in section content
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", section.content)
                for link_text, _ in links:
                    link_text_lower = link_text.lower()
                    # Check if link text references another term
                    for j, other_term in enumerate(terms):
                        if i != j and other_term.lower() in link_text_lower:
                            # This section links to content about other_term
                            # So other_term should be read first
                            dep = (j + 1, i + 1)  # 1-indexed
                            if dep not in dependencies:
                                dependencies.append(dep)

        return dependencies

    def _topological_sort(self, num_queries: int, dependencies: list[tuple[int, int]]) -> list[int]:
        """
        Sort query IDs respecting dependencies using Kahn's algorithm.

        Args:
            num_queries: Number of queries (1-indexed IDs)
            dependencies: List of (a, b) tuples meaning a should come before b

        Returns:
            Sorted list of query IDs
        """
        if num_queries == 0:
            return []

        # Build graph
        in_degree = [0] * (num_queries + 1)
        graph: list[list[int]] = [[] for _ in range(num_queries + 1)]

        for a, b in dependencies:
            if 1 <= a <= num_queries and 1 <= b <= num_queries:
                graph[a].append(b)
                in_degree[b] += 1

        # Initialize queue with nodes having no dependencies
        queue = deque([i for i in range(1, num_queries + 1) if in_degree[i] == 0])
        result: list[int] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If we couldn't sort all (cycle detected), return sequential order
        if len(result) != num_queries:
            return list(range(1, num_queries + 1))

        return result

    # ============ PHASE 4.6: SUMMARY STORAGE HANDLERS ============

    async def _handle_store_summary(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_store_summary - store an LLM-generated summary for a document.

        This allows client LLMs to store summaries they generate, which can be
        retrieved later for faster context retrieval without re-processing.

        Args:
            params: Dict containing:
                - document_path: Path to the document
                - summary: The summary text to store
                - summary_type: Type of summary (concise, detailed, technical, keywords, custom)
                - section_id: Optional section identifier for partial summaries
                - line_start: Optional start line for section summary
                - line_end: Optional end line for section summary
                - generated_by: Optional model name that generated the summary

        Returns:
            ToolResult with StoreSummaryResult containing summary ID
        """
        document_path = params.get("document_path", "")
        summary = params.get("summary", "")
        summary_type_str = params.get("summary_type", "concise")
        section_id = params.get("section_id")
        line_start = params.get("line_start")
        line_end = params.get("line_end")
        generated_by = params.get("generated_by")

        # Plan gating
        if self.plan not in SUMMARY_STORAGE_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_store_summary requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=count_tokens(summary),
                output_tokens=0,
            )

        # Validate inputs
        if not document_path:
            return ToolResult(
                data={"error": "document_path is required"},
                input_tokens=0,
                output_tokens=0,
            )

        if not summary:
            return ToolResult(
                data={"error": "summary text is required"},
                input_tokens=0,
                output_tokens=0,
            )

        # Parse summary type
        try:
            summary_type = SummaryType(summary_type_str)
        except ValueError:
            summary_type = SummaryType.CONCISE

        db = await get_db()

        # Find the document
        document = await db.document.find_first(
            where={
                "projectId": self.project_id,
                "path": document_path,
            }
        )

        if not document:
            return ToolResult(
                data={"error": f"Document not found: {document_path}"},
                input_tokens=count_tokens(summary),
                output_tokens=0,
            )

        # Calculate token count for the summary
        token_count = count_tokens(summary)

        # Check if summary already exists (upsert)
        existing = await db.documentsummary.find_first(
            where={
                "documentId": document.id,
                "summaryType": summary_type.value,
                "sectionId": section_id,
            }
        )

        if existing:
            # Update existing summary
            await db.documentsummary.update(
                where={"id": existing.id},
                data={
                    "summary": summary,
                    "tokenCount": token_count,
                    "lineStart": line_start,
                    "lineEnd": line_end,
                    "generatedBy": generated_by,
                },
            )
            created = False
            summary_id = existing.id
        else:
            # Create new summary
            created_summary = await db.documentsummary.create(
                data={
                    "documentId": document.id,
                    "projectId": self.project_id,
                    "summary": summary,
                    "summaryType": summary_type.value,
                    "sectionId": section_id,
                    "lineStart": line_start,
                    "lineEnd": line_end,
                    "tokenCount": token_count,
                    "generatedBy": generated_by,
                }
            )
            created = True
            summary_id = created_summary.id

        result = StoreSummaryResult(
            summary_id=summary_id,
            document_path=document_path,
            summary_type=summary_type,
            token_count=token_count,
            created=created,
            message=f"Summary {'created' if created else 'updated'} successfully ({token_count} tokens)",
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=count_tokens(summary),
            output_tokens=count_tokens(str(result.model_dump())),
        )

    async def _handle_get_summaries(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_get_summaries - retrieve stored summaries.

        Args:
            params: Dict containing:
                - document_path: Filter by document path (optional)
                - summary_type: Filter by summary type (optional)
                - section_id: Filter by section ID (optional)
                - include_content: Include summary content in response (default True)

        Returns:
            ToolResult with GetSummariesResult containing matching summaries
        """
        document_path = params.get("document_path")
        summary_type_str = params.get("summary_type")
        section_id = params.get("section_id")
        include_content = params.get("include_content", True)

        # Plan gating
        if self.plan not in SUMMARY_STORAGE_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_get_summaries requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        db = await get_db()

        # Build query filters
        where_clause: dict[str, Any] = {"projectId": self.project_id}

        if document_path:
            # Find document ID first
            document = await db.document.find_first(
                where={
                    "projectId": self.project_id,
                    "path": document_path,
                }
            )
            if document:
                where_clause["documentId"] = document.id
            else:
                # No document found, return empty
                return ToolResult(
                    data=GetSummariesResult(
                        summaries=[],
                        total_count=0,
                        total_tokens=0,
                    ).model_dump(),
                    input_tokens=0,
                    output_tokens=0,
                )

        if summary_type_str:
            try:
                summary_type = SummaryType(summary_type_str)
                where_clause["summaryType"] = summary_type.value
            except ValueError:
                pass

        if section_id:
            where_clause["sectionId"] = section_id

        # Query summaries with document info
        summaries = await db.documentsummary.find_many(
            where=where_clause,
            include={"document": True},
            order={"createdAt": "desc"},
        )

        # Build response
        summary_infos: list[SummaryInfo] = []
        total_tokens = 0

        for s in summaries:
            try:
                summary_type_enum = SummaryType(s.summaryType)
            except ValueError:
                summary_type_enum = SummaryType.CUSTOM

            summary_info = SummaryInfo(
                summary_id=s.id,
                document_path=s.document.path if s.document else "unknown",
                summary_type=summary_type_enum,
                section_id=s.sectionId,
                line_start=s.lineStart,
                line_end=s.lineEnd,
                token_count=s.tokenCount,
                generated_by=s.generatedBy,
                content=s.summary if include_content else None,
                created_at=s.createdAt,
                updated_at=s.updatedAt,
            )
            summary_infos.append(summary_info)
            total_tokens += s.tokenCount

        result = GetSummariesResult(
            summaries=summary_infos,
            total_count=len(summary_infos),
            total_tokens=total_tokens,
        )

        output_tokens = total_tokens if include_content else len(summary_infos) * 50

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=output_tokens,
        )

    async def _handle_delete_summary(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_delete_summary - delete stored summaries.

        Args:
            params: Dict containing (at least one required):
                - summary_id: Specific summary ID to delete
                - document_path: Delete all summaries for this document
                - summary_type: Delete summaries of this type

        Returns:
            ToolResult with DeleteSummaryResult containing deletion count
        """
        summary_id = params.get("summary_id")
        document_path = params.get("document_path")
        summary_type_str = params.get("summary_type")

        # Plan gating
        if self.plan not in SUMMARY_STORAGE_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_delete_summary requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Require at least one filter
        if not summary_id and not document_path and not summary_type_str:
            return ToolResult(
                data={
                    "error": "At least one of summary_id, document_path, or summary_type is required"
                },
                input_tokens=0,
                output_tokens=0,
            )

        db = await get_db()

        # Build delete filter
        where_clause: dict[str, Any] = {"projectId": self.project_id}

        if summary_id:
            # Delete specific summary
            where_clause["id"] = summary_id
        else:
            if document_path:
                document = await db.document.find_first(
                    where={
                        "projectId": self.project_id,
                        "path": document_path,
                    }
                )
                if document:
                    where_clause["documentId"] = document.id
                else:
                    return ToolResult(
                        data=DeleteSummaryResult(
                            deleted_count=0,
                            message="Document not found",
                        ).model_dump(),
                        input_tokens=0,
                        output_tokens=0,
                    )

            if summary_type_str:
                try:
                    summary_type = SummaryType(summary_type_str)
                    where_clause["summaryType"] = summary_type.value
                except ValueError:
                    pass

        # Execute delete
        deleted = await db.documentsummary.delete_many(where=where_clause)
        deleted_count = deleted if isinstance(deleted, int) else getattr(deleted, "count", 0)

        result = DeleteSummaryResult(
            deleted_count=deleted_count,
            message=f"Deleted {deleted_count} summary(ies)",
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=count_tokens(str(result.model_dump())),
        )

    # ============ PHASE 7: SHARED CONTEXT HANDLERS ============

    async def _handle_shared_context(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_shared_context - get shared context for the project.

        This tool retrieves shared context documents from linked collections,
        respecting token budget and category priorities.

        Args:
            params: Dict containing:
                - max_tokens: Maximum tokens to return (default 4000)
                - categories: List of categories to include (default all)
                - include_content: Include merged content string (default True)

        Returns:
            ToolResult with SharedContextResult containing documents and content
        """
        max_tokens = params.get("max_tokens", 4000)
        categories_raw = params.get("categories")
        include_content = params.get("include_content", True)

        # Plan gating
        if self.plan not in SHARED_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_shared_context requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Parse categories filter
        category_filter: list[DocumentCategory] | None = None
        if categories_raw:
            category_filter = []
            for cat_str in categories_raw:
                try:
                    category_filter.append(DocumentCategory(cat_str))
                except ValueError:
                    pass

        # Load shared context
        shared_ctx = await load_project_shared_context(self.project_id)

        if not shared_ctx.documents:
            return ToolResult(
                data=SharedContextResult(
                    documents=[],
                    merged_content=None,
                    total_tokens=0,
                    collections_loaded=0,
                    context_hash="",
                ).model_dump(),
                input_tokens=0,
                output_tokens=0,
            )

        # Apply category filter if specified
        if category_filter:
            shared_ctx.documents = [
                d for d in shared_ctx.documents if d.category in category_filter
            ]

        # Allocate budget
        allocated_docs = allocate_shared_context_budget(shared_ctx, max_tokens)

        # Build response
        doc_infos: list[SharedDocumentInfo] = []
        for doc in allocated_docs:
            try:
                cat_enum = DocumentCategoryEnum(doc.category.value)
            except ValueError:
                cat_enum = DocumentCategoryEnum.BEST_PRACTICES

            doc_infos.append(
                SharedDocumentInfo(
                    id=doc.id,
                    title=doc.title,
                    category=cat_enum,
                    token_count=doc.token_count,
                    collection_name=doc.collection_name,
                    tags=doc.tags,
                )
            )

        # Build merged content if requested
        merged_content: str | None = None
        if include_content:
            merged_content = merge_shared_context_with_project_docs(
                allocated_docs,
                "",  # No project content, just shared context
            )

        total_tokens = sum(d.token_count for d in allocated_docs)
        context_hash = compute_context_hash(shared_ctx)

        result = SharedContextResult(
            documents=doc_infos,
            merged_content=merged_content,
            total_tokens=total_tokens,
            collections_loaded=len(shared_ctx.collection_versions),
            context_hash=context_hash,
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=total_tokens if include_content else len(doc_infos) * 50,
        )

    async def _handle_list_templates(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_list_templates - list available prompt templates.

        Args:
            params: Dict containing:
                - category: Optional category filter

        Returns:
            ToolResult with ListTemplatesResult containing templates
        """
        category = params.get("category")

        # Plan gating
        if self.plan not in SHARED_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_list_templates requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Get templates
        templates = await get_shared_prompt_templates(self.project_id, category)

        # Build response
        template_infos: list[PromptTemplateInfo] = []
        categories_seen: set[str] = set()

        for t in templates:
            template_infos.append(
                PromptTemplateInfo(
                    id=t["id"],
                    name=t["name"],
                    slug=t["slug"],
                    description=t.get("description"),
                    prompt=t["prompt"],
                    variables=t.get("variables", []),
                    category=t["category"],
                    collection_name=t["collection_name"],
                )
            )
            categories_seen.add(t["category"])

        result = ListTemplatesResult(
            templates=template_infos,
            total_count=len(template_infos),
            categories=sorted(categories_seen),
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=len(template_infos) * 100,
        )

    async def _handle_get_template(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_get_template - get a specific template and optionally render it.

        Args:
            params: Dict containing:
                - template_id: Template ID (optional)
                - slug: Template slug (optional, requires collection context)
                - variables: Dict of variable values to substitute

        Returns:
            ToolResult with GetTemplateResult containing template and rendered prompt
        """
        template_id = params.get("template_id")
        slug = params.get("slug")
        variables = params.get("variables", {})

        # Plan gating
        if self.plan not in SHARED_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_get_template requires Pro plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        if not template_id and not slug:
            return ToolResult(
                data={"error": "Either template_id or slug is required"},
                input_tokens=0,
                output_tokens=0,
            )

        # Get all templates for this project
        templates = await get_shared_prompt_templates(self.project_id)

        # Find the template
        template_data: dict | None = None
        for t in templates:
            if template_id and t["id"] == template_id:
                template_data = t
                break
            if slug and t["slug"] == slug:
                template_data = t
                break

        if not template_data:
            return ToolResult(
                data=GetTemplateResult(
                    template=None,
                    rendered_prompt=None,
                    missing_variables=[],
                ).model_dump(),
                input_tokens=0,
                output_tokens=0,
            )

        # Build template info
        template_info = PromptTemplateInfo(
            id=template_data["id"],
            name=template_data["name"],
            slug=template_data["slug"],
            description=template_data.get("description"),
            prompt=template_data["prompt"],
            variables=template_data.get("variables", []),
            category=template_data["category"],
            collection_name=template_data["collection_name"],
        )

        # Render prompt with variables
        rendered_prompt = template_data["prompt"]
        missing_variables: list[str] = []

        for var_name in template_data.get("variables", []):
            placeholder = f"{{{{{var_name}}}}}"  # {{var_name}}
            if var_name in variables:
                rendered_prompt = rendered_prompt.replace(placeholder, str(variables[var_name]))
            else:
                missing_variables.append(var_name)

        result = GetTemplateResult(
            template=template_info,
            rendered_prompt=rendered_prompt,
            missing_variables=missing_variables,
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=count_tokens(rendered_prompt),
        )

    async def _handle_list_collections(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_list_collections - list accessible shared context collections.

        Args:
            params: Dict containing:
                - include_public: Whether to include public collections (default True)

        Returns:
            ToolResult with list of collection info
        """
        include_public = params.get("include_public", True)

        try:
            collections = await list_shared_collections(
                user_id=self.user_id,
                include_public=include_public,
            )

            return ToolResult(
                data={
                    "collections": collections,
                    "count": len(collections),
                },
                input_tokens=0,
                output_tokens=0,
            )
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return ToolResult(
                data={"error": f"Failed to list collections: {str(e)}"},
                input_tokens=0,
                output_tokens=0,
            )

    async def _handle_upload_shared_document(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_upload_shared_document - upload a document to a shared collection.

        Args:
            params: Dict containing:
                - collection_id: The shared collection ID
                - title: Document title
                - content: Document content (markdown)
                - category: Optional (MANDATORY, BEST_PRACTICES, GUIDELINES, REFERENCE)
                - tags: Optional list of tags
                - priority: Optional priority (0-100)

        Returns:
            ToolResult with document ID and status
        """
        collection_id = params.get("collection_id", "")
        title = params.get("title", "")
        content = params.get("content", "")
        category = params.get("category")
        tags = params.get("tags")
        priority = params.get("priority", 0)

        if not collection_id:
            return ToolResult(
                data={"error": "collection_id is required"},
                input_tokens=0,
                output_tokens=0,
            )

        if not title:
            return ToolResult(
                data={"error": "title is required"},
                input_tokens=0,
                output_tokens=0,
            )

        if not content:
            return ToolResult(
                data={"error": "content is required"},
                input_tokens=0,
                output_tokens=0,
            )

        # Plan gating - require Team+ for shared context write operations
        if self.plan not in {Plan.TEAM, Plan.ENTERPRISE}:
            return ToolResult(
                data={
                    "error": "rlm_upload_shared_document requires Team plan or higher",
                    "upgrade_url": "/billing/upgrade",
                },
                input_tokens=0,
                output_tokens=0,
            )

        try:
            result = await create_shared_document(
                collection_id=collection_id,
                user_id=self.user_id,
                title=title,
                content=content,
                category=category,
                tags=tags,
                priority=priority,
            )

            return ToolResult(
                data=result,
                input_tokens=count_tokens(content),
                output_tokens=0,
            )
        except ValueError as e:
            return ToolResult(
                data={"error": str(e)},
                input_tokens=0,
                output_tokens=0,
            )
        except Exception as e:
            logger.error(f"Error uploading shared document: {e}")
            return ToolResult(
                data={"error": f"Failed to upload document: {str(e)}"},
                input_tokens=0,
                output_tokens=0,
            )

    # ============ PHASE 8.2: AGENT MEMORY HANDLERS ============

    async def _handle_remember(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_remember - store a memory for later recall.

        Args:
            params: Dict containing:
                - content: Memory content to store
                - type: Memory type (fact, decision, learning, preference, todo, context)
                - scope: Visibility scope (agent, project, team, user)
                - category: Optional grouping category
                - ttl_days: Days until expiration
                - related_to: IDs of related memories
                - document_refs: Referenced document paths

        Returns:
            ToolResult with memory ID and confirmation
        """
        content = params.get("content", "")
        memory_type = params.get("type", "fact")
        scope = params.get("scope", "project")
        category = params.get("category")
        ttl_days = params.get("ttl_days")
        related_to = params.get("related_to")
        document_refs = params.get("document_refs")

        if not content:
            return ToolResult(
                data={"error": "content is required"},
                input_tokens=0,
                output_tokens=0,
            )

        # Check memory limits
        allowed, error = await check_memory_limits(self.project_id)
        if not allowed:
            return ToolResult(
                data={"error": error, "upgrade_url": "/billing/upgrade"},
                input_tokens=count_tokens(content),
                output_tokens=0,
            )

        result = await store_memory(
            project_id=self.project_id,
            content=content,
            memory_type=memory_type,
            scope=scope,
            category=category,
            ttl_days=ttl_days,
            related_to=related_to,
            document_refs=document_refs,
            source="mcp",
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(content),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_recall(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_recall - semantically recall relevant memories.

        Args:
            params: Dict containing:
                - query: Search query
                - type: Filter by memory type
                - scope: Filter by scope
                - category: Filter by category
                - limit: Maximum memories to return
                - min_relevance: Minimum relevance score (0-1)

        Returns:
            ToolResult with recalled memories and relevance scores
        """
        query = params.get("query", "")
        memory_type = params.get("type")
        scope = params.get("scope")
        category = params.get("category")
        limit = params.get("limit", 5)
        min_relevance = params.get("min_relevance", 0.5)

        if not query:
            return ToolResult(
                data={"error": "query is required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await semantic_recall(
            project_id=self.project_id,
            query=query,
            memory_type=memory_type,
            scope=scope,
            category=category,
            limit=limit,
            min_relevance=min_relevance,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(query),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_memories(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_memories - list memories with filters.

        Args:
            params: Dict containing:
                - type: Filter by memory type
                - scope: Filter by scope
                - category: Filter by category
                - search: Text search in content
                - limit: Maximum memories to return
                - offset: Pagination offset

        Returns:
            ToolResult with memories list and pagination info
        """
        memory_type = params.get("type")
        scope = params.get("scope")
        category = params.get("category")
        search = params.get("search")
        limit = params.get("limit", 20)
        offset = params.get("offset", 0)

        result = await list_memories(
            project_id=self.project_id,
            memory_type=memory_type,
            scope=scope,
            category=category,
            search=search,
            limit=limit,
            offset=offset,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_forget(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_forget - delete memories by ID or filter criteria.

        Args:
            params: Dict containing (at least one):
                - memory_id: Specific memory to delete
                - type: Delete all of this type
                - category: Delete all in this category
                - older_than_days: Delete memories older than N days

        Returns:
            ToolResult with deletion count
        """
        memory_id = params.get("memory_id")
        memory_type = params.get("type")
        category = params.get("category")
        older_than_days = params.get("older_than_days")

        # Require at least one filter
        if not any([memory_id, memory_type, category, older_than_days]):
            return ToolResult(
                data={"error": "At least one filter is required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await delete_memories(
            project_id=self.project_id,
            memory_id=memory_id,
            memory_type=memory_type,
            category=category,
            older_than_days=older_than_days,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    # ============ PHASE 9.1: MULTI-AGENT SWARM HANDLERS ============

    async def _handle_swarm_create(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_swarm_create - create a new agent swarm.

        Args:
            params: Dict containing:
                - name: Swarm name
                - description: Optional description
                - max_agents: Maximum agents allowed
                - config: Optional swarm configuration

        Returns:
            ToolResult with swarm ID and info
        """
        name = params.get("name", "")
        description = params.get("description")
        max_agents = params.get("max_agents", 10)
        config = params.get("config")

        if not name:
            return ToolResult(
                data={"error": "name is required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await create_swarm(
            project_id=self.project_id,
            name=name,
            description=description,
            max_agents=max_agents,
            config=config,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(name + (description or "")),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_swarm_join(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_swarm_join - join an existing swarm as an agent.

        Args:
            params: Dict containing:
                - swarm_id: Swarm to join
                - agent_id: Unique agent identifier
                - role: Agent role (coordinator, worker, observer)
                - capabilities: List of capabilities

        Returns:
            ToolResult with join status
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        role = params.get("role", "worker")
        capabilities = params.get("capabilities")

        if not swarm_id or not agent_id:
            return ToolResult(
                data={"error": "swarm_id and agent_id are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await join_swarm(
            swarm_id=swarm_id,
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_claim(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_claim - claim exclusive access to a resource.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Agent identifier
                - resource_type: Type of resource
                - resource_id: Resource identifier
                - timeout_seconds: Claim timeout

        Returns:
            ToolResult with claim status
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        resource_type = params.get("resource_type", "")
        resource_id = params.get("resource_id", "")
        timeout_seconds = params.get("timeout_seconds", 300)

        if not all([swarm_id, agent_id, resource_type, resource_id]):
            return ToolResult(
                data={"error": "swarm_id, agent_id, resource_type, and resource_id are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await acquire_claim(
            swarm_id=swarm_id,
            agent_id=agent_id,
            resource_type=resource_type,
            resource_id=resource_id,
            timeout_seconds=timeout_seconds,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_release(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_release - release a claimed resource.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Agent identifier
                - claim_id: Claim ID (optional)
                - resource_type: Resource type (alternative)
                - resource_id: Resource ID (alternative)

        Returns:
            ToolResult with release status
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        claim_id = params.get("claim_id")
        resource_type = params.get("resource_type")
        resource_id = params.get("resource_id")

        if not swarm_id or not agent_id:
            return ToolResult(
                data={"error": "swarm_id and agent_id are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await release_claim(
            swarm_id=swarm_id,
            agent_id=agent_id,
            claim_id=claim_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_state_get(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_state_get - read shared swarm state.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - key: State key to read

        Returns:
            ToolResult with state value and version
        """
        swarm_id = params.get("swarm_id", "")
        key = params.get("key", "")

        if not swarm_id or not key:
            return ToolResult(
                data={"error": "swarm_id and key are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await get_state(swarm_id=swarm_id, key=key)

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_state_set(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_state_set - write shared swarm state.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Agent identifier
                - key: State key
                - value: Value to set
                - expected_version: For optimistic locking

        Returns:
            ToolResult with new version
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        key = params.get("key", "")
        value = params.get("value")
        expected_version = params.get("expected_version")

        if not swarm_id or not agent_id or not key:
            return ToolResult(
                data={"error": "swarm_id, agent_id, and key are required"},
                input_tokens=0,
                output_tokens=0,
            )

        if value is None:
            return ToolResult(
                data={"error": "value is required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await set_state(
            swarm_id=swarm_id,
            agent_id=agent_id,
            key=key,
            value=value,
            expected_version=expected_version,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(str(value)),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_broadcast(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_broadcast - send an event to all agents in swarm.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Sending agent identifier
                - event_type: Event type string
                - payload: Event data

        Returns:
            ToolResult with broadcast status
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        event_type = params.get("event_type", "")
        payload = params.get("payload")

        if not swarm_id or not agent_id or not event_type:
            return ToolResult(
                data={"error": "swarm_id, agent_id, and event_type are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await broadcast_event(
            swarm_id=swarm_id,
            agent_id=agent_id,
            event_type=event_type,
            payload=payload,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(str(payload) if payload else ""),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_task_create(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_task_create - create a task in the swarm's queue.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Creating agent
                - title: Task title
                - description: Task description
                - priority: Priority level
                - depends_on: Task IDs this depends on
                - metadata: Additional task data

        Returns:
            ToolResult with task ID
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        title = params.get("title", "")
        description = params.get("description")
        priority = params.get("priority", 0)
        depends_on = params.get("depends_on")
        metadata = params.get("metadata")

        if not swarm_id or not agent_id or not title:
            return ToolResult(
                data={"error": "swarm_id, agent_id, and title are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await create_task(
            swarm_id=swarm_id,
            agent_id=agent_id,
            title=title,
            description=description,
            priority=priority,
            depends_on=depends_on,
            metadata=metadata,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(title + (description or "")),
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_task_claim(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_task_claim - claim a task from the queue.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Claiming agent
                - task_id: Specific task to claim (optional)
                - timeout_seconds: Task timeout

        Returns:
            ToolResult with claimed task info
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        task_id = params.get("task_id")
        timeout_seconds = params.get("timeout_seconds", 600)

        if not swarm_id or not agent_id:
            return ToolResult(
                data={"error": "swarm_id and agent_id are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await claim_task(
            swarm_id=swarm_id,
            agent_id=agent_id,
            task_id=task_id,
            timeout_seconds=timeout_seconds,
        )

        return ToolResult(
            data=result,
            input_tokens=0,
            output_tokens=count_tokens(str(result)),
        )

    async def _handle_task_complete(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_task_complete - mark a task as completed or failed.

        Args:
            params: Dict containing:
                - swarm_id: Swarm ID
                - agent_id: Completing agent
                - task_id: Task to complete
                - success: Whether task succeeded
                - result: Task result data

        Returns:
            ToolResult with completion status
        """
        swarm_id = params.get("swarm_id", "")
        agent_id = params.get("agent_id", "")
        task_id = params.get("task_id", "")
        success = params.get("success", True)
        task_result = params.get("result")

        if not swarm_id or not agent_id or not task_id:
            return ToolResult(
                data={"error": "swarm_id, agent_id, and task_id are required"},
                input_tokens=0,
                output_tokens=0,
            )

        result = await complete_task(
            swarm_id=swarm_id,
            agent_id=agent_id,
            task_id=task_id,
            result=task_result,
            success=success,
        )

        return ToolResult(
            data=result,
            input_tokens=count_tokens(str(task_result) if task_result else ""),
            output_tokens=count_tokens(str(result)),
        )

    # ============ Phase 10: Document Sync Handlers ============

    async def _handle_upload_document(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_upload_document - upload or update a document.

        Args:
            params: Dict containing:
                - path: Document path (e.g., 'docs/api.md')
                - content: Document content (markdown)

        Returns:
            ToolResult with upload status
        """
        path = params.get("path", "")
        content = params.get("content", "")

        if not path or not content:
            return ToolResult(
                data={"error": "path and content are required"},
                input_tokens=0,
                output_tokens=0,
            )

        # Validate path
        if not path.endswith((".md", ".txt", ".mdx")):
            return ToolResult(
                data={"error": "Only .md, .txt, .mdx files are supported"},
                input_tokens=0,
                output_tokens=0,
            )

        db = await get_db()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        size = len(content.encode())

        # Check if document exists
        existing = await db.document.find_first(where={"projectId": self.project_id, "path": path})

        if existing:
            # Check if content changed
            if existing.hash == content_hash:
                result = UploadDocumentResult(
                    path=path,
                    action="unchanged",
                    size=size,
                    hash=content_hash,
                    message=f"Document '{path}' is unchanged",
                )
            else:
                # Update existing document
                await db.document.update(
                    where={"id": existing.id},
                    data={"content": content, "hash": content_hash, "size": size},
                )
                # Invalidate index cache
                self.index = None
                result = UploadDocumentResult(
                    path=path,
                    action="updated",
                    size=size,
                    hash=content_hash,
                    message=f"Document '{path}' updated ({size} bytes)",
                )
        else:
            # Create new document
            await db.document.create(
                data={
                    "projectId": self.project_id,
                    "path": path,
                    "content": content,
                    "hash": content_hash,
                    "size": size,
                }
            )
            # Invalidate index cache
            self.index = None
            result = UploadDocumentResult(
                path=path,
                action="created",
                size=size,
                hash=content_hash,
                message=f"Document '{path}' created ({size} bytes)",
            )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=count_tokens(content),
            output_tokens=count_tokens(str(result.model_dump())),
        )

    async def _handle_sync_documents(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_sync_documents - bulk sync multiple documents.

        Args:
            params: Dict containing:
                - documents: List of {path, content} objects
                - delete_missing: Whether to delete docs not in list

        Returns:
            ToolResult with sync status
        """
        documents = params.get("documents", [])
        delete_missing = params.get("delete_missing", False)

        if not documents:
            return ToolResult(
                data={"error": "documents list is required"},
                input_tokens=0,
                output_tokens=0,
            )

        db = await get_db()
        created = 0
        updated = 0
        unchanged = 0
        deleted = 0
        input_tokens = 0

        # Get all existing documents
        existing_docs = await db.document.find_many(where={"projectId": self.project_id})
        existing_by_path = {doc.path: doc for doc in existing_docs}
        synced_paths = set()

        for doc_data in documents:
            path = doc_data.get("path", "")
            content = doc_data.get("content", "")

            if not path or not content:
                continue

            # Validate path
            if not path.endswith((".md", ".txt", ".mdx")):
                continue

            synced_paths.add(path)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            size = len(content.encode())
            input_tokens += count_tokens(content)

            if path in existing_by_path:
                existing = existing_by_path[path]
                if existing.hash == content_hash:
                    unchanged += 1
                else:
                    await db.document.update(
                        where={"id": existing.id},
                        data={"content": content, "hash": content_hash, "size": size},
                    )
                    updated += 1
            else:
                await db.document.create(
                    data={
                        "projectId": self.project_id,
                        "path": path,
                        "content": content,
                        "hash": content_hash,
                        "size": size,
                    }
                )
                created += 1

        # Delete missing documents if requested
        if delete_missing:
            for path, doc in existing_by_path.items():
                if path not in synced_paths:
                    await db.document.delete(where={"id": doc.id})
                    deleted += 1

        # Invalidate index cache if any changes
        if created > 0 or updated > 0 or deleted > 0:
            self.index = None

        total = created + updated + unchanged
        result = SyncDocumentsResult(
            created=created,
            updated=updated,
            unchanged=unchanged,
            deleted=deleted,
            total=total,
            message=f"Synced {total} documents: {created} created, {updated} updated, {unchanged} unchanged, {deleted} deleted",
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=input_tokens,
            output_tokens=count_tokens(str(result.model_dump())),
        )

    async def _handle_settings(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_settings - get project settings.

        Args:
            params: Dict (no parameters required)

        Returns:
            ToolResult with project settings
        """
        result = SettingsResult(
            project_id=self.project_id,
            max_tokens_per_query=self.settings.max_tokens_per_query,
            search_mode=self.settings.search_mode,
            include_summaries=self.settings.include_summaries,
            auto_inject_context=self.settings.auto_inject_context,
            message=f"Settings for project {self.project_id}",
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=count_tokens(str(result.model_dump())),
        )

    # ============ PHASE 11: ACCESS CONTROL HANDLERS ============

    async def _handle_request_access(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_request_access - request access to a project.

        This tool allows team members with NONE access level to request
        higher access levels (VIEWER, EDITOR, ADMIN) from project admins.

        Args:
            params: Dict containing:
                - requested_level: The access level to request (VIEWER, EDITOR, ADMIN)
                - reason: Optional reason for requesting access

        Returns:
            ToolResult with RequestAccessResult containing request status
        """
        requested_level = params.get("requested_level", "VIEWER").upper()
        reason = params.get("reason", "")

        # Validate requested level
        valid_levels = {"VIEWER", "EDITOR", "ADMIN"}
        if requested_level not in valid_levels:
            return ToolResult(
                data={
                    "error": f"Invalid level. Must be one of: {', '.join(valid_levels)}",
                    "valid_levels": list(valid_levels),
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Check if user_id is available (needed for access requests)
        if not self.user_id:
            return ToolResult(
                data={
                    "error": "User context required for access requests. This typically means you're using a project API key which already has access.",
                },
                input_tokens=0,
                output_tokens=0,
            )

        db = await get_db()

        # Get project info
        project = await db.project.find_first(
            where={"id": self.project_id},
            select={"id": True, "name": True, "slug": True, "teamId": True},
        )

        if not project:
            return ToolResult(
                data={"error": "Project not found"},
                input_tokens=0,
                output_tokens=0,
            )

        # Get team member ID for the user
        team_member = await db.teammember.find_first(
            where={
                "userId": self.user_id,
                "teamId": project.teamId,
            },
            select={"id": True},
        )

        if not team_member:
            return ToolResult(
                data={"error": "You must be a team member to request project access"},
                input_tokens=0,
                output_tokens=0,
            )

        # Check for existing pending request
        existing = await db.accessrequest.find_first(
            where={
                "projectId": project.id,
                "teamMemberId": team_member.id,
                "status": "PENDING",
            },
        )

        if existing:
            return ToolResult(
                data={
                    "error": "You already have a pending access request for this project",
                    "request_id": existing.id,
                    "status": "pending",
                },
                input_tokens=0,
                output_tokens=0,
            )

        # Create access request
        access_request = await db.accessrequest.create(
            data={
                "projectId": project.id,
                "teamMemberId": team_member.id,
                "requestedLevel": requested_level,
                "reason": reason[:500] if reason else None,
                "status": "PENDING",
            },
        )

        result = RequestAccessResult(
            request_id=access_request.id,
            project_id=project.id,
            project_name=project.name,
            requested_level=requested_level,
            status="pending",
            message="Access request submitted. A project admin will review your request.",
            dashboard_url=f"https://app.snipara.com/team/projects/{project.slug}/access-requests",
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=count_tokens(str(result.model_dump())),
        )

    # ============ PHASE 12: RLM ORCHESTRATION HANDLERS ============

    def _get_file_content(self, file_path: str) -> str | None:
        """Get raw content for a single file using file boundary tracking.

        Args:
            file_path: The document path (e.g. 'docs/api.md')

        Returns:
            Raw file content as a string, or None if not found.
        """
        if not self.index or file_path not in self.index.file_boundaries:
            return None
        start, end = self.index.file_boundaries[file_path]
        return "\n".join(self.index.lines[start:end])

    async def _handle_load_document(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_load_document - return raw document content by path.

        Enables RLM-style exploration where the model receives raw content
        and navigates it with its own logic, rather than pre-optimized sections.

        Plan-gating: PRO+
        """
        if self.plan not in RAW_DOCUMENT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_load_document requires a Pro plan or higher.",
                    "current_plan": self.plan.value,
                    "upgrade_url": "https://app.snipara.com/settings/billing",
                },
                input_tokens=0,
                output_tokens=0,
            )

        file_path = params.get("path", "")
        if not file_path:
            return ToolResult(
                data={"error": "Missing required parameter: path"},
                input_tokens=0,
                output_tokens=0,
            )

        if not self.index:
            return ToolResult(
                data={"error": "No documentation loaded"},
                input_tokens=0,
                output_tokens=0,
            )

        content = self._get_file_content(file_path)
        if content is None:
            # Fuzzy match: suggest similar paths
            available = self.index.files
            suggestions = [f for f in available if file_path.split("/")[-1] in f]
            return ToolResult(
                data={
                    "error": f"Document not found: {file_path}",
                    "available_files": available[:20],
                    "suggestions": suggestions[:5],
                },
                input_tokens=0,
                output_tokens=0,
            )

        token_count = count_tokens(content)
        response = {
            "path": file_path,
            "content": content,
            "token_count": token_count,
            "lines": content.count("\n") + 1,
        }

        return ToolResult(
            data=response,
            input_tokens=0,
            output_tokens=token_count,
        )

    async def _handle_load_project(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_load_project - return structured map of all documents.

        Returns a token-budgeted overview of every file in the project,
        with optional path filtering and content inclusion.

        Plan-gating: TEAM+
        """
        if self.plan not in ORCHESTRATION_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_load_project requires a Team plan or higher.",
                    "current_plan": self.plan.value,
                    "upgrade_url": "https://app.snipara.com/settings/billing",
                },
                input_tokens=0,
                output_tokens=0,
            )

        if not self.index:
            return ToolResult(
                data={"error": "No documentation loaded"},
                input_tokens=0,
                output_tokens=0,
            )

        max_tokens = params.get("max_tokens", 16000)
        paths_filter = params.get("paths_filter", [])
        include_content = params.get("include_content", True)

        files = self.index.files
        if paths_filter:
            # Filter to matching paths (prefix match)
            files = [f for f in files if any(f.startswith(pf) for pf in paths_filter)]

        documents: list[dict[str, Any]] = []
        total_tokens = 0

        for file_path in files:
            content = self._get_file_content(file_path)
            if content is None:
                continue

            file_tokens = count_tokens(content)

            doc_entry: dict[str, Any] = {
                "path": file_path,
                "token_count": file_tokens,
                "lines": content.count("\n") + 1,
            }

            if include_content:
                # Check budget
                if total_tokens + file_tokens > max_tokens:
                    doc_entry["content"] = content[: (max_tokens - total_tokens) * 4]
                    doc_entry["truncated"] = True
                    documents.append(doc_entry)
                    total_tokens = max_tokens
                    break
                else:
                    doc_entry["content"] = content
                    doc_entry["truncated"] = False
                    total_tokens += file_tokens

            documents.append(doc_entry)

        response = {
            "total_files": len(self.index.files),
            "returned_files": len(documents),
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "documents": documents,
        }

        return ToolResult(
            data=response,
            input_tokens=0,
            output_tokens=total_tokens,
        )

    async def _handle_orchestrate(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_orchestrate - multi-round context exploration.

        Performs a 3-round exploration strategy in a single tool call:
          1. Sections scan: list all sections to understand project structure
          2. Ranked search: run context_query to find top relevant sections
          3. Raw load: load full raw content for the top-scoring files

        This combines the intelligence of rlm_context_query with the raw
        access of rlm_load_document, automated in one call.

        Plan-gating: TEAM+
        """
        if self.plan not in ORCHESTRATION_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_orchestrate requires a Team plan or higher.",
                    "current_plan": self.plan.value,
                    "upgrade_url": "https://app.snipara.com/settings/billing",
                },
                input_tokens=0,
                output_tokens=0,
            )

        query = params.get("query", "")
        if not query:
            return ToolResult(
                data={"error": "Missing required parameter: query"},
                input_tokens=0,
                output_tokens=0,
            )

        max_tokens = params.get("max_tokens", 16000)
        top_k = params.get("top_k", 5)
        search_mode_str = params.get("search_mode", self.settings.search_mode)

        try:
            search_mode = SearchMode(search_mode_str)
        except ValueError:
            search_mode = SearchMode.KEYWORD

        if not self.index:
            return ToolResult(
                data={"error": "No documentation loaded"},
                input_tokens=0,
                output_tokens=0,
            )

        # Round 1: Section scan (lightweight structure overview)
        all_sections = [
            {
                "id": s.id,
                "title": s.title,
                "start_line": s.start_line,
                "end_line": s.end_line,
            }
            for s in self.index.sections
        ]

        # Round 2: Ranked search (reuse scoring logic from context_query)
        scored_sections = await self._score_sections(query, search_mode)

        # Take top-k sections
        top_sections = scored_sections[:top_k]

        # Identify unique files that contain the top sections
        file_hits: dict[str, float] = {}
        for section, score in top_sections:
            # Find which file contains this section
            for file_path, (start, end) in self.index.file_boundaries.items():
                if section.start_line >= start + 1 and section.end_line <= end:
                    if file_path not in file_hits or score > file_hits[file_path]:
                        file_hits[file_path] = score
                    break

        # Sort files by highest score
        ranked_files = sorted(file_hits.items(), key=lambda x: x[1], reverse=True)

        # Round 3: Load raw content for top files within budget
        loaded_files: list[dict[str, Any]] = []
        total_tokens = 0
        remaining_budget = max_tokens

        for file_path, score in ranked_files:
            content = self._get_file_content(file_path)
            if content is None:
                continue

            file_tokens = count_tokens(content)

            if file_tokens <= remaining_budget:
                loaded_files.append(
                    {
                        "path": file_path,
                        "content": content,
                        "token_count": file_tokens,
                        "relevance_score": round(score, 4),
                        "truncated": False,
                    }
                )
                total_tokens += file_tokens
                remaining_budget -= file_tokens
            elif remaining_budget > 200:
                # Partial load with remaining budget
                truncated_content = content[: remaining_budget * 4]
                trunc_tokens = count_tokens(truncated_content)
                loaded_files.append(
                    {
                        "path": file_path,
                        "content": truncated_content,
                        "token_count": trunc_tokens,
                        "relevance_score": round(score, 4),
                        "truncated": True,
                    }
                )
                total_tokens += trunc_tokens
                remaining_budget -= trunc_tokens
                break
            else:
                break

        # Build ranked section summaries (Round 2 results without full content)
        ranked_summaries = [
            {
                "title": section.title,
                "relevance_score": round(score, 4),
                "start_line": section.start_line,
                "end_line": section.end_line,
            }
            for section, score in top_sections
        ]

        response = {
            "query": query,
            "rounds": {
                "sections_scan": {
                    "total_sections": len(all_sections),
                    "total_files": len(self.index.files),
                },
                "ranked_search": {
                    "top_sections": ranked_summaries,
                    "search_mode": search_mode.value,
                },
                "raw_load": {
                    "files_loaded": len(loaded_files),
                    "total_tokens": total_tokens,
                    "max_tokens": max_tokens,
                    "documents": loaded_files,
                },
            },
        }

        return ToolResult(
            data=response,
            input_tokens=count_tokens(query),
            output_tokens=total_tokens,
        )

    # ============ Phase 13: REPL Context Bridge ============

    async def _handle_repl_context(self, params: dict[str, Any]) -> ToolResult:
        """Handle rlm_repl_context - package project context for REPL consumption.

        Returns project context as a structured Python-ready dict plus helper
        code that can be injected into an rlm-runtime REPL session via
        set_repl_context. Optionally filters by a relevance query.

        Plan-gating: PRO+
        """
        if self.plan not in REPL_CONTEXT_PLANS:
            return ToolResult(
                data={
                    "error": "rlm_repl_context requires a Pro plan or higher.",
                    "current_plan": self.plan.value,
                    "upgrade_url": "https://app.snipara.com/settings/billing",
                },
                input_tokens=0,
                output_tokens=0,
            )

        if not self.index:
            return ToolResult(
                data={"error": "No documentation loaded"},
                input_tokens=0,
                output_tokens=0,
            )

        query = params.get("query", "")
        max_tokens = params.get("max_tokens", 8000)
        include_helpers = params.get("include_helpers", True)

        # Build file map
        files: dict[str, dict[str, Any]] = {}
        total_tokens = 0

        if query:
            # Query-driven: score sections and load top files
            search_mode_str = params.get("search_mode", self.settings.search_mode)
            try:
                search_mode = SearchMode(search_mode_str)
            except ValueError:
                search_mode = SearchMode.KEYWORD

            scored_sections = await self._score_sections(query, search_mode)

            # Map to files by highest score
            file_scores: dict[str, float] = {}
            for section, score in scored_sections:
                for fp, (start, end) in self.index.file_boundaries.items():
                    if section.start_line >= start + 1 and section.end_line <= end:
                        if fp not in file_scores or score > file_scores[fp]:
                            file_scores[fp] = score
                        break

            # Load files in score order within budget
            for fp, score in sorted(file_scores.items(), key=lambda x: x[1], reverse=True):
                content = self._get_file_content(fp)
                if content is None:
                    continue
                ftokens = count_tokens(content)
                if total_tokens + ftokens > max_tokens:
                    if total_tokens == 0:
                        # At least include first file truncated
                        content = content[: max_tokens * 4]
                        ftokens = count_tokens(content)
                        files[fp] = {
                            "content": content,
                            "tokens": ftokens,
                            "relevance": round(score, 4),
                            "truncated": True,
                        }
                        total_tokens += ftokens
                    break
                files[fp] = {
                    "content": content,
                    "tokens": ftokens,
                    "relevance": round(score, 4),
                    "truncated": False,
                }
                total_tokens += ftokens
        else:
            # No query: load all files within budget
            for fp in self.index.files:
                content = self._get_file_content(fp)
                if content is None:
                    continue
                ftokens = count_tokens(content)
                if total_tokens + ftokens > max_tokens:
                    break
                files[fp] = {
                    "content": content,
                    "tokens": ftokens,
                    "truncated": False,
                }
                total_tokens += ftokens

        # Build section map
        section_map = [
            {
                "title": s.title,
                "file": next(
                    (
                        fp
                        for fp, (start, end) in self.index.file_boundaries.items()
                        if s.start_line >= start + 1 and s.end_line <= end
                    ),
                    "unknown",
                ),
                "lines": [s.start_line, s.end_line],
            }
            for s in self.index.sections
        ]

        # Helper functions as executable Python code for REPL injection
        helpers_code = ""
        if include_helpers:
            helpers_code = '''import re as _re

def peek(path, start=None, end=None):
    """View file content or a line range."""
    f = context.get("files", {}).get(path)
    if not f:
        avail = list(context.get("files", {}).keys())
        return {"error": f"Not found: {path}", "available": avail[:10]}
    lines = f["content"].split("\\n")
    if start is not None:
        end = end or start + 20
        lines = lines[max(0, start - 1):end]
    return "\\n".join(lines)

def grep(pattern, path=None):
    """Search across loaded files. Returns {file: [matching lines]}."""
    results = {}
    files = context.get("files", {})
    targets = {path: files[path]} if path and path in files else files
    for fp, fdata in targets.items():
        matches = []
        for i, line in enumerate(fdata["content"].split("\\n"), 1):
            if _re.search(pattern, line):
                matches.append(f"{i}: {line}")
        if matches:
            results[fp] = matches
    return results

def sections(path=None):
    """List sections, optionally filtered to a file."""
    secs = context.get("sections", [])
    if path:
        secs = [s for s in secs if s.get("file") == path]
    return secs

def files():
    """List loaded files with token counts."""
    return {fp: {"tokens": f["tokens"], "truncated": f.get("truncated", False)}
            for fp, f in context.get("files", {}).items()}
'''

        response = {
            "context_data": {
                "files": files,
                "sections": section_map,
                "total_files_in_project": len(self.index.files),
                "loaded_files": len(files),
            },
            "setup_code": helpers_code,
            "total_tokens": total_tokens,
            "usage_hint": (
                "Inject into REPL: call set_repl_context(key='context', "
                "value=<context_data>), then execute_python with setup_code, "
                "then use peek(), grep(), sections(), files() helpers."
            ),
        }

        return ToolResult(
            data=response,
            input_tokens=count_tokens(query) if query else 0,
            output_tokens=total_tokens,
        )

    # ============ PHASE 14: PASS-BY-REFERENCE HANDLERS ============

    async def _handle_get_chunk(self, params: dict[str, Any]) -> ToolResult:
        """
        Handle rlm_get_chunk - retrieve full content by chunk ID.

        This tool is used with pass-by-reference mode of rlm_context_query.
        When return_references=True, rlm_context_query returns chunk IDs and
        previews. Use this tool to retrieve the full content of specific chunks.

        Args:
            params: Dict containing:
                - chunk_id: The chunk ID from rlm_context_query (required)

        Returns:
            ToolResult with GetChunkResult containing:
                - chunk_id: The requested chunk ID
                - found: Whether the chunk was found
                - title: Section title
                - content: Full section content
                - file: Source file path
                - lines: Start and end line numbers
                - token_count: Content token count
                - message: Optional message (e.g., "Chunk not found or expired")
        """
        chunk_id = params.get("chunk_id", "")

        if not chunk_id:
            return ToolResult(
                data=GetChunkResult(
                    chunk_id="",
                    found=False,
                    message="chunk_id is required",
                ).model_dump(),
                input_tokens=0,
                output_tokens=0,
            )

        # Retrieve from cache
        chunk_data = self.get_chunk(chunk_id)

        if not chunk_data:
            return ToolResult(
                data=GetChunkResult(
                    chunk_id=chunk_id,
                    found=False,
                    message="Chunk not found or expired. Re-run rlm_context_query with return_references=True.",
                ).model_dump(),
                input_tokens=0,
                output_tokens=0,
            )

        result = GetChunkResult(
            chunk_id=chunk_id,
            found=True,
            title=chunk_data.get("title", ""),
            content=chunk_data.get("content", ""),
            file=chunk_data.get("file", ""),
            lines=chunk_data.get("lines"),
            token_count=chunk_data.get("token_count", 0),
        )

        return ToolResult(
            data=result.model_dump(),
            input_tokens=0,
            output_tokens=result.token_count,
        )
