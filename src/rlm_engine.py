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
from datetime import datetime
from typing import Any

import tiktoken

from .db import get_db
from .models import (
    ContextQueryResult,
    ContextSection,
    DecomposeResult,
    DecomposeStrategy,
    DeleteSummaryResult,
    DocumentCategoryEnum,
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
from .services.cache import get_cache
from .services.chunker import get_chunker
from .services.embeddings import get_embeddings_service
from .services.shared_context import (
    allocate_shared_context_budget,
    compute_context_hash,
    DocumentCategory,
    get_shared_prompt_templates,
    load_project_shared_context,
    merge_shared_context_with_project_docs,
)
from .services.agent_memory import (
    store_memory,
    semantic_recall,
    list_memories,
    delete_memories,
)
from .services.agent_limits import (
    check_memory_limits,
    check_swarm_limits,
)
from .services.swarm_coordinator import (
    create_swarm,
    join_swarm,
    leave_swarm,
    acquire_claim,
    release_claim,
    get_state,
    set_state,
    create_task,
    claim_task,
    complete_task,
)
from .services.swarm_events import (
    broadcast_event,
    get_recent_events,
)

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
    ):
        """Initialize the engine for a project.

        Args:
            project_id: The project ID
            plan: The user's subscription plan (affects feature access)
            settings: Project automation settings from dashboard (optional)
        """
        self.project_id = project_id
        self.plan = plan
        self.index: DocumentationIndex | None = None
        self.session_context: str = ""
        self._chunks_available: bool | None = None  # Cache for chunk availability check

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

    async def load_documents(self) -> None:
        """Load and index project documents from database."""
        db = await get_db()

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

            # Parse sections from this document
            self._parse_sections(doc_lines, line_offset, doc.path)

    def _parse_sections(self, lines: list[str], offset: int, file_path: str) -> None:
        """Parse markdown sections from lines."""
        if self.index is None:
            return

        current_section: Section | None = None
        section_content: list[str] = []

        for i, line in enumerate(lines):
            # Check for markdown headers
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = "\n".join(section_content)
                    current_section.end_line = offset + i - 1
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

        # Save last section
        if current_section:
            current_section.content = "\n".join(section_content)
            current_section.end_line = offset + len(lines)
            self.index.sections.append(current_section)

    def _generate_section_id(self, title: str) -> str:
        """Generate a unique section ID from title."""
        # Clean the title for use as an ID
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", title)
        clean = re.sub(r"\s+", "_", clean.strip())
        clean = clean.upper()[:20]
        return f"[{clean}]"

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
                '''
                SELECT COUNT(*) as count FROM document_chunks dc
                JOIN documents d ON dc."documentId" = d.id
                WHERE d."projectId" = $1
                LIMIT 1
                ''',
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
        }

        handler = handlers.get(tool)
        if not handler:
            raise ValueError(f"Unknown tool: {tool}")

        return await handler(params)

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
                response_parts.append(f"\n## {section.title} (lines {section.start_line}-{section.end_line})")
                # Truncate content if too long
                content = section.content[:2000] + "..." if len(section.content) > 2000 else section.content
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
                        results.append({
                            "line_number": i,
                            "content": line.strip()[:500],  # Limit content length in results
                        })
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
        except asyncio.TimeoutError:
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

        # Generate a key for this context
        context_key = hashlib.md5(context.encode()).hexdigest()[:8]

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
        await db.sessioncontext.delete_many(
            where={"projectId": self.project_id}
        )

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
        """Handle rlm_sections - list all documentation sections."""
        if not self.index:
            return ToolResult(data="No documentation loaded", input_tokens=0, output_tokens=0)

        sections = [
            SectionInfo(
                id=s.id,
                title=s.title,
                start_line=s.start_line,
                end_line=s.end_line,
            ).model_dump()
            for s in self.index.sections
        ]

        response = {
            "total_sections": len(sections),
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
        query = params.get("query", "")
        # Use dashboard settings as defaults, allow request params to override
        max_tokens = params.get("max_tokens", self.settings.max_tokens_per_query)
        search_mode_str = params.get("search_mode", self.settings.search_mode)
        include_metadata = params.get("include_metadata", True)
        prefer_summaries = params.get("prefer_summaries", self.settings.include_summaries)
        include_shared_context = params.get("include_shared_context", True)
        shared_context_budget_percent = params.get("shared_context_budget_percent", 30)

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

        # Account for session context tokens if present
        session_context_tokens = 0
        session_context_included = False
        remaining_budget = max_tokens

        if self.session_context:
            session_context_tokens = count_tokens(self.session_context)
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

                    # Convert shared documents to ContextSection format
                    for doc in allocated_docs:
                        try:
                            cat_enum = DocumentCategoryEnum(doc.category.value)
                        except ValueError:
                            cat_enum = DocumentCategoryEnum.BEST_PRACTICES

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

        # Build result - shared context sections come FIRST
        all_sections = shared_context_sections + selected_sections
        search_mode_downgraded = original_search_mode != search_mode

        # Include system instructions (custom from project or default)
        instructions = self.settings.system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS

        result = ContextQueryResult(
            sections=all_sections,
            total_tokens=total_tokens,
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

        keywords = re.findall(r"\w+", query.lower())
        scored: list[tuple[Section, float]] = []

        # Calculate keyword scores for all sections (always in-memory, fast)
        keyword_scores: dict[str, float] = {}
        for section in self.index.sections:
            keyword_scores[section.id] = self._calculate_keyword_score(section, keywords)

        # Handle different search modes
        if search_mode == SearchMode.KEYWORD:
            # Pure keyword search
            for section in self.index.sections:
                score = keyword_scores[section.id]
                if score > 0:
                    scored.append((section, score))

        elif search_mode == SearchMode.SEMANTIC:
            # Pure semantic search - try pre-computed chunks first
            use_chunks = await self._has_precomputed_chunks()

            if use_chunks:
                semantic_scores = await self._calculate_semantic_scores_from_chunks(query)
                logger.info(f"Using pre-computed chunks for semantic search (project: {self.project_id})")
            else:
                semantic_scores = self._calculate_semantic_scores(query)
                logger.info(f"Using on-the-fly embedding (no chunks for project: {self.project_id})")

            for section in self.index.sections:
                score = semantic_scores.get(section.id, 0.0) * 100  # Scale to 0-100
                if score > 10:  # Minimum similarity threshold
                    scored.append((section, score))

        elif search_mode == SearchMode.HYBRID:
            # Combined keyword + semantic search - try pre-computed chunks first
            use_chunks = await self._has_precomputed_chunks()

            if use_chunks:
                semantic_scores = await self._calculate_semantic_scores_from_chunks(query)
                logger.info(f"Using pre-computed chunks for hybrid search (project: {self.project_id})")
            else:
                semantic_scores = self._calculate_semantic_scores(query)
                logger.info(f"Using on-the-fly embedding for hybrid search (project: {self.project_id})")

            for section in self.index.sections:
                keyword_score = keyword_scores[section.id]
                semantic_score = semantic_scores.get(section.id, 0.0) * 100

                # Weighted combination: 40% keyword, 60% semantic
                # This gives more weight to semantic understanding
                combined_score = (keyword_score * 0.4) + (semantic_score * 0.6)

                # Boost if both signals agree
                if keyword_score > 0 and semantic_score > 20:
                    combined_score *= 1.2

                if combined_score > 5:  # Minimum threshold
                    scored.append((section, combined_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_semantic_scores(self, query: str) -> dict[str, float]:
        """
        Calculate semantic similarity scores for all sections.

        Uses embedding cosine similarity to find semantically similar sections.
        """
        if not self.index or not self.index.sections:
            return {}

        try:
            embeddings_service = get_embeddings_service()

            # Generate query embedding
            query_embedding = embeddings_service.embed_text(query)

            # Generate section embeddings (could be cached in production)
            section_texts = [
                f"{s.title}\n{s.content[:500]}"  # Use title + first 500 chars
                for s in self.index.sections
            ]
            section_embeddings = embeddings_service.embed_texts(section_texts)

            # Calculate similarities
            similarities = embeddings_service.cosine_similarity(
                query_embedding, section_embeddings
            )

            # Map to section IDs
            return {
                section.id: similarity
                for section, similarity in zip(self.index.sections, similarities)
            }
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to empty scores: {e}")
            return {}

    def _calculate_keyword_score(self, section: Section, keywords: list[str]) -> float:
        """
        Calculate keyword relevance score for a section.

        Scoring factors:
        - Title matches weighted 3x
        - Content matches weighted 1x
        - Exact phrase matches weighted 2x
        - Section level bonus (higher level = more important)
        """
        score = 0.0
        title_lower = section.title.lower()
        content_lower = section.content.lower()

        for keyword in keywords:
            if len(keyword) < 2:  # Skip very short words
                continue

            # Title matches (3x weight)
            title_count = title_lower.count(keyword)
            score += title_count * 3.0

            # Content matches (1x weight)
            content_count = content_lower.count(keyword)
            score += content_count * 1.0

        # Bonus for higher-level sections (h1, h2 more important)
        level_bonus = max(0, 4 - section.level) * 0.5
        score += level_bonus if score > 0 else 0

        return score

    def _find_file_for_section(self, section: Section) -> str:
        """Find which file a section belongs to based on line numbers."""
        if not self.index:
            return "unknown"

        cumulative_lines = 0
        for file_path in self.index.files:
            # This is a simplified approach - in production, we'd track
            # file boundaries more precisely during parsing
            if section.start_line > cumulative_lines:
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
        max_depth = params.get("max_depth", 2)
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

        # Build sub-queries from terms with matching sections
        sub_queries: list[SubQuery] = []
        for i, (term, sections) in enumerate(term_sections.items(), start=1):
            # Estimate tokens based on section content
            estimated_tokens = sum(
                min(count_tokens(s.content), 1500) for s in sections[:3]
            )

            sub_queries.append(SubQuery(
                id=i,
                query=term,
                priority=i,  # Earlier terms have higher priority
                estimated_tokens=estimated_tokens,
                key_terms=[term],
            ))

        # Analyze dependencies based on document links
        dependencies = self._analyze_document_links(term_sections)

        # Generate suggested sequence using topological sort
        suggested_sequence = self._topological_sort(
            len(sub_queries), dependencies
        )

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
                queries.append({
                    "query": q.get("query", ""),
                    "max_tokens": q.get("max_tokens"),
                })

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
        async def execute_single_query(
            query_item: dict[str, Any]
        ) -> MultiQueryResultItem:
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

        steps.append(PlanStep(
            step=1,
            action="decompose",
            params={
                "query": query,
                "max_depth": 2,
            },
            depends_on=[],
            expected_output="sub_queries",
        ))

        # Step 2: Execute sub-queries based on strategy
        if strategy == PlanStrategy.BREADTH_FIRST:
            # Execute all sub-queries at same level first
            steps.append(PlanStep(
                step=2,
                action="multi_query",
                params={
                    "queries": "$step1.sub_queries",
                    "max_tokens": max_tokens - 2000,  # Reserve some for synthesis
                    "search_mode": "hybrid",
                },
                depends_on=[1],
                expected_output="sections",
            ))
        elif strategy == PlanStrategy.DEPTH_FIRST:
            # Execute sub-queries one at a time, depth first
            steps.append(PlanStep(
                step=2,
                action="context_query",
                params={
                    "query": "$step1.sub_queries[0].query",
                    "max_tokens": max_tokens // 4,
                    "search_mode": "hybrid",
                },
                depends_on=[1],
                expected_output="sections",
            ))
            steps.append(PlanStep(
                step=3,
                action="multi_query",
                params={
                    "queries": "$step1.sub_queries[1:]",
                    "max_tokens": (max_tokens * 3) // 4 - 1000,
                    "search_mode": "hybrid",
                },
                depends_on=[1, 2],
                expected_output="sections",
            ))
        else:  # RELEVANCE_FIRST
            # Execute most relevant sub-query first, then batch rest
            steps.append(PlanStep(
                step=2,
                action="context_query",
                params={
                    "query": "$step1.sub_queries[0].query",
                    "max_tokens": max_tokens // 3,
                    "search_mode": "hybrid",
                },
                depends_on=[1],
                expected_output="sections",
            ))
            steps.append(PlanStep(
                step=3,
                action="multi_query",
                params={
                    "queries": "$step1.sub_queries[1:5]",  # Next 4 most relevant
                    "max_tokens": (max_tokens * 2) // 3 - 1000,
                    "search_mode": "hybrid",
                },
                depends_on=[1],
                expected_output="sections",
            ))

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

    def _topological_sort(
        self, num_queries: int, dependencies: list[tuple[int, int]]
    ) -> list[int]:
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
            updated = await db.documentsummary.update(
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
                data={"error": "At least one of summary_id, document_path, or summary_type is required"},
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
                d for d in shared_ctx.documents
                if d.category in category_filter
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

            doc_infos.append(SharedDocumentInfo(
                id=doc.id,
                title=doc.title,
                category=cat_enum,
                token_count=doc.token_count,
                collection_name=doc.collection_name,
                tags=doc.tags,
            ))

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
            template_infos.append(PromptTemplateInfo(
                id=t["id"],
                name=t["name"],
                slug=t["slug"],
                description=t.get("description"),
                prompt=t["prompt"],
                variables=t.get("variables", []),
                category=t["category"],
                collection_name=t["collection_name"],
            ))
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
                rendered_prompt = rendered_prompt.replace(
                    placeholder, str(variables[var_name])
                )
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
        existing = await db.document.find_first(
            where={"projectId": self.project_id, "path": path}
        )

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
        existing_docs = await db.document.find_many(
            where={"projectId": self.project_id}
        )
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
