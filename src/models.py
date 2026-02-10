"""Pydantic models for RLM MCP Server request/response schemas."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ============ ENUMS ============


class ToolName(StrEnum):
    """Available RLM tools."""

    RLM_ASK = "rlm_ask"
    RLM_SEARCH = "rlm_search"
    RLM_INJECT = "rlm_inject"
    RLM_CONTEXT = "rlm_context"
    RLM_CLEAR_CONTEXT = "rlm_clear_context"
    RLM_STATS = "rlm_stats"
    RLM_SECTIONS = "rlm_sections"
    RLM_READ = "rlm_read"
    RLM_CONTEXT_QUERY = "rlm_context_query"
    # Phase 4.5: Recursive Context Tools
    RLM_DECOMPOSE = "rlm_decompose"
    RLM_MULTI_QUERY = "rlm_multi_query"
    RLM_MULTI_PROJECT_QUERY = "rlm_multi_project_query"
    RLM_PLAN = "rlm_plan"
    # Phase 4.6: Summary Storage Tools
    RLM_STORE_SUMMARY = "rlm_store_summary"
    RLM_GET_SUMMARIES = "rlm_get_summaries"
    RLM_DELETE_SUMMARY = "rlm_delete_summary"
    # Phase 7: Shared Context Tools
    RLM_SHARED_CONTEXT = "rlm_shared_context"
    RLM_LIST_TEMPLATES = "rlm_list_templates"
    RLM_GET_TEMPLATE = "rlm_get_template"
    RLM_LIST_COLLECTIONS = "rlm_list_collections"
    RLM_UPLOAD_SHARED_DOCUMENT = "rlm_upload_shared_document"
    # Phase 8.2: Agent Memory Tools
    RLM_REMEMBER = "rlm_remember"
    RLM_RECALL = "rlm_recall"
    RLM_MEMORIES = "rlm_memories"
    RLM_FORGET = "rlm_forget"
    # Phase 9.1: Multi-Agent Swarm Tools
    RLM_SWARM_CREATE = "rlm_swarm_create"
    RLM_SWARM_JOIN = "rlm_swarm_join"
    RLM_CLAIM = "rlm_claim"
    RLM_RELEASE = "rlm_release"
    RLM_STATE_GET = "rlm_state_get"
    RLM_STATE_SET = "rlm_state_set"
    RLM_BROADCAST = "rlm_broadcast"
    RLM_TASK_CREATE = "rlm_task_create"
    RLM_TASK_CLAIM = "rlm_task_claim"
    RLM_TASK_COMPLETE = "rlm_task_complete"
    # Phase 10: Document Sync Tools
    RLM_UPLOAD_DOCUMENT = "rlm_upload_document"
    RLM_SYNC_DOCUMENTS = "rlm_sync_documents"
    RLM_SETTINGS = "rlm_settings"
    # Phase 11: Access Control Tools
    RLM_REQUEST_ACCESS = "rlm_request_access"
    # Phase 12: RLM Orchestration Tools
    RLM_LOAD_DOCUMENT = "rlm_load_document"
    RLM_LOAD_PROJECT = "rlm_load_project"
    RLM_ORCHESTRATE = "rlm_orchestrate"
    # Phase 13: REPL Context Bridge
    RLM_REPL_CONTEXT = "rlm_repl_context"
    # Phase 14: Pass-by-Reference (reduce hallucination)
    RLM_GET_CHUNK = "rlm_get_chunk"


class SearchMode(StrEnum):
    """Search mode for context queries."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"  # Future: embedding-based
    HYBRID = "hybrid"  # Future: keyword + semantic


class Plan(StrEnum):
    """Subscription plans."""

    FREE = "FREE"
    PRO = "PRO"
    TEAM = "TEAM"
    ENTERPRISE = "ENTERPRISE"


# ============ REQUEST MODELS ============


class MCPRequest(BaseModel):
    """MCP tool execution request."""

    tool: ToolName = Field(..., description="The RLM tool to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class AskParams(BaseModel):
    """Parameters for rlm_ask tool."""

    question: str = Field(..., description="The question to ask about the documentation")


class SearchParams(BaseModel):
    """Parameters for rlm_search tool."""

    pattern: str = Field(..., description="Regex pattern to search for")
    max_results: int = Field(default=20, description="Maximum results to return")


class InjectParams(BaseModel):
    """Parameters for rlm_inject tool."""

    context: str = Field(..., description="The context to inject")
    append: bool = Field(default=False, description="Append to existing context")


class ReadParams(BaseModel):
    """Parameters for rlm_read tool."""

    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")


class ContextQueryParams(BaseModel):
    """Parameters for rlm_context_query tool - the main context optimization tool."""

    query: str = Field(..., description="The query/question to get context for")
    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=100000,
        description="Maximum tokens to return (respects client's token budget)",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.KEYWORD,
        description="Search strategy: keyword, semantic (future), or hybrid (future)",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include file paths, line numbers, and relevance scores",
    )
    prefer_summaries: bool = Field(
        default=False,
        description="Prefer stored summaries over full document content when available",
    )
    return_references: bool = Field(
        default=False,
        description="Return chunk references (IDs + previews) instead of full content. "
        "Use rlm_get_chunk to retrieve full content by ID. Reduces hallucination by "
        "maintaining clear source attribution.",
    )


class MultiProjectQueryParams(BaseModel):
    """Parameters for rlm_multi_project_query tool."""

    query: str = Field(..., description="The query/question to get context for")
    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=100000,
        description="Maximum tokens to return across all projects",
    )
    per_project_limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum sections to return per project",
    )
    project_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of project IDs or slugs to include",
    )
    exclude_project_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of project IDs or slugs to exclude",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.KEYWORD,
        description="Search strategy: keyword, semantic, or hybrid",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include file paths, line numbers, and relevance scores",
    )
    prefer_summaries: bool = Field(
        default=False,
        description="Prefer stored summaries when available",
    )


# ============ RESPONSE MODELS ============


class UsageInfo(BaseModel):
    """Usage information for a request."""

    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    latency_ms: int = Field(..., description="Request latency in milliseconds")


class MCPResponse(BaseModel):
    """MCP tool execution response."""

    success: bool = Field(..., description="Whether the request was successful")
    result: Any | None = Field(default=None, description="Tool result data")
    error: str | None = Field(default=None, description="Error message if failed")
    usage: UsageInfo = Field(..., description="Usage information")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadyResponse(BaseModel):
    """Readiness check response with component statuses."""

    status: str = Field(default="ready")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: dict[str, bool] = Field(default_factory=dict, description="Component health checks")


class LimitsInfo(BaseModel):
    """Usage limits information."""

    current: int = Field(..., description="Current usage count")
    max: int = Field(..., description="Maximum allowed (-1 for unlimited)")
    exceeded: bool = Field(..., description="Whether limits are exceeded")
    resets_at: datetime | None = Field(default=None, description="When limits reset")


class ProjectContext(BaseModel):
    """Project context information."""

    key: str
    value: str
    created_at: datetime
    expires_at: datetime | None = None


class DocumentInfo(BaseModel):
    """Document information."""

    path: str
    size: int
    hash: str
    updated_at: datetime


class StatsResponse(BaseModel):
    """Documentation statistics response."""

    files_loaded: int
    total_lines: int
    total_characters: int
    sections: int
    files: list[str]
    project_id: str


class SectionInfo(BaseModel):
    """Section information."""

    id: str
    title: str
    start_line: int
    end_line: int


# ============ CONTEXT QUERY RESPONSE MODELS ============


class ContextSection(BaseModel):
    """A section of relevant context returned by rlm_context_query."""

    title: str = Field(..., description="Section title/heading")
    content: str = Field(..., description="Section content (may be truncated)")
    file: str = Field(..., description="Source file path")
    lines: tuple[int, int] = Field(..., description="Start and end line numbers")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    token_count: int = Field(..., ge=0, description="Token count for this section")
    truncated: bool = Field(
        default=False, description="Whether content was truncated to fit budget"
    )


class ContextSectionRef(BaseModel):
    """A chunk reference returned when return_references=True.

    Pass-by-reference architecture: returns chunk IDs + previews instead of full content.
    Use rlm_get_chunk(chunk_id) to retrieve full content when needed.
    This reduces hallucination by maintaining clear source attribution.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier for retrieval via rlm_get_chunk")
    title: str = Field(..., description="Section title/heading")
    preview: str = Field(..., description="First ~100 characters of content for context")
    file: str = Field(..., description="Source file path")
    lines: tuple[int, int] = Field(..., description="Start and end line numbers")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    token_count: int = Field(..., ge=0, description="Full content token count (for budget planning)")
    keyword_score: float = Field(default=0.0, description="Keyword match score (for debugging)")
    semantic_score: float = Field(default=0.0, description="Semantic similarity score (for debugging)")


class GetChunkResult(BaseModel):
    """Result of rlm_get_chunk tool - retrieves full content by chunk ID."""

    chunk_id: str = Field(..., description="The chunk ID that was requested")
    title: str = Field(..., description="Section title/heading")
    content: str = Field(..., description="Full section content")
    file: str = Field(..., description="Source file path")
    lines: tuple[int, int] = Field(..., description="Start and end line numbers")
    token_count: int = Field(..., ge=0, description="Token count for this section")
    found: bool = Field(default=True, description="Whether the chunk was found")


class ContextQueryResult(BaseModel):
    """Result of rlm_context_query tool - optimized context for the client's LLM."""

    sections: list[ContextSection] = Field(
        default_factory=list, description="Ranked list of relevant sections (when return_references=False)"
    )
    section_refs: list[ContextSectionRef] = Field(
        default_factory=list,
        description="Chunk references with IDs + previews (when return_references=True). "
        "Use rlm_get_chunk(chunk_id) to retrieve full content.",
    )
    references_mode: bool = Field(
        default=False,
        description="True if returning references instead of full content",
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens returned (preview tokens if references_mode)")
    max_tokens: int = Field(..., description="Token budget that was requested")
    query: str = Field(..., description="Original query")
    search_mode: SearchMode = Field(..., description="Search mode used")
    search_mode_downgraded: bool = Field(
        default=False,
        description="Whether search mode was downgraded due to plan restrictions",
    )
    session_context_included: bool = Field(
        default=False, description="Whether session context was prepended"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Additional sections that may be relevant but didn't fit",
    )
    summaries_used: int = Field(
        default=0,
        ge=0,
        description="Number of stored summaries used instead of full content",
    )
    timing: dict[str, int] | None = Field(
        default=None,
        description="Timing breakdown in milliseconds (embed_ms, search_ms, score_ms, total_ms)",
    )
    system_instructions: str | None = Field(
        default=None,
        description="System instructions to guide LLM behavior when using Snipara tools",
    )
    shared_context_included: bool = Field(
        default=False,
        description="Whether shared best practices were included (from linked collections)",
    )
    shared_context_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens from shared context collections",
    )
    first_query_tips_included: bool = Field(
        default=False,
        description="Whether first-query tool tips were included (shown only on first query)",
    )
    # Smart Routing Hints (helps clients decide when to use RLM-Runtime)
    routing_recommendation: str | None = Field(
        default=None,
        description="Recommended execution mode: 'direct' (use context as-is) or 'rlm_runtime' (use RLM for complex reasoning)",
    )
    routing_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for the routing recommendation",
    )
    routing_reason: str | None = Field(
        default=None,
        description="Human-readable explanation for routing recommendation",
    )
    query_complexity: str | None = Field(
        default=None,
        description="Assessed query complexity: 'simple', 'moderate', or 'complex'",
    )


# ============ RECURSIVE CONTEXT MODELS (Phase 4.5) ============


class DecomposeStrategy(StrEnum):
    """Strategy for query decomposition."""

    AUTO = "auto"  # Let the engine decide
    TERM_BASED = "term_based"  # Extract key terms
    STRUCTURAL = "structural"  # Follow document structure


class PlanStrategy(StrEnum):
    """Strategy for execution planning."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RELEVANCE_FIRST = "relevance_first"


class DecomposeParams(BaseModel):
    """Parameters for rlm_decompose tool."""

    query: str = Field(..., description="The complex question to decompose")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum recursion depth")
    strategy: DecomposeStrategy = Field(
        default=DecomposeStrategy.AUTO, description="Decomposition strategy"
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Optional hints to guide decomposition",
    )


class SubQuery(BaseModel):
    """A sub-query generated by decomposition."""

    id: int = Field(..., description="Sub-query ID (1-indexed)")
    query: str = Field(..., description="The sub-query text")
    priority: int = Field(default=1, ge=1, le=10, description="Priority (1=highest)")
    estimated_tokens: int = Field(default=1000, ge=0, description="Estimated tokens for this query")
    key_terms: list[str] = Field(default_factory=list, description="Key terms identified")


class DecomposeResult(BaseModel):
    """Result of rlm_decompose tool."""

    original_query: str = Field(..., description="The original query")
    sub_queries: list[SubQuery] = Field(default_factory=list, description="Generated sub-queries")
    dependencies: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Dependencies between sub-queries [(a, b) means a should be read before b]",
    )
    suggested_sequence: list[int] = Field(
        default_factory=list, description="Suggested execution order (query IDs)"
    )
    total_estimated_tokens: int = Field(
        default=0, ge=0, description="Total estimated tokens for all sub-queries"
    )
    strategy_used: DecomposeStrategy = Field(..., description="Strategy that was used")


class MultiQueryItem(BaseModel):
    """A single query in a multi-query batch."""

    query: str = Field(..., description="The query text")
    max_tokens: int | None = Field(default=None, description="Optional per-query token budget")


class MultiQueryParams(BaseModel):
    """Parameters for rlm_multi_query tool."""

    queries: list[MultiQueryItem] = Field(
        ..., min_length=1, max_length=10, description="List of queries to execute"
    )
    max_tokens: int = Field(default=8000, ge=500, le=50000, description="Total token budget")
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID, description="Search mode for all queries"
    )


class MultiQueryResultItem(BaseModel):
    """Result for a single query in a multi-query batch."""

    query: str = Field(..., description="The original query")
    sections: list[ContextSection] = Field(default_factory=list, description="Relevant sections")
    tokens_used: int = Field(default=0, ge=0, description="Tokens used for this query")
    success: bool = Field(default=True, description="Whether query succeeded")
    error: str | None = Field(default=None, description="Error message if failed")


class MultiQueryResult(BaseModel):
    """Result of rlm_multi_query tool."""

    results: list[MultiQueryResultItem] = Field(
        default_factory=list, description="Results for each query"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    queries_executed: int = Field(default=0, ge=0, description="Number of queries executed")
    queries_skipped: int = Field(default=0, ge=0, description="Queries skipped due to budget")
    search_mode: SearchMode = Field(..., description="Search mode used")


class PlanStep(BaseModel):
    """A step in an execution plan."""

    step: int = Field(..., ge=1, description="Step number")
    action: str = Field(..., description="Action to perform: decompose, context_query, multi_query")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    depends_on: list[int] = Field(default_factory=list, description="Steps this step depends on")
    expected_output: str = Field(default="sections", description="Expected output type")


class PlanParams(BaseModel):
    """Parameters for rlm_plan tool."""

    query: str = Field(..., description="The complex question to plan for")
    strategy: PlanStrategy = Field(
        default=PlanStrategy.RELEVANCE_FIRST, description="Execution strategy"
    )
    max_tokens: int = Field(default=16000, ge=1000, le=100000, description="Total token budget")


class PlanResult(BaseModel):
    """Result of rlm_plan tool."""

    plan_id: str = Field(..., description="Unique plan identifier")
    query: str = Field(..., description="The original query")
    steps: list[PlanStep] = Field(default_factory=list, description="Execution steps")
    estimated_total_tokens: int = Field(default=0, ge=0, description="Estimated total tokens")
    strategy: PlanStrategy = Field(..., description="Strategy used")
    estimated_queries: int = Field(default=0, ge=0, description="Estimated number of queries")


# ============ SUMMARY STORAGE MODELS (Phase 4.6) ============


class SummaryType(StrEnum):
    """Type of summary stored."""

    CONCISE = "concise"  # Brief 1-2 sentence summary
    DETAILED = "detailed"  # Full multi-paragraph summary
    TECHNICAL = "technical"  # Technical details focus
    KEYWORDS = "keywords"  # Key terms and concepts
    CUSTOM = "custom"  # User-defined summary type


class StoreSummaryParams(BaseModel):
    """Parameters for rlm_store_summary tool."""

    document_path: str = Field(..., description="Path to the document (relative to project root)")
    summary: str = Field(..., min_length=1, description="The summary text to store")
    summary_type: SummaryType = Field(default=SummaryType.CONCISE, description="Type of summary")
    section_id: str | None = Field(
        default=None, description="Optional section identifier for partial summaries"
    )
    line_start: int | None = Field(default=None, ge=1, description="Start line for section summary")
    line_end: int | None = Field(default=None, ge=1, description="End line for section summary")
    generated_by: str | None = Field(
        default=None,
        description="Model that generated the summary (e.g., 'claude-3.5-sonnet')",
    )


class StoreSummaryResult(BaseModel):
    """Result of rlm_store_summary tool."""

    summary_id: str = Field(..., description="Unique identifier for the stored summary")
    document_path: str = Field(..., description="Document path")
    summary_type: SummaryType = Field(..., description="Type of summary stored")
    token_count: int = Field(..., ge=0, description="Token count of the summary")
    created: bool = Field(default=True, description="True if new, False if updated existing")
    message: str = Field(..., description="Human-readable status message")


class GetSummariesParams(BaseModel):
    """Parameters for rlm_get_summaries tool."""

    document_path: str | None = Field(default=None, description="Filter by document path")
    summary_type: SummaryType | None = Field(default=None, description="Filter by summary type")
    section_id: str | None = Field(default=None, description="Filter by section ID")
    include_content: bool = Field(default=True, description="Include summary content in response")


class SummaryInfo(BaseModel):
    """Information about a stored summary."""

    summary_id: str = Field(..., description="Unique identifier")
    document_path: str = Field(..., description="Document path")
    summary_type: SummaryType = Field(..., description="Type of summary")
    section_id: str | None = Field(default=None, description="Section identifier")
    line_start: int | None = Field(default=None, description="Start line")
    line_end: int | None = Field(default=None, description="End line")
    token_count: int = Field(..., ge=0, description="Token count")
    generated_by: str | None = Field(default=None, description="Generator model")
    content: str | None = Field(
        default=None, description="Summary content (if include_content=True)"
    )
    created_at: datetime = Field(..., description="When summary was created")
    updated_at: datetime = Field(..., description="When summary was last updated")


class GetSummariesResult(BaseModel):
    """Result of rlm_get_summaries tool."""

    summaries: list[SummaryInfo] = Field(
        default_factory=list, description="List of summaries matching filters"
    )
    total_count: int = Field(default=0, ge=0, description="Total number of summaries")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens across all summaries")


class DeleteSummaryParams(BaseModel):
    """Parameters for rlm_delete_summary tool."""

    summary_id: str | None = Field(default=None, description="Specific summary ID")
    document_path: str | None = Field(default=None, description="Delete all summaries for document")
    summary_type: SummaryType | None = Field(
        default=None, description="Delete summaries of this type"
    )


class DeleteSummaryResult(BaseModel):
    """Result of rlm_delete_summary tool."""

    deleted_count: int = Field(default=0, ge=0, description="Number of summaries deleted")
    message: str = Field(..., description="Human-readable status message")


# ============ SHARED CONTEXT MODELS (Phase 7) ============


class DocumentCategoryEnum(StrEnum):
    """Document category for token budget allocation."""

    MANDATORY = "MANDATORY"
    BEST_PRACTICES = "BEST_PRACTICES"
    GUIDELINES = "GUIDELINES"
    REFERENCE = "REFERENCE"


class SharedDocumentInfo(BaseModel):
    """Information about a shared document."""

    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    category: DocumentCategoryEnum = Field(..., description="Document category")
    token_count: int = Field(..., ge=0, description="Token count")
    collection_name: str = Field(..., description="Source collection name")
    tags: list[str] = Field(default_factory=list, description="Document tags")


class SharedContextParams(BaseModel):
    """Parameters for rlm_shared_context tool."""

    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=100000,
        description="Maximum tokens for shared context",
    )
    categories: list[DocumentCategoryEnum] | None = Field(
        default=None,
        description="Filter by categories (null = all categories)",
    )
    include_content: bool = Field(
        default=True,
        description="Include document content in response",
    )


class SharedContextResult(BaseModel):
    """Result of rlm_shared_context tool."""

    documents: list[SharedDocumentInfo] = Field(
        default_factory=list,
        description="Shared documents matching criteria",
    )
    merged_content: str | None = Field(
        default=None,
        description="Merged content string (if include_content=True)",
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens returned")
    collections_loaded: int = Field(default=0, ge=0, description="Number of collections loaded")
    context_hash: str = Field(
        default="",
        description="Hash for cache invalidation",
    )


class PromptTemplateInfo(BaseModel):
    """Information about a prompt template."""

    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    slug: str = Field(..., description="Template slug")
    description: str | None = Field(default=None, description="Template description")
    prompt: str = Field(..., description="The prompt template text")
    variables: list[str] = Field(default_factory=list, description="Variables in the template")
    category: str = Field(..., description="Template category")
    collection_name: str = Field(..., description="Source collection name")


class ListTemplatesParams(BaseModel):
    """Parameters for rlm_list_templates tool."""

    category: str | None = Field(default=None, description="Filter by category")


class ListTemplatesResult(BaseModel):
    """Result of rlm_list_templates tool."""

    templates: list[PromptTemplateInfo] = Field(
        default_factory=list, description="Available prompt templates"
    )
    total_count: int = Field(default=0, ge=0, description="Total templates found")
    categories: list[str] = Field(default_factory=list, description="Available categories")


class GetTemplateParams(BaseModel):
    """Parameters for rlm_get_template tool."""

    template_id: str | None = Field(default=None, description="Template ID")
    slug: str | None = Field(default=None, description="Template slug")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Variable values to substitute in the template",
    )


class GetTemplateResult(BaseModel):
    """Result of rlm_get_template tool."""

    template: PromptTemplateInfo | None = Field(default=None, description="The template info")
    rendered_prompt: str | None = Field(
        default=None, description="Prompt with variables substituted"
    )
    missing_variables: list[str] = Field(
        default_factory=list, description="Variables that weren't provided"
    )


# ============ AGENT MEMORY MODELS (Phase 8.2) ============


class AgentMemoryType(StrEnum):
    """Type of agent memory."""

    FACT = "fact"  # Objective information
    DECISION = "decision"  # Choice made with rationale
    LEARNING = "learning"  # Pattern or insight discovered
    PREFERENCE = "preference"  # User/team preference
    TODO = "todo"  # Deferred task or reminder
    CONTEXT = "context"  # General session context


class AgentMemoryScope(StrEnum):
    """Scope of agent memory visibility."""

    AGENT = "agent"  # Specific to one agent/session
    PROJECT = "project"  # Shared across project
    TEAM = "team"  # Shared across team
    USER = "user"  # Personal across all projects


class RememberParams(BaseModel):
    """Parameters for rlm_remember tool."""

    content: str = Field(..., min_length=1, description="The memory content to store")
    type: AgentMemoryType = Field(default=AgentMemoryType.FACT, description="Type of memory")
    scope: AgentMemoryScope = Field(
        default=AgentMemoryScope.PROJECT, description="Visibility scope"
    )
    category: str | None = Field(default=None, description="Optional grouping category")
    ttl_days: int | None = Field(
        default=None, ge=1, le=365, description="Days until memory expires (null = permanent)"
    )
    related_to: list[str] = Field(default_factory=list, description="IDs of related memories")
    document_refs: list[str] = Field(default_factory=list, description="Referenced document paths")
    source: str | None = Field(
        default=None, description="What created this memory (e.g., 'user', 'agent', 'import')"
    )


class RememberResult(BaseModel):
    """Result of rlm_remember tool."""

    memory_id: str = Field(..., description="Unique identifier for the stored memory")
    content: str = Field(..., description="The stored content")
    type: AgentMemoryType = Field(..., description="Memory type")
    scope: AgentMemoryScope = Field(..., description="Memory scope")
    category: str | None = Field(default=None, description="Category if provided")
    expires_at: datetime | None = Field(default=None, description="When memory expires")
    created: bool = Field(default=True, description="True if new, False if updated")
    message: str = Field(..., description="Human-readable status message")


class RecallParams(BaseModel):
    """Parameters for rlm_recall tool - semantic memory retrieval."""

    query: str = Field(..., min_length=1, description="Search query for semantic recall")
    type: AgentMemoryType | None = Field(default=None, description="Filter by memory type")
    scope: AgentMemoryScope | None = Field(default=None, description="Filter by scope")
    category: str | None = Field(default=None, description="Filter by category")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum memories to return")
    min_relevance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum relevance score (0-1)"
    )
    include_expired: bool = Field(default=False, description="Include expired memories in recall")


class RecalledMemory(BaseModel):
    """A memory recalled with relevance scoring."""

    memory_id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    type: AgentMemoryType = Field(..., description="Memory type")
    scope: AgentMemoryScope = Field(..., description="Memory scope")
    category: str | None = Field(default=None, description="Category")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence after decay")
    created_at: datetime = Field(..., description="When memory was created")
    last_accessed_at: datetime | None = Field(default=None, description="Last access time")
    access_count: int = Field(default=0, ge=0, description="Times accessed")


class RecallResult(BaseModel):
    """Result of rlm_recall tool."""

    memories: list[RecalledMemory] = Field(
        default_factory=list, description="Recalled memories ranked by relevance"
    )
    total_searched: int = Field(default=0, ge=0, description="Total memories searched")
    query: str = Field(..., description="Original query")
    timing_ms: int = Field(default=0, ge=0, description="Recall latency in milliseconds")


class MemoriesParams(BaseModel):
    """Parameters for rlm_memories tool - list memories with filters."""

    type: AgentMemoryType | None = Field(default=None, description="Filter by type")
    scope: AgentMemoryScope | None = Field(default=None, description="Filter by scope")
    category: str | None = Field(default=None, description="Filter by category")
    search: str | None = Field(default=None, description="Text search in content")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum memories to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    include_expired: bool = Field(default=False, description="Include expired memories")


class MemoryInfo(BaseModel):
    """Information about a stored memory."""

    memory_id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    type: AgentMemoryType = Field(..., description="Memory type")
    scope: AgentMemoryScope = Field(..., description="Memory scope")
    category: str | None = Field(default=None, description="Category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Current confidence")
    source: str | None = Field(default=None, description="Memory source")
    created_at: datetime = Field(..., description="Creation time")
    expires_at: datetime | None = Field(default=None, description="Expiration time")
    access_count: int = Field(default=0, ge=0, description="Access count")


class MemoriesResult(BaseModel):
    """Result of rlm_memories tool."""

    memories: list[MemoryInfo] = Field(default_factory=list, description="List of memories")
    total_count: int = Field(default=0, ge=0, description="Total matching memories")
    has_more: bool = Field(default=False, description="More results available")


class ForgetParams(BaseModel):
    """Parameters for rlm_forget tool."""

    memory_id: str | None = Field(default=None, description="Specific memory ID to delete")
    type: AgentMemoryType | None = Field(
        default=None, description="Delete all memories of this type"
    )
    category: str | None = Field(default=None, description="Delete all memories in this category")
    older_than_days: int | None = Field(
        default=None, ge=1, description="Delete memories older than N days"
    )


class ForgetResult(BaseModel):
    """Result of rlm_forget tool."""

    deleted_count: int = Field(default=0, ge=0, description="Number of memories deleted")
    message: str = Field(..., description="Human-readable status message")


# ============ MULTI-AGENT SWARM MODELS (Phase 9.1) ============


class SwarmCreateParams(BaseModel):
    """Parameters for rlm_swarm_create tool."""

    name: str = Field(..., min_length=1, max_length=100, description="Swarm name")
    description: str | None = Field(default=None, description="Swarm description")
    max_agents: int = Field(default=10, ge=2, le=50, description="Maximum agents allowed")
    task_timeout: int = Field(default=300, ge=60, le=3600, description="Task timeout in seconds")
    claim_timeout: int = Field(
        default=600, ge=60, le=7200, description="Resource claim timeout in seconds"
    )


class SwarmCreateResult(BaseModel):
    """Result of rlm_swarm_create tool."""

    swarm_id: str = Field(..., description="Unique swarm identifier")
    name: str = Field(..., description="Swarm name")
    max_agents: int = Field(..., description="Maximum agents")
    task_timeout: int = Field(..., description="Task timeout seconds")
    claim_timeout: int = Field(..., description="Claim timeout seconds")
    created_at: datetime = Field(..., description="Creation time")
    message: str = Field(..., description="Human-readable status message")


class SwarmJoinParams(BaseModel):
    """Parameters for rlm_swarm_join tool."""

    swarm_id: str = Field(..., description="ID of swarm to join")
    agent_id: str = Field(..., description="Unique identifier for this agent")
    name: str | None = Field(default=None, description="Human-readable agent name")


class SwarmJoinResult(BaseModel):
    """Result of rlm_swarm_join tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID")
    swarm_name: str = Field(..., description="Swarm name")
    current_agents: int = Field(..., ge=0, description="Current number of agents")
    max_agents: int = Field(..., description="Maximum agents allowed")
    message: str = Field(..., description="Human-readable status message")


class ClaimParams(BaseModel):
    """Parameters for rlm_claim tool - claim exclusive resource access."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID making the claim")
    resource_type: str = Field(
        ..., description="Resource type: 'file', 'function', 'module', 'custom'"
    )
    resource_id: str = Field(..., description="Resource identifier (e.g., 'src/auth.ts')")
    ttl_seconds: int | None = Field(
        default=None, ge=60, le=7200, description="Custom TTL (uses swarm default if null)"
    )


class ClaimResult(BaseModel):
    """Result of rlm_claim tool."""

    claim_id: str = Field(..., description="Unique claim identifier")
    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")
    acquired: bool = Field(..., description="Whether claim was acquired")
    expires_at: datetime = Field(..., description="When claim expires")
    held_by: str | None = Field(
        default=None, description="If not acquired, agent ID holding the resource"
    )
    message: str = Field(..., description="Human-readable status message")


class ReleaseParams(BaseModel):
    """Parameters for rlm_release tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID releasing the claim")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")


class ReleaseResult(BaseModel):
    """Result of rlm_release tool."""

    released: bool = Field(..., description="Whether resource was released")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")
    message: str = Field(..., description="Human-readable status message")


class StateGetParams(BaseModel):
    """Parameters for rlm_state_get tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    key: str = Field(..., description="State key to retrieve")


class StateGetResult(BaseModel):
    """Result of rlm_state_get tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    key: str = Field(..., description="State key")
    value: Any | None = Field(default=None, description="State value (JSON)")
    version: int = Field(default=0, ge=0, description="State version for concurrency")
    exists: bool = Field(..., description="Whether key exists")
    updated_by: str | None = Field(default=None, description="Agent that last updated")
    updated_at: datetime | None = Field(default=None, description="Last update time")


class StateSetParams(BaseModel):
    """Parameters for rlm_state_set tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID setting the state")
    key: str = Field(..., description="State key")
    value: Any = Field(..., description="State value (JSON-serializable)")
    expected_version: int | None = Field(
        default=None, ge=0, description="Expected version for optimistic locking (null = overwrite)"
    )


class StateSetResult(BaseModel):
    """Result of rlm_state_set tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    key: str = Field(..., description="State key")
    version: int = Field(..., ge=1, description="New state version")
    success: bool = Field(..., description="Whether update succeeded")
    conflict: bool = Field(
        default=False, description="True if version mismatch (optimistic lock failed)"
    )
    message: str = Field(..., description="Human-readable status message")


class BroadcastParams(BaseModel):
    """Parameters for rlm_broadcast tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID broadcasting")
    event_type: str = Field(..., description="Event type (e.g., 'file_changed', 'task_done')")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload")


class BroadcastResult(BaseModel):
    """Result of rlm_broadcast tool."""

    event_id: str = Field(..., description="Unique event identifier")
    swarm_id: str = Field(..., description="Swarm ID")
    event_type: str = Field(..., description="Event type")
    delivered: bool = Field(..., description="Whether event was published")
    message: str = Field(..., description="Human-readable status message")


class TaskCreateParams(BaseModel):
    """Parameters for rlm_task_create tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    title: str = Field(..., min_length=1, description="Task title")
    description: str | None = Field(default=None, description="Task description")
    priority: int = Field(default=0, ge=0, le=100, description="Priority (higher = more urgent)")
    depends_on: list[str] = Field(
        default_factory=list, description="Task IDs that must complete first"
    )


class TaskCreateResult(BaseModel):
    """Result of rlm_task_create tool."""

    task_id: str = Field(..., description="Unique task identifier")
    swarm_id: str = Field(..., description="Swarm ID")
    title: str = Field(..., description="Task title")
    priority: int = Field(..., description="Task priority")
    status: str = Field(default="pending", description="Task status")
    depends_on: list[str] = Field(default_factory=list, description="Dependencies")
    message: str = Field(..., description="Human-readable status message")


class TaskClaimParams(BaseModel):
    """Parameters for rlm_task_claim tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID claiming")
    task_id: str | None = Field(
        default=None, description="Specific task ID (null = get next available)"
    )


class TaskClaimResult(BaseModel):
    """Result of rlm_task_claim tool."""

    task_id: str | None = Field(
        default=None, description="Claimed task ID (null if none available)"
    )
    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID")
    title: str | None = Field(default=None, description="Task title")
    description: str | None = Field(default=None, description="Task description")
    priority: int = Field(default=0, description="Task priority")
    claimed: bool = Field(..., description="Whether a task was claimed")
    message: str = Field(..., description="Human-readable status message")


class TaskCompleteParams(BaseModel):
    """Parameters for rlm_task_complete tool."""

    swarm_id: str = Field(..., description="Swarm ID")
    agent_id: str = Field(..., description="Agent ID completing")
    task_id: str = Field(..., description="Task ID to complete")
    result: dict[str, Any] | None = Field(default=None, description="Task result data")
    error: str | None = Field(default=None, description="Error message if task failed")


class TaskCompleteResult(BaseModel):
    """Result of rlm_task_complete tool."""

    task_id: str = Field(..., description="Task ID")
    swarm_id: str = Field(..., description="Swarm ID")
    status: str = Field(..., description="Final task status ('completed' or 'failed')")
    completed: bool = Field(..., description="Whether task was completed successfully")
    unblocked_tasks: list[str] = Field(
        default_factory=list, description="Task IDs now unblocked by this completion"
    )
    message: str = Field(..., description="Human-readable status message")


# ============ PHASE 10: DOCUMENT SYNC MODELS ============


class UploadDocumentResult(BaseModel):
    """Result of rlm_upload_document tool."""

    path: str = Field(..., description="Document path")
    action: str = Field(..., description="Action taken: 'created' or 'updated'")
    size: int = Field(..., ge=0, description="Document size in bytes")
    hash: str = Field(..., description="Content hash")
    message: str = Field(..., description="Human-readable status message")


class SyncDocumentItem(BaseModel):
    """A document to sync."""

    path: str = Field(..., description="Document path")
    content: str = Field(..., description="Document content")


class SyncDocumentsParams(BaseModel):
    """Parameters for rlm_sync_documents tool."""

    documents: list[SyncDocumentItem] = Field(
        ..., description="Documents to sync", min_length=1, max_length=100
    )
    delete_missing: bool = Field(default=False, description="Delete documents not in list")


class SyncDocumentsResult(BaseModel):
    """Result of rlm_sync_documents tool."""

    created: int = Field(default=0, ge=0, description="Documents created")
    updated: int = Field(default=0, ge=0, description="Documents updated")
    unchanged: int = Field(default=0, ge=0, description="Documents unchanged")
    deleted: int = Field(default=0, ge=0, description="Documents deleted")
    total: int = Field(default=0, ge=0, description="Total documents processed")
    message: str = Field(..., description="Human-readable status message")


class SettingsResult(BaseModel):
    """Result of rlm_settings tool."""

    project_id: str = Field(..., description="Project ID")
    max_tokens_per_query: int = Field(..., description="Max tokens per query")
    search_mode: str = Field(..., description="Default search mode")
    include_summaries: bool = Field(..., description="Include summaries in queries")
    auto_inject_context: bool = Field(..., description="Auto-inject context")
    message: str = Field(..., description="Human-readable status message")


# ============ ACCESS REQUEST MODELS ============


class RequestAccessParams(BaseModel):
    """Parameters for rlm_request_access tool."""

    requested_level: str = Field(
        default="VIEWER",
        description="Requested access level: VIEWER, EDITOR, or ADMIN",
    )
    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Optional reason for requesting access",
    )


class RequestAccessResult(BaseModel):
    """Result of rlm_request_access tool."""

    request_id: str = Field(..., description="Access request ID")
    project_id: str = Field(..., description="Project ID")
    project_name: str = Field(..., description="Project name")
    requested_level: str = Field(..., description="Requested access level")
    status: str = Field(default="pending", description="Request status")
    message: str = Field(..., description="Human-readable status message")
    dashboard_url: str = Field(..., description="URL to view request status")


# ============ INDEX JOB MODELS (Async Indexing) ============


class IndexJobStatus(StrEnum):
    """Status of an index job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IndexJobCreateResponse(BaseModel):
    """Response from creating an index job."""

    job_id: str = Field(..., description="Unique job identifier")
    project_id: str = Field(..., description="Project being indexed")
    status: IndexJobStatus = Field(..., description="Current job status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage (0-100)")
    created_at: datetime | None = Field(default=None, description="Job creation time")
    status_url: str = Field(..., description="URL to poll for job status")
    already_exists: bool = Field(
        default=False, description="Whether a job already existed for this project"
    )


class IndexJobStatusResponse(BaseModel):
    """Response from checking index job status."""

    id: str = Field(..., alias="job_id", description="Job ID")
    project_id: str = Field(..., description="Project being indexed")
    status: IndexJobStatus = Field(..., description="Current job status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage (0-100)")
    error_message: str | None = Field(default=None, description="Error message if failed")
    documents_total: int = Field(default=0, ge=0, description="Total documents to index")
    documents_processed: int = Field(default=0, ge=0, description="Documents processed so far")
    chunks_created: int = Field(default=0, ge=0, description="Total chunks created")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries allowed")
    worker_id: str | None = Field(default=None, description="Worker processing this job")
    created_at: datetime | None = Field(default=None, description="Job creation time")
    started_at: datetime | None = Field(default=None, description="When processing started")
    completed_at: datetime | None = Field(default=None, description="When job completed")
    updated_at: datetime | None = Field(default=None, description="Last update time")
    triggered_by: str | None = Field(default=None, description="User who triggered the job")
    triggered_via: str | None = Field(
        default=None, description="How the job was triggered (api_key, internal)"
    )
    results: dict[str, int] | None = Field(
        default=None, description="Results mapping document paths to chunk counts"
    )
