"""Pydantic models for RLM MCP Server request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============ ENUMS ============


class ToolName(str, Enum):
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
    RLM_PLAN = "rlm_plan"
    # Phase 4.6: Summary Storage Tools
    RLM_STORE_SUMMARY = "rlm_store_summary"
    RLM_GET_SUMMARIES = "rlm_get_summaries"
    RLM_DELETE_SUMMARY = "rlm_delete_summary"
    # Phase 5: Document Upload Tools
    RLM_UPLOAD_DOCUMENT = "rlm_upload_document"
    RLM_SYNC_DOCUMENTS = "rlm_sync_documents"
    RLM_SETTINGS = "rlm_settings"


class SearchMode(str, Enum):
    """Search mode for context queries."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"  # Future: embedding-based
    HYBRID = "hybrid"  # Future: keyword + semantic


class Plan(str, Enum):
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
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0-1)"
    )
    token_count: int = Field(..., ge=0, description="Token count for this section")
    truncated: bool = Field(
        default=False, description="Whether content was truncated to fit budget"
    )


class ContextQueryResult(BaseModel):
    """Result of rlm_context_query tool - optimized context for the client's LLM."""

    sections: list[ContextSection] = Field(
        default_factory=list, description="Ranked list of relevant sections"
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens returned")
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


# ============ RECURSIVE CONTEXT MODELS (Phase 4.5) ============


class DecomposeStrategy(str, Enum):
    """Strategy for query decomposition."""

    AUTO = "auto"  # Let the engine decide
    TERM_BASED = "term_based"  # Extract key terms
    STRUCTURAL = "structural"  # Follow document structure


class PlanStrategy(str, Enum):
    """Strategy for execution planning."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RELEVANCE_FIRST = "relevance_first"


class DecomposeParams(BaseModel):
    """Parameters for rlm_decompose tool."""

    query: str = Field(..., description="The complex question to decompose")
    max_depth: int = Field(
        default=2, ge=1, le=5, description="Maximum recursion depth"
    )
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
    priority: int = Field(
        default=1, ge=1, le=10, description="Priority (1=highest)"
    )
    estimated_tokens: int = Field(
        default=1000, ge=0, description="Estimated tokens for this query"
    )
    key_terms: list[str] = Field(
        default_factory=list, description="Key terms identified"
    )


class DecomposeResult(BaseModel):
    """Result of rlm_decompose tool."""

    original_query: str = Field(..., description="The original query")
    sub_queries: list[SubQuery] = Field(
        default_factory=list, description="Generated sub-queries"
    )
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
    strategy_used: DecomposeStrategy = Field(
        ..., description="Strategy that was used"
    )


class MultiQueryItem(BaseModel):
    """A single query in a multi-query batch."""

    query: str = Field(..., description="The query text")
    max_tokens: int | None = Field(
        default=None, description="Optional per-query token budget"
    )


class MultiQueryParams(BaseModel):
    """Parameters for rlm_multi_query tool."""

    queries: list[MultiQueryItem] = Field(
        ..., min_length=1, max_length=10, description="List of queries to execute"
    )
    max_tokens: int = Field(
        default=8000, ge=500, le=50000, description="Total token budget"
    )
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID, description="Search mode for all queries"
    )


class MultiQueryResultItem(BaseModel):
    """Result for a single query in a multi-query batch."""

    query: str = Field(..., description="The original query")
    sections: list[ContextSection] = Field(
        default_factory=list, description="Relevant sections"
    )
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
    queries_skipped: int = Field(
        default=0, ge=0, description="Queries skipped due to budget"
    )
    search_mode: SearchMode = Field(..., description="Search mode used")


class PlanStep(BaseModel):
    """A step in an execution plan."""

    step: int = Field(..., ge=1, description="Step number")
    action: str = Field(
        ..., description="Action to perform: decompose, context_query, multi_query"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Steps this step depends on"
    )
    expected_output: str = Field(
        default="sections", description="Expected output type"
    )


class PlanParams(BaseModel):
    """Parameters for rlm_plan tool."""

    query: str = Field(..., description="The complex question to plan for")
    strategy: PlanStrategy = Field(
        default=PlanStrategy.RELEVANCE_FIRST, description="Execution strategy"
    )
    max_tokens: int = Field(
        default=16000, ge=1000, le=100000, description="Total token budget"
    )


class PlanResult(BaseModel):
    """Result of rlm_plan tool."""

    plan_id: str = Field(..., description="Unique plan identifier")
    query: str = Field(..., description="The original query")
    steps: list[PlanStep] = Field(default_factory=list, description="Execution steps")
    estimated_total_tokens: int = Field(
        default=0, ge=0, description="Estimated total tokens"
    )
    strategy: PlanStrategy = Field(..., description="Strategy used")
    estimated_queries: int = Field(
        default=0, ge=0, description="Estimated number of queries"
    )


# ============ SUMMARY STORAGE MODELS (Phase 4.6) ============


class SummaryType(str, Enum):
    """Type of summary stored."""

    CONCISE = "concise"  # Brief 1-2 sentence summary
    DETAILED = "detailed"  # Full multi-paragraph summary
    TECHNICAL = "technical"  # Technical details focus
    KEYWORDS = "keywords"  # Key terms and concepts
    CUSTOM = "custom"  # User-defined summary type


class StoreSummaryParams(BaseModel):
    """Parameters for rlm_store_summary tool."""

    document_path: str = Field(
        ..., description="Path to the document (relative to project root)"
    )
    summary: str = Field(..., min_length=1, description="The summary text to store")
    summary_type: SummaryType = Field(
        default=SummaryType.CONCISE, description="Type of summary"
    )
    section_id: str | None = Field(
        default=None, description="Optional section identifier for partial summaries"
    )
    line_start: int | None = Field(
        default=None, ge=1, description="Start line for section summary"
    )
    line_end: int | None = Field(
        default=None, ge=1, description="End line for section summary"
    )
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
    created: bool = Field(
        default=True, description="True if new, False if updated existing"
    )
    message: str = Field(..., description="Human-readable status message")


class GetSummariesParams(BaseModel):
    """Parameters for rlm_get_summaries tool."""

    document_path: str | None = Field(
        default=None, description="Filter by document path"
    )
    summary_type: SummaryType | None = Field(
        default=None, description="Filter by summary type"
    )
    section_id: str | None = Field(default=None, description="Filter by section ID")
    include_content: bool = Field(
        default=True, description="Include summary content in response"
    )


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
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens across all summaries"
    )


class DeleteSummaryParams(BaseModel):
    """Parameters for rlm_delete_summary tool."""

    summary_id: str | None = Field(default=None, description="Specific summary ID")
    document_path: str | None = Field(
        default=None, description="Delete all summaries for document"
    )
    summary_type: SummaryType | None = Field(
        default=None, description="Delete summaries of this type"
    )


class DeleteSummaryResult(BaseModel):
    """Result of rlm_delete_summary tool."""

    deleted_count: int = Field(default=0, ge=0, description="Number of summaries deleted")
    message: str = Field(..., description="Human-readable status message")


# ============ DOCUMENT UPLOAD MODELS (Phase 5) ============


class UploadDocumentParams(BaseModel):
    """Parameters for rlm_upload_document tool."""

    path: str = Field(
        ...,
        description="Document path relative to project root (e.g., 'docs/api.md')",
    )
    content: str = Field(..., min_length=1, description="Document content (markdown)")


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
        ..., min_length=1, max_length=100, description="Documents to sync"
    )
    delete_missing: bool = Field(
        default=False,
        description="Delete documents not in the sync list",
    )


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
