"""General response models for RLM MCP Server."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class ToolResult:
    """Result from executing an RLM tool.

    This is the standard return type for all tool handlers.
    """

    data: Any
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProjectSettings:
    """Project automation settings from dashboard."""

    # Core query settings
    max_tokens_per_query: int = 4000
    search_mode: str = "hybrid"
    include_summaries: bool = True

    # Automation settings
    automation_client: str = "claude-code"
    auto_inject_context: bool = False
    track_accessed_files: bool = False
    preserve_on_compaction: bool = True
    restore_on_session_start: bool = True
    enrich_prompts: bool = False
    system_instructions: str | None = None

    # Memory injection settings (Agents feature)
    memory_injection_enabled: bool = False
    memory_inject_types: list[str] | None = None
    memory_exclude_session_checkpoints: bool = False
    memory_min_confidence: float = 0.2
    memory_recall_query: str | None = None
    memory_save_on_commit: bool = False


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
