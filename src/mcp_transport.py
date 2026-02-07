"""
MCP Streamable HTTP Transport for Snipara.

This module implements the MCP (Model Context Protocol) Streamable HTTP transport
specification, enabling direct connections from MCP-compatible AI clients.

Supported Clients:
    - Claude Code (Anthropic)
    - Cursor IDE
    - ChatGPT (with MCP support)
    - Windsurf
    - Any MCP-compatible client

Protocol:
    Uses JSON-RPC 2.0 over HTTP with the following methods:
    - initialize: Establish connection and exchange capabilities
    - tools/list: List available tools
    - tools/call: Execute a tool with arguments
    - ping: Keep-alive check

Endpoints:
    POST /mcp/{project_id}  - Main JSON-RPC endpoint for tool execution
    GET  /mcp/{project_id}  - SSE endpoint for server-initiated messages

Authentication:
    Accepts either:
    - X-API-Key header: Project API key (rlm_...) or Team API key
    - Authorization: Bearer header: API key or OAuth token (snipara_at_...)

Example Configuration (Claude Code .mcp.json):
    {
        "mcpServers": {
            "snipara": {
                "type": "http",
                "url": "https://api.snipara.com/mcp/{project_slug}",
                "headers": {"X-API-Key": "rlm_..."}
            }
        }
    }

Note:
    Team-scoped queries (/mcp/team/{team_id}) are handled in server.py
    to avoid circular imports with execute_multi_project_query.
"""

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import get_effective_plan, get_project_with_team, validate_api_key, validate_oauth_token
from .config import settings
from .models import Plan, ToolName
from .rlm_engine import RLMEngine
from .usage import (
    check_rate_limit,
    check_usage_limits,
    is_scan_blocked,
    log_security_event,
    track_usage,
)

# ============ HELPERS ============


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP from X-Forwarded-For header or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


# ============ ROUTER CONFIGURATION ============

router = APIRouter(prefix="/mcp", tags=["MCP Transport"])

#: MCP protocol version (spec: 2024-11-05)
MCP_VERSION = "2024-11-05"


# ============ TOOL DEFINITIONS ============
#
# These definitions are returned by the tools/list method and define
# the schema for each tool's input parameters.
#
# Tool Categories:
#   - Context Retrieval: rlm_context_query, rlm_ask, rlm_search, rlm_read
#   - Query Optimization: rlm_decompose, rlm_multi_query, rlm_plan
#   - Team Queries: rlm_multi_project_query (requires team API key)
#   - Session Management: rlm_inject, rlm_context, rlm_clear_context
#   - Metadata: rlm_stats, rlm_sections, rlm_settings
#   - Summaries: rlm_store_summary, rlm_get_summaries, rlm_delete_summary
#   - Shared Context: rlm_shared_context, rlm_list_templates, rlm_get_template
#   - Agent Memory: rlm_remember, rlm_recall, rlm_memories, rlm_forget
#   - Multi-Agent Swarm: rlm_swarm_*, rlm_claim, rlm_release, rlm_state_*, rlm_task_*
#   - Document Sync: rlm_upload_document, rlm_sync_documents

TOOL_DEFINITIONS = [
    {
        "name": "rlm_context_query",
        "description": "Query optimized context from documentation. Returns ranked sections within token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or topic"},
                "max_tokens": {
                    "type": "integer",
                    "default": 4000,
                    "minimum": 100,
                    "maximum": 100000,
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid"],
                    "default": "hybrid",
                },
                "include_metadata": {"type": "boolean", "default": True},
                "prefer_summaries": {"type": "boolean", "default": False},
                "return_references": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return chunk references (IDs + previews) instead of full content. Use rlm_get_chunk to retrieve full content by ID. Reduces hallucination by maintaining clear source attribution.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_ask",
        "description": "Query documentation with a question (basic). Use rlm_context_query for better results.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "rlm_search",
        "description": "Search documentation for a regex pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "rlm_decompose",
        "description": "Break complex query into sub-queries with execution order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_multi_query",
        "description": "Execute multiple queries in one call with shared token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_tokens": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "max_tokens": {"type": "integer", "default": 8000},
            },
            "required": ["queries"],
        },
    },
    {
        "name": "rlm_plan",
        "description": "Generate full execution plan for complex questions. Returns steps for decomposition, context queries, and synthesis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The complex question to plan for"},
                "strategy": {
                    "type": "string",
                    "enum": ["breadth_first", "depth_first", "relevance_first"],
                    "default": "relevance_first",
                    "description": "Execution strategy",
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 16000,
                    "minimum": 1000,
                    "maximum": 100000,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_multi_project_query",
        "description": "Query across all projects in a team. Requires team API key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or topic"},
                "max_tokens": {
                    "type": "integer",
                    "default": 4000,
                    "minimum": 100,
                    "maximum": 100000,
                },
                "per_project_limit": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20},
                "project_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional project IDs/slugs to include",
                },
                "exclude_project_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional project IDs/slugs to exclude",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid"],
                    "default": "keyword",
                },
                "include_metadata": {"type": "boolean", "default": True},
                "prefer_summaries": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_inject",
        "description": "Set session context for subsequent queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["context"],
        },
    },
    {
        "name": "rlm_context",
        "description": "Show current session context.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_clear_context",
        "description": "Clear session context.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_stats",
        "description": "Show documentation statistics.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_sections",
        "description": "List indexed document sections with optional pagination and filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum sections to return (default: 50, max: 500)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of sections to skip for pagination (default: 0)",
                },
                "filter": {
                    "type": "string",
                    "description": "Filter sections by title prefix (case-insensitive)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_read",
        "description": "Read specific lines from documentation.",
        "inputSchema": {
            "type": "object",
            "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["start_line", "end_line"],
        },
    },
    {
        "name": "rlm_store_summary",
        "description": "Store an LLM-generated summary for a document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_path": {"type": "string"},
                "summary": {"type": "string"},
                "summary_type": {
                    "type": "string",
                    "enum": ["concise", "detailed", "technical", "keywords", "custom"],
                    "default": "concise",
                },
                "generated_by": {"type": "string"},
            },
            "required": ["document_path", "summary"],
        },
    },
    {
        "name": "rlm_get_summaries",
        "description": "Retrieve stored summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_path": {"type": "string"},
                "summary_type": {
                    "type": "string",
                    "enum": ["concise", "detailed", "technical", "keywords", "custom"],
                },
                "include_content": {"type": "boolean", "default": True},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_delete_summary",
        "description": "Delete stored summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {"summary_id": {"type": "string"}, "document_path": {"type": "string"}},
            "required": [],
        },
    },
    # Phase 7: Shared Context Tools
    {
        "name": "rlm_shared_context",
        "description": "Get merged context from linked shared collections. Returns categorized docs with budget allocation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "default": 4000,
                    "minimum": 100,
                    "maximum": 100000,
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["MANDATORY", "BEST_PRACTICES", "GUIDELINES", "REFERENCE"],
                    },
                    "description": "Filter by categories (default: all)",
                },
                "include_content": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include merged content",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_list_templates",
        "description": "List available prompt templates from shared collections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_get_template",
        "description": "Get a specific prompt template by ID or slug. Optionally render with variables.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "description": "Template ID"},
                "slug": {"type": "string", "description": "Template slug"},
                "variables": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Variables to substitute in template",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_list_collections",
        "description": "List all shared context collections accessible to you. Returns collections you own, team collections you're a member of, and public collections. Use this to find collection IDs for uploading documents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_public": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include public collections in the results",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_upload_shared_document",
        "description": "Upload or update a document in a shared context collection. Use for team best practices, coding standards, and guidelines. Requires Team plan or higher.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection_id": {"type": "string", "description": "The shared collection ID"},
                "title": {"type": "string", "description": "Document title"},
                "content": {"type": "string", "description": "Document content (markdown)"},
                "category": {
                    "type": "string",
                    "enum": ["MANDATORY", "BEST_PRACTICES", "GUIDELINES", "REFERENCE"],
                    "default": "BEST_PRACTICES",
                    "description": "Document category for token budget allocation",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for filtering and organization",
                },
                "priority": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Priority within category (higher = more important)",
                },
            },
            "required": ["collection_id", "title", "content"],
        },
    },
    # Phase 8.2: Agent Memory Tools
    {
        "name": "rlm_remember",
        "description": "Store a memory for later semantic recall. Supports types: fact, decision, learning, preference, todo, context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                    "default": "fact",
                },
                "scope": {
                    "type": "string",
                    "enum": ["agent", "project", "team", "user"],
                    "default": "project",
                },
                "category": {"type": "string", "description": "Optional category for grouping"},
                "ttl_days": {
                    "type": "integer",
                    "description": "Days until expiration (null = permanent)",
                },
                "related_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of related memories",
                },
                "document_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Referenced document paths",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "rlm_recall",
        "description": "Semantically recall relevant memories based on a query. Uses embeddings weighted by confidence decay.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                },
                "scope": {"type": "string", "enum": ["agent", "project", "team", "user"]},
                "category": {"type": "string", "description": "Filter by category"},
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum memories to return",
                },
                "min_relevance": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum relevance score (0-1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_memories",
        "description": "List memories with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                },
                "scope": {"type": "string", "enum": ["agent", "project", "team", "user"]},
                "category": {"type": "string"},
                "search": {"type": "string", "description": "Text search in content"},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_forget",
        "description": "Delete memories by ID or filter criteria.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Specific memory ID to delete"},
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                },
                "category": {"type": "string", "description": "Delete all in this category"},
                "older_than_days": {
                    "type": "integer",
                    "description": "Delete memories older than N days",
                },
            },
            "required": [],
        },
    },
    # Phase 9.1: Multi-Agent Swarm Tools
    {
        "name": "rlm_swarm_create",
        "description": "Create a new agent swarm for multi-agent coordination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Swarm name"},
                "description": {"type": "string"},
                "max_agents": {"type": "integer", "default": 10},
                "config": {"type": "object"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "rlm_swarm_join",
        "description": "Join an existing swarm as an agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string", "description": "Swarm to join"},
                "agent_id": {"type": "string", "description": "Your unique agent identifier"},
                "role": {
                    "type": "string",
                    "enum": ["coordinator", "worker", "observer"],
                    "default": "worker",
                },
                "capabilities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_claim",
        "description": "Claim exclusive access to a resource (file, function, module). Claims auto-expire.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "resource_type": {
                    "type": "string",
                    "enum": ["file", "function", "module", "component", "other"],
                },
                "resource_id": {
                    "type": "string",
                    "description": "Resource identifier (e.g., file path)",
                },
                "timeout_seconds": {"type": "integer", "default": 300},
            },
            "required": ["swarm_id", "agent_id", "resource_type", "resource_id"],
        },
    },
    {
        "name": "rlm_release",
        "description": "Release a claimed resource.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "claim_id": {"type": "string"},
                "resource_type": {"type": "string"},
                "resource_id": {"type": "string"},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_state_get",
        "description": "Read shared swarm state by key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "key": {"type": "string", "description": "State key to read"},
            },
            "required": ["swarm_id", "key"],
        },
    },
    {
        "name": "rlm_state_set",
        "description": "Write shared swarm state with optimistic locking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "key": {"type": "string"},
                "value": {"description": "Value to set (any JSON-serializable type)"},
                "expected_version": {
                    "type": "integer",
                    "description": "Expected version for optimistic locking",
                },
            },
            "required": ["swarm_id", "agent_id", "key", "value"],
        },
    },
    {
        "name": "rlm_broadcast",
        "description": "Send an event to all agents in the swarm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "event_type": {"type": "string", "description": "Event type"},
                "payload": {"type": "object", "description": "Event data"},
            },
            "required": ["swarm_id", "agent_id", "event_type"],
        },
    },
    {
        "name": "rlm_task_create",
        "description": "Create a task in the swarm's distributed task queue.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {
                    "type": "integer",
                    "default": 0,
                    "description": "Higher = more urgent",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task IDs this depends on",
                },
                "metadata": {"type": "object"},
            },
            "required": ["swarm_id", "agent_id", "title"],
        },
    },
    {
        "name": "rlm_task_claim",
        "description": "Claim a task from the queue. If task_id not specified, claims highest priority available task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "task_id": {"type": "string", "description": "Specific task to claim (optional)"},
                "timeout_seconds": {"type": "integer", "default": 600},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_task_complete",
        "description": "Mark a claimed task as completed or failed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "task_id": {"type": "string"},
                "success": {"type": "boolean", "default": True},
                "result": {"description": "Task result data"},
            },
            "required": ["swarm_id", "agent_id", "task_id"],
        },
    },
    # Phase 10: Document Sync Tools
    {
        "name": "rlm_settings",
        "description": "Get current project settings from dashboard (max_tokens, search_mode, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "refresh": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force refresh from API",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_upload_document",
        "description": "Upload or update a document in the project. Supports .md, .txt, .mdx files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document path (e.g., 'docs/api.md')"},
                "content": {"type": "string", "description": "Document content (markdown)"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "rlm_sync_documents",
        "description": "Bulk sync multiple documents. Use for batch uploads or CI/CD integration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                    "description": "Documents to sync",
                },
                "delete_missing": {
                    "type": "boolean",
                    "default": False,
                    "description": "Delete docs not in list",
                },
            },
            "required": ["documents"],
        },
    },
    # Phase 12: RLM Orchestration Tools
    {
        "name": "rlm_load_document",
        "description": "Load raw document content by file path. Returns the full unprocessed content of a single document for RLM-style exploration where the model navigates raw content directly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document path (e.g., 'docs/api.md')"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "rlm_load_project",
        "description": "Load structured map of all project documents with content. Returns a token-budgeted dump of every file, with optional path filtering. Use for full-project exploration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "default": 16000,
                    "description": "Total token budget for returned content",
                },
                "paths_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include files matching these path prefixes (e.g., ['docs/', 'src/'])",
                },
                "include_content": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include file content (false = metadata only)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "rlm_orchestrate",
        "description": "Multi-round context exploration in a single call. Performs: (1) section scan for project structure, (2) ranked search for top relevant sections, (3) raw file load for highest-scoring documents. Combines search intelligence with raw access.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or topic to explore"},
                "max_tokens": {
                    "type": "integer",
                    "default": 16000,
                    "description": "Token budget for raw file content",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top sections to use for file selection",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_repl_context",
        "description": "Package project context for REPL consumption. Returns a structured dict of files and sections plus Python helper code (peek, grep, sections, files) ready for injection into an rlm-runtime REPL session via set_repl_context + execute_python. Optionally filter by a relevance query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional query to filter context by relevance. If empty, loads files in order within budget.",
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 8000,
                    "description": "Token budget for file content",
                },
                "include_helpers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include Python helper code (peek, grep, sections, files) in setup_code",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic", "hybrid"],
                    "default": "hybrid",
                    "description": "Search mode when query is provided",
                },
            },
            "required": [],
        },
    },
    # Phase 14: Pass-by-Reference
    {
        "name": "rlm_get_chunk",
        "description": "Retrieve full content by chunk ID. Use with rlm_context_query(return_references=True) to fetch full content of specific sections. This pass-by-reference pattern reduces hallucination by maintaining clear source attribution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk ID from rlm_context_query results (when return_references=True)",
                },
            },
            "required": ["chunk_id"],
        },
    },
]


# ============ JSON-RPC HELPERS ============


def jsonrpc_response(id: Any, result: Any) -> dict:
    """
    Create a JSON-RPC 2.0 success response.

    Args:
        id: Request ID (must match the request)
        result: The result payload

    Returns:
        JSON-RPC 2.0 response dict
    """
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id: Any, code: int, message: str) -> dict:
    """
    Create a JSON-RPC 2.0 error response.

    Standard error codes:
        -32700: Parse error
        -32600: Invalid request
        -32601: Method not found
        -32602: Invalid params
        -32603: Internal error
        -32000 to -32099: Server errors (application-specific)

    Args:
        id: Request ID (can be None for parse errors)
        code: Error code (negative integer)
        message: Human-readable error message

    Returns:
        JSON-RPC 2.0 error response dict
    """
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ============ REQUEST VALIDATION ============


async def validate_request(
    project_id_or_slug: str, api_key: str, client_ip: str | None = None
) -> tuple[dict | None, Plan, str | None, str | None]:
    """
    Validate authentication and check usage limits.

    Supports both API keys (rlm_...) and OAuth tokens (snipara_at_...).

    Args:
        project_id_or_slug: Project ID or slug from URL
        api_key: API key or OAuth token from header

    Returns:
        Tuple of (auth_info, plan, error_message, actual_project_id)
        - auth_info: Dict with API key info if valid, None otherwise
        - plan: Subscription plan (FREE, PRO, TEAM, ENTERPRISE)
        - error_message: Error string if validation failed, None if success
        - actual_project_id: Database ID (not slug) for operations
    """
    # Anti-scan check
    key_prefix = api_key[:12]
    if await is_scan_blocked(key_prefix):
        log_security_event("scan.blocked", "api_key", key_prefix, key_prefix)
        return None, Plan.FREE, "Too many failed requests. Try again later.", None

    auth_info = None

    # Check if it's an OAuth token
    if api_key.startswith("snipara_at_"):
        auth_info = await validate_oauth_token(api_key, project_id_or_slug)
        if not auth_info:
            return (
                None,
                Plan.FREE,
                "Invalid or expired OAuth token. Re-authenticate at https://snipara.com/dashboard or run /snipara:quickstart",
                None,
            )
    else:
        # Fall back to API key validation
        auth_info = await validate_api_key(api_key, project_id_or_slug)
        if not auth_info:
            return (
                None,
                Plan.FREE,
                "Invalid API key. Get a free key at https://snipara.com/dashboard (100 queries/month, no credit card)",
                None,
            )

    project = await get_project_with_team(project_id_or_slug)
    if not project:
        return None, Plan.FREE, "Project not found", None

    # Use actual database ID for all operations
    actual_project_id = project.id

    if not await check_rate_limit(auth_info["id"], client_ip=client_ip):
        log_security_event(
            "rate_limit.exceeded",
            "api_key",
            auth_info["id"],
            auth_info.get("user_id", auth_info["id"]),
        )
        return None, Plan.FREE, f"Rate limit exceeded: {settings.rate_limit_requests}/min", None

    plan = get_effective_plan(project.team.subscription if project.team else None)
    limits = await check_usage_limits(actual_project_id, plan)
    if limits.exceeded:
        return None, plan, f"Monthly limit exceeded: {limits.current}/{limits.max}", None

    return auth_info, plan, None, actual_project_id


# ============ REQUEST HANDLERS ============


async def handle_call_tool(id: Any, params: dict, project_id: str, plan: Plan) -> dict:
    """
    Handle MCP tools/call request.

    Executes a tool through the RLMEngine and tracks usage.

    Args:
        id: JSON-RPC request ID
        params: Tool call parameters containing:
            - name: Tool name (e.g., "rlm_context_query")
            - arguments: Tool-specific arguments
        project_id: Database project ID
        plan: Subscription plan for rate limiting

    Returns:
        JSON-RPC response with tool result or error
    """
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    try:
        tool_enum = ToolName(tool_name)
    except ValueError:
        return jsonrpc_error(id, -32602, f"Unknown tool: {tool_name}")

    try:
        engine = RLMEngine(project_id, plan=plan)
        result = await engine.execute(tool_enum, arguments)

        await track_usage(
            project_id=project_id,
            tool=tool_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=0,
            success=True,
        )

        return jsonrpc_response(
            id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(result.data, indent=2, default=str)}
                ],
            },
        )
    except Exception as e:
        await track_usage(
            project_id=project_id,
            tool=tool_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            success=False,
            error=str(e),
        )
        return jsonrpc_error(id, -32000, str(e))


async def handle_request(body: dict, project_id: str, plan: Plan) -> dict | None:
    """
    Handle a single JSON-RPC request.

    Routes requests to appropriate handlers based on method.

    Supported Methods:
        - initialize: Returns server info and capabilities
        - tools/list: Returns available tool definitions
        - tools/call: Executes a tool
        - ping: Returns empty response (keep-alive)

    Args:
        body: JSON-RPC request body
        project_id: Database project ID
        plan: Subscription plan

    Returns:
        JSON-RPC response dict, or None for notifications (requests without id)
    """
    method, id, params = body.get("method"), body.get("id"), body.get("params", {})

    if id is None:  # Notification
        return None

    if method == "initialize":
        return jsonrpc_response(
            id,
            {
                "protocolVersion": MCP_VERSION,
                "serverInfo": {"name": "snipara", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        )
    elif method == "tools/list":
        return jsonrpc_response(id, {"tools": TOOL_DEFINITIONS})
    elif method == "tools/call":
        return await handle_call_tool(id, params, project_id, plan)
    elif method == "ping":
        return jsonrpc_response(id, {})
    else:
        return jsonrpc_error(id, -32601, f"Method not found: {method}")


# ============ HTTP ENDPOINTS ============


@router.post("/{project_id}")
async def mcp_endpoint(
    project_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """
    MCP Streamable HTTP endpoint.

    Accepts authentication via either X-API-Key or Authorization: Bearer header.

    Config example (Claude Code):
    ```json
    {"mcpServers": {"snipara": {"type": "http", "url": "https://api.snipara.com/mcp/{project_id}", "headers": {"X-API-Key": "rlm_..."}}}}
    ```

    Alternative (Authorization Bearer):
    ```json
    {"mcpServers": {"snipara": {"type": "http", "url": "https://api.snipara.com/mcp/{project_id}", "headers": {"Authorization": "Bearer rlm_..."}}}}
    ```
    """
    # Accept X-API-Key header (preferred) or Authorization: Bearer
    if x_api_key:
        api_key = x_api_key
    elif authorization:
        api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    else:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing authentication. Get started free (100 queries/month, no credit card):\n"
                "- Claude Code: Run /snipara:quickstart\n"
                "- VS Code: Install 'Snipara' extension and click 'Sign in with GitHub'\n"
                "- Manual: Get an API key at https://snipara.com/dashboard\n"
                "Docs: https://snipara.com/docs/quickstart"
            ),
        )

    client_ip = _get_client_ip(request)
    api_key_info, plan, error, actual_project_id = await validate_request(project_id, api_key, client_ip=client_ip)
    if error:
        raise HTTPException(status_code=401 if "Invalid" in error else 429, detail=error)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # Use actual database ID for all operations (not URL slug)
    if isinstance(body, list):
        responses = [r for req in body if (r := await handle_request(req, actual_project_id, plan))]
        return JSONResponse(responses)

    response = await handle_request(body, actual_project_id, plan)
    return JSONResponse(response) if response else Response(status_code=204)


@router.get("/{project_id}")
async def mcp_sse(
    project_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """
    MCP Server-Sent Events (SSE) endpoint.

    Provides a persistent connection for server-initiated messages.
    Currently used for keep-alive pings every 30 seconds.

    Args:
        project_id: Project ID or slug
        x_api_key: API key via X-API-Key header
        authorization: API key via Authorization: Bearer header

    Returns:
        SSE stream with JSON messages:
        - {"type": "connected"} on connection
        - {"type": "ping"} every 30 seconds
    """
    # Accept X-API-Key header (preferred) or Authorization: Bearer
    if x_api_key:
        api_key = x_api_key
    elif authorization:
        api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    else:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing authentication. Get started free (100 queries/month, no credit card):\n"
                "- Claude Code: Run /snipara:quickstart\n"
                "- VS Code: Install 'Snipara' extension and click 'Sign in with GitHub'\n"
                "- Manual: Get an API key at https://snipara.com/dashboard\n"
                "Docs: https://snipara.com/docs/quickstart"
            ),
        )

    client_ip = _get_client_ip(request)
    _, _, error, _ = await validate_request(project_id, api_key, client_ip=client_ip)
    if error:
        raise HTTPException(status_code=401, detail=error)

    async def stream():
        import asyncio

        yield f"data: {json.dumps({'type': 'connected'})}\n\n"
        try:
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
