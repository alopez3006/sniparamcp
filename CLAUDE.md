# CLAUDE.md - Snipara FastAPI Server

This document helps Claude Code understand the snipara-fastapi project.

**PyPI Package:** [`snipara-fastapi`](https://pypi.org/project/snipara-fastapi/)

## Project Overview

Snipara MCP Server is a **Context Optimization as a Service** backend. It indexes documentation and returns optimized context to LLM clients via the Model Context Protocol (MCP).

**Core Value:**
- **90% context reduction** - From 500K tokens to ~5K tokens of highly relevant content
- **Client uses their own LLM** - No vendor lock-in, no LLM costs for us
- **High margins** - Pure compute, no LLM costs passed through

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  MCP Client (Claude Code, Cursor, etc.)                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │ MCP Protocol (HTTP/SSE)
┌─────────────────────▼───────────────────────────────────────────┐
│  Snipara MCP Server (FastAPI)                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Keyword      │  │ Semantic     │  │ Hybrid Ranking        │  │
│  │ Search       │  │ Search       │  │ (Score + Relevance)   │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Smart        │  │ Token        │  │ Session Context       │  │
│  │ Chunking     │  │ Budgeting    │  │ Persistence           │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  PostgreSQL (Neon) + Redis                                      │
│  - Documents, Embeddings, Sessions, Usage Tracking              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Directories

```
snipara-fastapi/
├── src/
│   ├── server.py           # FastAPI app entry point
│   ├── rlm_engine.py       # Context optimization engine
│   ├── auth.py             # API key + OAuth validation
│   ├── db.py               # Database queries
│   ├── usage.py            # Usage tracking
│   └── services/
│       ├── embeddings.py   # Embedding generation
│       ├── indexer.py      # Document indexing
│       ├── chunker.py      # Smart chunking
│       ├── cache.py        # Redis caching
│       └── shared_context.py
├── prisma/                 # Database schema
├── tests/                  # Test suite
└── Dockerfile              # Railway deployment
```

## MCP Tools

### Primary Tools

| Tool | Description |
|------|-------------|
| `rlm_context_query` | Main tool - Returns optimized context for a query |
| `rlm_decompose` | Break complex queries into sub-queries |
| `rlm_multi_query` | Execute multiple queries in one call |

### Document Management

| Tool | Description |
|------|-------------|
| `rlm_upload_document` | Upload or update a single document |
| `rlm_sync_documents` | Bulk sync multiple documents (CI/CD) |
| `rlm_settings` | Get project settings |

### Supporting Tools

| Tool | Description |
|------|-------------|
| `rlm_search` | Regex pattern search |
| `rlm_inject` | Inject session context |
| `rlm_context` | Show current session context |
| `rlm_clear_context` | Clear session context |
| `rlm_stats` | Documentation statistics |
| `rlm_sections` | List all sections |
| `rlm_read` | Read specific line ranges |

### Summary Storage

| Tool | Description |
|------|-------------|
| `rlm_store_summary` | Store LLM-generated summary |
| `rlm_get_summaries` | Retrieve stored summaries |
| `rlm_delete_summary` | Delete summaries |

### Shared Context

| Tool | Description |
|------|-------------|
| `rlm_shared_context` | Get merged context from collections |
| `rlm_list_templates` | List prompt templates |
| `rlm_get_template` | Get specific template |

## Integration with rlm-runtime

Snipara MCP works together with **rlm-runtime** MCP for powerful workflows:

```
Claude Code (Single LLM Instance)
    │
    ├── snipara-mcp (context retrieval)
    │   └── rlm_context_query, rlm_search, rlm_shared_context
    │
    └── rlm-runtime-mcp (code sandbox)
        └── execute_python, get/set_repl_context
```

### Combined Workflow Example

```
User: "Implement the pricing algorithm from our spec"

Claude workflow:
1. [snipara: rlm_context_query] → Get pricing specification
2. [snipara: rlm_search] → Find test cases from documentation
3. [rlm-runtime: execute_python] → Implement algorithm
4. [rlm-runtime: set_repl_context] → Store implementation
5. [rlm-runtime: execute_python] → Run test cases
6. Iterate until tests pass
```

### When to Use Each MCP

| Task Type | Use snipara-mcp | Use rlm-runtime |
|-----------|-----------------|-----------------|
| Understanding codebase | Y | |
| Finding patterns | Y | |
| Team best practices | Y | |
| Domain knowledge | Y | |
| Math/calculations | | Y |
| Data processing | | Y |
| Algorithm verification | | Y |
| Code structure analysis | | Y |
| **Complex tasks** | Y + Y | Y + Y |

### rlm-runtime Installation

```bash
pip install rlm-runtime

# MCP tools available:
# - execute_python: Sandboxed Python execution (RestrictedPython)
# - get_repl_context: Get persistent variables
# - set_repl_context: Set persistent variables
# - clear_repl_context: Reset state
```

**Repository:** https://github.com/alopez3006/rlm-runtime
**PyPI:** https://pypi.org/project/rlm-runtime/

## Installation

```bash
# From PyPI
pip install snipara-fastapi

# From source
pip install -r requirements.txt
```

## Development

```bash
# Run locally
uvicorn src.server:app --reload --port 8000

# Run tests
pytest tests/

# Type check
mypy src/
```

## Deployment

Deployed on Railway with auto-deploy from main branch.

```bash
# Auto-deploy (default)
git push origin main

# Manual deploy (if auto-deploy fails)
railway link    # Link to project (first time only)
railway up      # Deploy current code
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL (Neon) connection | Yes |
| `REDIS_URL` | Redis for caching | Optional |
| `OPENAI_API_KEY` | For embeddings | Pro+ plans |
| `LOG_LEVEL` | Logging level | No |

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add MCP tool | `src/rlm_engine.py` |
| Modify search | `src/services/indexer.py` |
| Change chunking | `src/services/chunker.py` |
| Update auth | `src/auth.py` |
| Add caching | `src/services/cache.py` |

## Recent Changes

- **rlm-runtime Integration**: Combined workflows with code sandbox
- **Summary Storage**: Store/retrieve LLM-generated summaries
- **Shared Context**: Collections with category-based budgeting
- **OAuth Device Flow**: Secure authentication without API key copying
- **Document Upload**: Direct upload via MCP tools

---

*Last updated: January 2025*

---

## BMad Method (Global Commands)

BMad Method is available globally across all Claude Code sessions. Use these slash commands for structured workflows.

### Quick Start
```
/bmad/core/agents/bmad-master    # Main BMad agent - start here
/bmad-help                        # Get guidance on what to do next
```

### Core Workflows
| Command | Purpose |
|---------|---------|
| `/bmad/bmm/workflows/prd` | Create Product Requirements Document |
| `/bmad/bmm/workflows/create-architecture` | Design system architecture |
| `/bmad/bmm/workflows/create-story` | Create user stories |
| `/bmad/bmm/workflows/create-epics-and-stories` | Full epic breakdown |
| `/bmad/bmm/workflows/dev-story` | Develop/implement a story |
| `/bmad/bmm/workflows/quick-dev` | Quick development flow |
| `/bmad/bmm/workflows/sprint-planning` | Sprint planning session |

### Planning & Design
| Command | Purpose |
|---------|---------|
| `/bmad/bmm/workflows/create-product-brief` | Initial product brief |
| `/bmad/bmm/workflows/check-implementation-readiness` | Verify before coding |
| `/bmad/core/workflows/brainstorming` | Brainstorming session |

### Documentation & Diagrams
| Command | Purpose |
|---------|---------|
| `/bmad/bmm/workflows/document-project` | Generate project docs |
| `/bmad/bmm/workflows/create-excalidraw-diagram` | Create diagrams |
| `/bmad/bmm/workflows/create-excalidraw-flowchart` | Create flowcharts |
| `/bmad/bmm/workflows/create-excalidraw-wireframe` | Create wireframes |
| `/bmad/bmm/workflows/create-excalidraw-dataflow` | Create data flow diagrams |

### Installation
BMad is installed globally at `~/bmad-global/` and symlinked to `~/.claude/commands/bmad/`.
