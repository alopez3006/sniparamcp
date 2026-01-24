# Snipara MCP Server

[![PyPI version](https://badge.fury.io/py/snipara-fastapi.svg)](https://pypi.org/project/snipara-fastapi/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Context Optimization Server for LLMs via the Model Context Protocol (MCP).

Turn **500K tokens** of documentation into **5K tokens** of perfectly relevant context.

## What is Snipara?

Snipara is a **Context Optimization as a Service** platform. We don't run your LLM - we optimize and deliver the most relevant context to **your LLM** (Claude, GPT, Gemini, etc.).

**Key Benefits:**
- **90% cost reduction** - From $0.83 to $0.08 per query
- **Near-infinite context** - Handle docs 100x larger than your context window
- **Session persistence** - Context survives compaction events
- **Use your own LLM** - Zero vendor lock-in

## MCP Tools

This server exposes the following MCP tools:

### Primary Tools

| Tool | Description |
|------|-------------|
| `rlm_context_query` | Main tool - Returns optimized context for a query with token budget |
| `rlm_decompose` | Break complex queries into sub-queries |
| `rlm_multi_query` | Execute multiple queries in one call |

### Document Management Tools

| Tool | Description |
|------|-------------|
| `rlm_upload_document` | Upload or update a single document |
| `rlm_sync_documents` | Bulk sync multiple documents (for CI/CD) |
| `rlm_settings` | Get project settings from dashboard |

### Supporting Tools

| Tool | Description |
|------|-------------|
| `rlm_ask` | Query documentation with natural language |
| `rlm_search` | Search for patterns (regex) |
| `rlm_inject` | Inject session context |
| `rlm_context` | Show current session context |
| `rlm_clear_context` | Clear session context |
| `rlm_stats` | Show documentation statistics |
| `rlm_sections` | List all documentation sections |
| `rlm_read` | Read specific line ranges |

## Quick Start

### Using Snipara Cloud (Recommended)

Sign up at [snipara.com](https://snipara.com) and get your MCP endpoint:

```
https://api.snipara.com/v1/YOUR_PROJECT_ID/mcp
```

### Self-Hosting

#### Install from PyPI

```bash
pip install snipara-fastapi
```

#### Or install from source

1. Clone this repository:
```bash
git clone https://github.com/alopez3006/snipara-fastapi.git
cd snipara-fastapi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database URL and settings
```

4. Run the server:
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Deploy to Railway

Railway auto-deploys from main branch. If auto-deploy fails, deploy manually:

```bash
railway link    # Link to existing project (first time only)
railway up      # Deploy current code
```

## Integration

### Claude Code

```bash
claude mcp add snipara https://api.snipara.com/v1/YOUR_PROJECT_ID/mcp
```

### Cursor

Add to your `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "snipara": {
      "url": "https://api.snipara.com/v1/YOUR_PROJECT_ID/mcp"
    }
  }
}
```

### Continue.dev

Add to your `~/.continue/config.json`:
```json
{
  "mcpServers": [
    {
      "name": "snipara",
      "transport": {
        "type": "sse",
        "url": "https://api.snipara.com/v1/YOUR_PROJECT_ID/mcp/sse"
      }
    }
  ]
}
```

## API Reference

### POST /v1/{project_id}/mcp

MCP endpoint for tool calls.

**Headers:**
- `Authorization: Bearer YOUR_API_KEY`
- `Content-Type: application/json`

**Example Request:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "rlm_context_query",
    "arguments": {
      "query": "How does authentication work?",
      "max_tokens": 4000,
      "search_mode": "hybrid"
    }
  }
}
```

**Example Response:**
```json
{
  "sections": [
    {
      "title": "Authentication Flow",
      "content": "...",
      "file": "docs/auth.md",
      "lines": [45, 120],
      "relevance_score": 0.94,
      "token_count": 1200
    }
  ],
  "total_tokens": 3800,
  "suggestions": ["Also check: docs/security.md"]
}
```

## Document Upload

### Via MCP Tool

Upload documents directly from your LLM client:

```
rlm_upload_document(path="CLAUDE.md", content="# My Project\n...")
```

### Via Webhook (CI/CD)

Sync documents on git push:

```bash
curl -X POST "https://api.snipara.com/v1/YOUR_PROJECT_ID/webhook/sync" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"path": "CLAUDE.md", "content": "# My Project..."},
      {"path": "docs/api.md", "content": "# API Reference..."}
    ],
    "delete_missing": false
  }'
```

### GitHub Action Example

```yaml
name: Sync Docs to Snipara
on:
  push:
    paths: ['**/*.md']
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Collect and sync docs
        run: |
          # Build JSON payload with all markdown files
          docs=$(find . -name '*.md' -not -path './node_modules/*' | while read f; do
            content=$(cat "$f" | jq -Rs .)
            echo "{\"path\": \"${f#./}\", \"content\": $content}"
          done | jq -s '{documents: .}')

          curl -X POST "${{ secrets.SNIPARA_WEBHOOK_URL }}" \
            -H "X-API-Key: ${{ secrets.SNIPARA_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d "$docs"
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `OPENAI_API_KEY` | For embeddings (semantic search) | For Pro+ |
| `REDIS_URL` | For caching | Optional |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, etc.) | No |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  MCP Client (Claude Code, Cursor, etc.)             │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
┌─────────────────────▼───────────────────────────────┐
│  Snipara MCP Server                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Keyword      │  │ Semantic     │  │ Hybrid    │ │
│  │ Search       │  │ Search       │  │ Ranking   │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Chunking     │  │ Token        │  │ Session   │ │
│  │ Engine       │  │ Budgeting    │  │ Context   │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  PostgreSQL (Documents, Sessions, Usage)            │
└─────────────────────────────────────────────────────┘
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Website:** [snipara.com](https://snipara.com)
- **Documentation:** [snipara.com/docs](https://snipara.com/docs)
- **PyPI:** [pypi.org/project/snipara-fastapi](https://pypi.org/project/snipara-fastapi/)
- **Issues:** [GitHub Issues](https://github.com/alopez3006/snipara-fastapi/issues)

