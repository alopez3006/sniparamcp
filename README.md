# Snipara FastAPI - MCP Server

**Enterprise-grade context optimization server for LLMs**

---

## 🚨 Important Notice

This repository contains the **Self-Hosted Enterprise** version of Snipara's MCP server. **Public distribution via PyPI has been discontinued** as of January 2026.

### For Most Users: Use Hosted Snipara

**👉 Recommended:** [Get started with hosted Snipara](https://snipara.com/signup) (5 minutes setup)

- ✅ **No infrastructure management** - We handle PostgreSQL, Redis, scaling, monitoring
- ✅ **Always up-to-date** - Automatic security patches and feature updates
- ✅ **Free tier available** - 100 queries/month, perfect for personal projects
- ✅ **Starts at $19/month** - Pro plan with 5,000 queries/month

**Install the client:**
```bash
pip install snipara-mcp
snipara-mcp-login
```

Then add to your MCP config (e.g., Claude Code, Cursor, Windsurf):
```json
{
  "mcpServers": {
    "snipara": {
      "command": "snipara-mcp",
      "args": ["--project", "your-project-slug"]
    }
  }
}
```

---

## 🏢 For Enterprises: Self-Hosted Deployment

If your organization requires **on-premises deployment** for compliance, data sovereignty, or air-gapped environments, we offer **Self-Hosted Enterprise licensing**.

### Why Self-Host?

- **Data sovereignty:** Keep all documentation and embeddings within your infrastructure
- **Compliance:** Meet HIPAA, SOC2, FedRAMP, ISO 27001 requirements
- **Air-gapped environments:** Deploy in networks without internet access
- **Custom infrastructure:** Integrate with existing PostgreSQL, Redis, SSO, monitoring

### What's Included

- ✅ Full source code with all features (security, agents, team context)
- ✅ Professional deployment support (8 hours included)
- ✅ Quarterly security patches and feature updates
- ✅ 24-hour email/Slack support SLA
- ✅ Architecture docs, API reference, runbooks

### Pricing

**$2,000/month minimum** (billed annually: $24,000/year)

- Up to 50,000 queries/month
- Unlimited projects and users
- All enterprise features
- Standard support (24-hour SLA)

Volume pricing available for 100K+ queries/month.

**📄 [See full details in SELF_HOSTED_ENTERPRISE.md](./SELF_HOSTED_ENTERPRISE.md)**

### Contact Sales

**Email:** sales@snipara.com

Include:
- Company name and size
- Expected query volume
- Infrastructure preferences (AWS, GCP, Azure, on-prem)
- Compliance requirements

---

## 🔧 Technical Details

### What is Snipara?

Snipara is a **context optimization service** that helps LLMs work within token limits. We index your documentation, compress it intelligently, and deliver only the most relevant context to your LLM.

**Business Model:** We charge for context optimization, NOT LLM inference. You use your own Claude, GPT, Gemini, etc.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  MCP Client (Claude Code, Cursor, Windsurf)            │
│  → snipara-mcp (lightweight client)                     │
│    → FastAPI MCP Server (this repo)                     │
│      → RLM Engine (search + rank + compress)            │
│      → PostgreSQL (documents + embeddings)              │
│      → Redis (rate limiting + anti-scan protection)     │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI 0.109+ |
| Database | PostgreSQL 14+ with Prisma ORM |
| Cache | Redis 7+ |
| Embeddings | sentence-transformers (bge-large-en-v1.5, 1024 dims) |
| Vector Search | NumPy cosine similarity |
| Auth | API keys + OAuth (GitHub, Google) |

### Features

#### Core Features
- **Semantic Search:** 1024-dimensional embeddings for accurate document retrieval
- **Hybrid Search:** Combines keyword (BM25) and semantic search with configurable weighting
- **Token Budgeting:** Automatically fits results within specified token limits
- **Multi-Project Queries:** Search across all projects in a team

#### Security Features (v1.9.0+)
- **Audit Logging:** All API calls logged to PostgreSQL for compliance
- **Anti-Scan Protection:** Rate limiting + project enumeration prevention
- **Multi-Project ACL:** Fine-grained access control per project
- **API Key Management:** Team keys with scoped permissions

#### Agent Features
- **Memory System:** Store and semantically recall facts, decisions, learnings
- **Swarm Coordination:** Multi-agent task queues with distributed state
- **Resource Claims:** Exclusive access to files/functions during development

---

## 📚 Documentation

### For Hosted Users
- [Getting Started Guide](https://docs.snipara.com/getting-started)
- [MCP Tools Reference](https://docs.snipara.com/mcp-tools)
- [API Reference](https://docs.snipara.com/api)
- [Pricing](https://docs.snipara.com/pricing)

### For Self-Hosted Enterprise
- [Self-Hosted Enterprise Details](./SELF_HOSTED_ENTERPRISE.md)
- [Architecture Documentation](./ARCHITECTURE.md)
- [Deployment Guide](./docs/deployment.md) *(Coming soon)*
- [Operations Runbook](./docs/operations.md) *(Coming soon)*

---

## 🚀 Quick Start (Self-Hosted Enterprise Only)

**Prerequisites:**
- PostgreSQL 14+ (with pgvector extension)
- Redis 7+
- Python 3.11+

**Setup:**
```bash
# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Configure database, Redis, API keys
# See SELF_HOSTED_ENTERPRISE.md for full setup

# Run migrations
uv run prisma generate
uv run prisma migrate deploy

# Start server
uv run uvicorn src.server:app --host 0.0.0.0 --port 8000
```

**Production deployment requires:**
- SSL certificate (Let's Encrypt or custom)
- Load balancer (AWS ALB, nginx, Cloudflare)
- Auto-scaling (Kubernetes or Docker Swarm)
- Monitoring (Prometheus + Grafana)

**⚠️ Note:** Self-hosted deployment is complex. We **strongly recommend** our white-glove deployment service (included in enterprise licensing).

---

## 📦 Why PyPI Distribution Was Discontinued

As of **January 2026**, all versions of `snipara-fastapi` on PyPI have been yanked. Here's why:

### Business Reason
- **Lost enterprise revenue:** Organizations were self-hosting for free instead of paying for hosted service
- **Competitive disadvantage:** No differentiation between free self-host and $499/mo Enterprise plan
- **Support burden:** Community users expected free support for complex self-hosted deployments

### Technical Reason
- **Critical bugs in early versions:** v1.7.6 had a slug/ID bug causing 500 errors
- **Missing security features:** Versions < 1.8.0 lack audit logging, anti-scan protection
- **Complex infrastructure:** Requires PostgreSQL + Redis + GPU (optional) - not "pip install" simple

### New Strategy
- **Hosted service:** Simple, managed, always up-to-date (starts at $0/month free tier)
- **Self-Hosted Enterprise:** Custom licensing for organizations that truly need on-prem ($2K+/month)
- **Focus on value:** We optimize context, you use your LLM - clear value proposition

---

## 🤝 Contributing

This repository is **not open to public contributions** as it contains enterprise features under commercial license.

For bug reports or feature requests for the **hosted service**, please contact support@snipara.com.

---

## 📄 License

**Proprietary** - All rights reserved by Snipara

This software is provided under a **commercial license** for Self-Hosted Enterprise customers only.

**Public distribution via PyPI has been discontinued** as of January 2026.

For licensing inquiries: sales@snipara.com

---

## 🔗 Links

- **Website:** https://snipara.com
- **Documentation:** https://docs.snipara.com
- **Hosted Signup:** https://snipara.com/signup
- **Status Page:** https://status.snipara.com
- **Blog:** https://snipara.com/blog

**Contact:**
- **Sales:** sales@snipara.com
- **Support:** support@snipara.com
- **Security:** security@snipara.com

---

*Last updated: January 2026*
