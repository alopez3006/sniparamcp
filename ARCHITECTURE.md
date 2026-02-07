# Snipara MCP Architecture

## Repositories

| Repo                       | Type               | URL                                         | Déploiement |
| -------------------------- | ------------------ | ------------------------------------------- | ----------- |
| **Snipara/snipara**        | Monorepo privé     | `git@github.com:Snipara/snipara.git`        | -           |
| **Snipara/snipara-server** | Backend API public | `git@github.com:Snipara/snipara-server.git` | Railway     |
| **snipara-mcp** (PyPI)     | Client MCP         | https://pypi.org/project/snipara-mcp/       | pip install |

---

## Structure du Monorepo

```
RLMSaas/
├── apps/
│   ├── web/                          # Next.js Dashboard (Vercel)
│   └── mcp-server/
│       ├── snipara-mcp/              # Client MCP PyPI (pip install snipara-mcp)
│       │   ├── src/snipara_mcp/
│       │   │   ├── __init__.py
│       │   │   └── server.py         # MCP stdio client
│       │   ├── pyproject.toml
│       │   └── README.md
│       └── src/                      # Backend API (Railway) - copié vers snipara-fastapi
│           ├── server.py             # FastAPI server
│           ├── rlm_engine.py         # Context optimization engine
│           ├── auth.py
│           ├── db.py
│           └── ...
```

---

## Composants

### 1. Client MCP (`snipara-mcp` sur PyPI)

**Localisation:** `apps/mcp-server/snipara-mcp/`

**Description:** Package Python installé par les utilisateurs via `pip install snipara-mcp`.
Communique avec le backend via HTTP REST.

**Publication:**

```bash
cd apps/mcp-server/snipara-mcp
# Mettre à jour version dans pyproject.toml et __init__.py
rm -rf dist/
/opt/miniconda3/bin/python -m build
/opt/miniconda3/bin/python -m twine upload dist/*
```

**Version actuelle:** 1.1.0

---

### 2. Backend API (`snipara-fastapi` sur GitHub)

**Localisation source:** `apps/mcp-server/src/`

**Repo public:** `git@github.com:Snipara/snipara-server.git`

**Description:** Serveur FastAPI déployé sur Railway. Gère l'authentification,
le stockage des documents, et l'engine de context optimization.

**Synchronisation avec le repo public:**

```bash
# Cloner le repo public
cd /tmp
git clone git@github.com:Snipara/snipara-server.git
cd snipara-server

# Copier les fichiers mis à jour depuis le monorepo
cp -r /Users/alopez/Devs/RLMSaas/apps/mcp-server/src/* src/
cp /Users/alopez/Devs/RLMSaas/apps/mcp-server/requirements.txt .
cp /Users/alopez/Devs/RLMSaas/apps/mcp-server/pyproject.toml .

# Commit et push
git add -A
git commit -m "sync: update from monorepo"
git push origin main
```

**Déploiement:** Automatique via Railway (connecté au repo GitHub)

---

### 3. Dashboard Web

**Localisation:** `apps/web/`

**Déploiement:** Railway (connecté au monorepo GitHub)

**URLs:**

- Production: https://www.snipara.com (avec redirection depuis snipara.com)
- Note: Les assets email doivent utiliser `www.snipara.com` pour éviter la redirection 301

**Déploiement manuel (si besoin):**

```bash
cd /Users/alopez/Devs/RLMSaas
git add -A
git commit -m "fix(web): description"
git push origin main
# Railway déploie automatiquement
```

---

## Flux de données

```
┌─────────────────────────────────────────────────────────────────┐
│  Client LLM (Claude Code, Cursor, etc.)                        │
│       │                                                         │
│       ▼                                                         │
│  snipara-mcp (PyPI package)                                    │
│  - Installé localement via pip                                 │
│  - Communique via stdio avec le client LLM                     │
│       │                                                         │
│       ▼ HTTP REST                                               │
│  Backend API (Railway)                                          │
│  - https://api.snipara.com                                     │
│  - Auth via API Key                                            │
│  - Context optimization engine                                  │
│       │                                                         │
│       ▼                                                         │
│  PostgreSQL (Neon)                                              │
│  - Documents, users, queries, etc.                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Commandes fréquentes

### Publier une nouvelle version du client MCP

```bash
cd /Users/alopez/Devs/RLMSaas/apps/mcp-server/snipara-mcp

# 1. Mettre à jour les versions
# - pyproject.toml: version = "X.Y.Z"
# - src/snipara_mcp/__init__.py: __version__ = "X.Y.Z"

# 2. Build
rm -rf dist/
/opt/miniconda3/bin/python -m build

# 3. Upload sur PyPI
/opt/miniconda3/bin/python -m twine upload dist/*

# 4. Commit et push
cd /Users/alopez/Devs/RLMSaas
git add -A
git commit -m "chore(mcp): bump snipara-mcp to X.Y.Z"
git push origin main
```

### Synchroniser le backend avec le repo public

```bash
# 1. Cloner le repo public
cd /tmp && rm -rf snipara-fastapi
git clone git@github.com:Snipara/snipara-server.git
cd snipara-server

# 2. Copier les fichiers
cp -r /Users/alopez/Devs/RLMSaas/apps/mcp-server/src/* src/
cp /Users/alopez/Devs/RLMSaas/apps/mcp-server/requirements.txt .
# Ne pas copier pyproject.toml si différent

# 3. Commit et push
git add -A
git commit -m "sync: description des changements"
git push origin main

# Railway déploiera automatiquement
```

### Installer le client MCP localement (dev)

```bash
cd /Users/alopez/Devs/RLMSaas/apps/mcp-server/snipara-mcp
pip install -e .
```

### Déployer le Dashboard Web (Railway)

```bash
cd /Users/alopez/Devs/RLMSaas

# Commit les changements
git add -A
git commit -m "fix(web): description des changements"
git push origin main

# Railway déploie automatiquement depuis le monorepo
# Logs disponibles via: railway logs
```

---

## Variables d'environnement

### Client MCP (snipara-mcp)

```
SNIPARA_API_KEY=rlm_xxx          # Clé API du projet
SNIPARA_PROJECT_ID=project_slug  # Slug du projet
SNIPARA_API_URL=https://api.snipara.com  # (optionnel)
```

### Backend (Railway)

```
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
JWT_SECRET=xxx
```

---

## MCP Tools

### Context Optimization Tools

| Tool                | Description                                     | Plan |
| ------------------- | ----------------------------------------------- | ---- |
| `rlm_context_query` | Main tool - optimized context with token budget | All  |
| `rlm_ask`           | Legacy query (deprecated)                       | All  |
| `rlm_search`        | Regex pattern search                            | All  |
| `rlm_stats`         | Documentation statistics                        | All  |
| `rlm_sections`      | List all sections                               | All  |
| `rlm_read`          | Read specific line ranges                       | All  |

### Session Context Tools

| Tool                | Description                         | Plan |
| ------------------- | ----------------------------------- | ---- |
| `rlm_inject`        | Inject session context              | All  |
| `rlm_context`       | Show current session context        | All  |
| `rlm_clear_context` | Clear session context               | All  |
| `rlm_settings`      | Get project settings from dashboard | All  |

### Recursive Context Tools

| Tool              | Description                                   | Plan |
| ----------------- | --------------------------------------------- | ---- |
| `rlm_decompose`   | Break query into sub-queries                  | Pro+ |
| `rlm_multi_query` | Execute multiple queries in one call          | Pro+ |
| `rlm_plan`        | Generate execution plan for complex questions | Pro+ |

### Summary Storage Tools

| Tool                 | Description                 | Plan |
| -------------------- | --------------------------- | ---- |
| `rlm_store_summary`  | Store LLM-generated summary | Pro+ |
| `rlm_get_summaries`  | Retrieve stored summaries   | Pro+ |
| `rlm_delete_summary` | Delete stored summaries     | Pro+ |

### Shared Context Tools

| Tool                 | Description                                | Plan |
| -------------------- | ------------------------------------------ | ---- |
| `rlm_shared_context` | Get merged context from linked collections | Pro+ |
| `rlm_list_templates` | List available prompt templates            | Pro+ |
| `rlm_get_template`   | Get a specific prompt template             | Pro+ |

**Shared Context Parameters:**

```json
{
  "query": "How should I handle errors?",
  "max_tokens": 4000,
  "categories": ["MANDATORY", "BEST_PRACTICES"],
  "include_project_docs": true
}
```

**Document Categories (token budget allocation):**

- `MANDATORY` (40%): Rules that MUST be followed
- `BEST_PRACTICES` (30%): Recommended patterns
- `GUIDELINES` (20%): Nice-to-have suggestions
- `REFERENCE` (10%): Background info

---

## Search Relevance Filtering

### Ubiquitous Keyword Detection (Auto-Detection)

**Problem:** Project-specific terms like the project name appear everywhere in documentation. When users search for "What is Snipara's core value proposition?", sections like "Snipara VS Code Extension" rank highly simply because they contain "Snipara" - even though they're not relevant to the query.

**Solution:** Auto-detect ubiquitous keywords at index time and exclude them from "distinctive match" requirements.

**How It Works:**

1. **At Document Load Time** (`load_documents()`):
   - Analyze all section titles
   - Count keyword frequency across all titles
   - Keywords appearing in >40% of titles are marked as "ubiquitous"
   - Project slug is always added as ubiquitous (e.g., "snipara" for project "snipara")

2. **At Search Time** (KEYWORD and HYBRID modes):
   - For queries with 3+ keywords, require either:
     - At least 1 NON-ubiquitous keyword match in the section title, OR
     - High keyword score (>50) OR high semantic score (>40)
   - Ubiquitous keywords are excluded from the "distinctive match" check

**Example:**

```
Query: "What is Snipara's core value proposition?"
Keywords: ["snipara", "core", "value", "proposition"]

Ubiquitous keywords for project "snipara": {"snipara"}

Section: "Snipara VS Code Extension"
- "snipara" matches but is ubiquitous → doesn't count
- "core" not in title
- "value" not in title
- "proposition" not in title
- distinctive_title_hits = 0 → FILTERED OUT (unless high kw/sem score)

Section: "Core Value Proposition"
- "snipara" is ubiquitous → doesn't count
- "core" matches → counts as distinctive!
- "value" matches → counts as distinctive!
- distinctive_title_hits = 2 → KEPT
```

**Implementation Details:**

| Component | Location | Description |
|-----------|----------|-------------|
| `DocumentationIndex.ubiquitous_keywords` | `rlm_engine.py:432` | Field storing auto-detected keywords |
| `_compute_ubiquitous_keywords()` | `rlm_engine.py:608` | Computes ubiquitous keywords from titles |
| KEYWORD mode filter | `rlm_engine.py:1703-1737` | Filters sections in keyword search |
| HYBRID mode filter | `rlm_engine.py:1844-1862` | Filters sections in hybrid search |

**Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Frequency threshold | 40% | Keywords in >40% of titles are ubiquitous |
| Min query keywords | 3 | Filter only applies to 3+ keyword queries |
| Keyword score fallback | >50 | Keep section if raw keyword score exceeds this |
| Semantic score fallback | >40 | Keep section if semantic score exceeds this |

**Benchmark Results:**

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Precision@5 | ~60% | **92.4%** |
| Recall | ~70% | **95.0%** |
| Success Rate | ~80% | **100%** |
| Rating | Good | **EXCELLENT** |

**Works for All Users:** Unlike hardcoded keywords, this auto-detection works for ANY project - users don't need to configure anything.

---

### Stop Words and Stemming

**Stop Words:** Common words filtered from query matching to improve precision.

```python
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", ...
    "to", "of", "in", "for", "on", "with", "at", "by", ...
})
```

**Stemming:** Basic suffix stripping to match morphological variants.

```python
# Examples:
"pricing" → "pric" → matches "prices", "priced"
"authentication" → "authentic" → matches "authenticating"
```

The stemmer handles common suffixes: `-ing`, `-ed`, `-es`, `-s`, `-tion`, `-ment`, `-ness`, `-ity`, `-able`, `-ible`, `-ful`, `-less`, `-ous`, `-ive`, `-er`, `-est`.

---

## Historique des versions

| Version | Date       | Changements                                                |
| ------- | ---------- | ---------------------------------------------------------- |
| 1.9.1   | 2026-02-06 | Auto-detect ubiquitous keywords for relevance filtering    |
| 1.2.0   | 2026-01-20 | + rlm_shared_context, rlm_list_templates, rlm_get_template |
| 1.1.1   | 2026-01-20 | Fix Repository URL dans pyproject.toml                     |
| 1.1.0   | 2026-01-20 | + rlm_settings, + sync dashboard settings (cache 5min)     |
| 1.0.1   | 2026-01-20 | Fix rlm_stats formatting (non-numeric total_tokens)        |
| 1.0.0   | 2026-01-19 | Version initiale                                           |
