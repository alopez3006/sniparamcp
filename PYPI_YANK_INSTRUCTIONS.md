# PyPI Package Yanking Instructions

## Why We're Yanking snipara-fastapi

**Strategic Decision:** All public versions of `snipara-fastapi` are being yanked to protect enterprise revenue. Self-hosting will now only be available through custom enterprise licensing.

## Versions to Yank

| Version | Upload Date | Reason |
|---------|-------------|--------|
| 1.7.6 | 2026-01-24 | Has critical slug/ID bug (500 errors) |
| 1.7.8 | 2026-01-26 | Missing security features |
| 1.7.9 | 2026-01-26 | Missing security features |
| 1.8.0 | 2026-01-28 | Has some security features |
| 1.8.1 | 2026-01-28 | Has security features |
| 1.8.2 | 2026-01-28 | Has security + embeddings upgrade |
| 1.9.0 | 2026-01-29 | Latest, has all features |

## How to Yank (Manual via PyPI Web Interface)

1. **Log into PyPI:** https://pypi.org/account/login/
2. **Navigate to snipara-fastapi:** https://pypi.org/project/snipara-fastapi/
3. **For each version above:**
   - Click "Manage" → "Releases"
   - Select the version
   - Click "Options" → "Yank"
   - **Yank reason:** `Package moved to private distribution. Contact sales@snipara.com for self-hosted enterprise licensing.`

## Yank Reason Message

Use this exact message for all yanked versions:

```
Package moved to private distribution. Contact sales@snipara.com for self-hosted enterprise licensing.
```

## What Happens After Yanking?

- ✅ Package remains visible on PyPI (read-only)
- ✅ Yank reason displayed to visitors
- ❌ `pip install snipara-fastapi` will fail
- ❌ Cannot be re-uploaded (permanent)
- ✅ Users see contact email for enterprise licensing

## Alternative: Programmatic Yanking (If Needed Later)

If you need to yank programmatically in the future:

```bash
# Install latest twine (requires Python 3.8+)
pip install --upgrade twine

# Yank a specific version
twine yank snipara-fastapi==1.7.6 \
  --reason "Package moved to private distribution. Contact sales@snipara.com for self-hosted enterprise licensing."
```

**Note:** Current twine installations (4.0.2, 6.2.0) do not support the `yank` command in this environment. Use web interface instead.

## Impact Analysis

### What Stays Public ✅
- **snipara-mcp** (v2.2.0) - Client package that drives users TO hosted service
- GitHub repos remain public (but marked as "Internal Use Only")

### What Gets Yanked ❌
- All 7 versions of snipara-fastapi (server package)
- Prevents free enterprise self-hosting
- Forces enterprises to contact sales

## New Business Model

### Before (Lost Revenue):
- Enterprise downloads snipara-fastapi from PyPI
- Self-hosts for $0/month
- No reason to pay for hosted service

### After (Captured Revenue):
- Enterprise contacts sales@snipara.com
- Self-Hosted Enterprise license: **$2,000/month minimum**
- Includes support, updates, and custom deployment assistance

## Next Steps After Yanking

1. ✅ Update README.md to point to hosted service
2. ✅ Create SELF_HOSTED_ENTERPRISE.md with licensing details
3. ✅ Update docs/PRICING.md with new tier
4. ✅ Add sales@ contact to all relevant pages

---

**Status:** Pending manual yanking via PyPI web interface
