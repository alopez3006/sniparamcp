"""License key validation and trial management for self-hosted Snipara.

License keys are signed JWTs (HMAC-SHA256) containing:
    - plan: str ("PRO", "TEAM", "ENTERPRISE")
    - exp: int (Unix timestamp - expiry date)
    - iss: str ("snipara.com")
    - sub: str (customer email or org name)
    - features: list[str] (optional feature flags)

Trial tracking uses a PostgreSQL table (snipara_license_state) to
record first-run timestamp. After 30 days without a valid license
key, the server degrades to FREE tier.

Design decisions:
    - Offline JWT validation (no phone-home) for air-gapped deployments
    - PostgreSQL-based trial tracking (Docker setup includes PG)
    - Existing plan-gating in rlm_engine.py is reused unchanged
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

TRIAL_DURATION_DAYS = 30
LICENSE_ISSUER = "snipara.com"

# HMAC-SHA256 verification key for license JWTs
LICENSE_VERIFICATION_KEY = "snipara-license-verification-key-v1"

# Cache resolved license to avoid repeated DB queries
_cached_license: "LicenseInfo | None" = None
_cache_timestamp: float = 0
_CACHE_TTL_SECONDS = 300  # Re-check every 5 minutes


@dataclass
class LicenseInfo:
    """Resolved license information."""

    plan: str  # "FREE", "PRO", "TEAM", "ENTERPRISE"
    is_trial: bool  # True if in 30-day trial period
    trial_days_left: int  # Days remaining in trial (0 if not trial)
    licensed_to: str  # Customer identifier
    expires_at: datetime | None  # License expiry
    features: list[str] = field(default_factory=list)


# FREE tier tools (available without license after trial expires)
FREE_TIER_TOOLS = {
    "rlm_ask",
    "rlm_search",
    "rlm_read",
    "rlm_sections",
    "rlm_stats",
    "rlm_inject",
    "rlm_context",
    "rlm_clear_context",
    "rlm_settings",
    "rlm_context_query",
    "rlm_upload_document",
}


def _b64url_decode(data: str) -> bytes:
    """Decode base64url without padding."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def _b64url_encode(data: bytes) -> str:
    """Encode base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def decode_license_jwt(token: str) -> dict | None:
    """Decode and verify a Snipara license JWT (HMAC-SHA256).

    Returns the payload dict if valid, None if invalid/expired.
    """
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            logger.warning("License key: invalid JWT format (expected 3 parts)")
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(
            LICENSE_VERIFICATION_KEY.encode(),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_sig = _b64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.warning("License key: invalid signature")
            return None

        # Decode payload
        payload = json.loads(_b64url_decode(payload_b64))

        # Verify issuer
        if payload.get("iss") != LICENSE_ISSUER:
            logger.warning(f"License key: unexpected issuer '{payload.get('iss')}'")
            return None

        # Verify expiry
        exp = payload.get("exp", 0)
        if exp < time.time():
            logger.warning("License key: expired")
            return None

        return payload

    except Exception as e:
        logger.warning(f"License key: decode error: {e}")
        return None


def generate_license_jwt(
    plan: str,
    sub: str,
    days_valid: int = 365,
    features: list[str] | None = None,
) -> str:
    """Generate a signed license JWT. Used by Snipara admin tools.

    Args:
        plan: Plan name (PRO, TEAM, ENTERPRISE)
        sub: Customer identifier (email or org name)
        days_valid: Days until expiry
        features: Optional feature flags

    Returns:
        Signed JWT string
    """
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": LICENSE_ISSUER,
        "sub": sub,
        "plan": plan,
        "exp": int(time.time()) + (days_valid * 86400),
        "iat": int(time.time()),
        "features": features or [],
    }

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(
        LICENSE_VERIFICATION_KEY.encode(),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


async def ensure_license_table() -> None:
    """Create the license state table if it doesn't exist.

    Uses raw SQL to keep the license system decoupled from Prisma schema.
    """
    from .db import get_db

    db = await get_db()
    try:
        await db.execute_raw(
            """
            CREATE TABLE IF NOT EXISTS snipara_license_state (
                key VARCHAR(64) PRIMARY KEY,
                value TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        logger.debug("License state table ready")
    except Exception as e:
        logger.warning(f"Could not create license state table: {e}")


async def get_trial_start_date() -> datetime | None:
    """Get the trial start date from PostgreSQL."""
    from .db import get_db

    db = await get_db()
    try:
        result = await db.query_raw(
            "SELECT started_at FROM snipara_license_state WHERE key = 'trial_start' LIMIT 1"
        )
        if result:
            started_at = result[0].get("started_at")
            if isinstance(started_at, str):
                return datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if isinstance(started_at, datetime):
                if started_at.tzinfo is None:
                    return started_at.replace(tzinfo=UTC)
                return started_at
        return None
    except Exception as e:
        logger.warning(f"Could not read trial start date: {e}")
        return None


async def record_trial_start() -> datetime:
    """Record the trial start date in PostgreSQL."""
    from .db import get_db

    db = await get_db()
    now = datetime.now(UTC)
    try:
        await db.execute_raw(
            """
            INSERT INTO snipara_license_state (key, value, started_at)
            VALUES ('trial_start', $1, $2)
            ON CONFLICT (key) DO NOTHING
            """,
            now.isoformat(),
            now,
        )
        logger.info(f"Trial started: {now.isoformat()}")
    except Exception as e:
        logger.warning(f"Could not record trial start: {e}")
    return now


async def resolve_license() -> LicenseInfo:
    """Resolve the current license state.

    Priority:
        1. If SNIPARA_LICENSE_KEY is set and valid -> use licensed plan
        2. If in trial period -> use ENTERPRISE (all features)
        3. If trial expired and no key -> FREE tier
    """
    global _cached_license, _cache_timestamp

    # Return cached result if fresh enough
    if _cached_license and (time.time() - _cache_timestamp) < _CACHE_TTL_SECONDS:
        return _cached_license

    from .config import settings

    license_key = settings.snipara_license_key

    # Case 1: Valid license key
    if license_key:
        payload = decode_license_jwt(license_key)
        if payload and payload.get("exp", 0) > time.time():
            info = LicenseInfo(
                plan=payload.get("plan", "PRO"),
                is_trial=False,
                trial_days_left=0,
                licensed_to=payload.get("sub", "unknown"),
                expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
                features=payload.get("features", []),
            )
            _cached_license = info
            _cache_timestamp = time.time()
            return info
        else:
            logger.warning("SNIPARA_LICENSE_KEY is set but invalid or expired")

    # Case 2/3: Check trial period
    trial_start = await get_trial_start_date()
    if trial_start is None:
        trial_start = await record_trial_start()

    now = datetime.now(UTC)
    trial_end = trial_start + timedelta(days=TRIAL_DURATION_DAYS)

    if now < trial_end:
        days_left = (trial_end - now).days
        info = LicenseInfo(
            plan="ENTERPRISE",
            is_trial=True,
            trial_days_left=days_left,
            licensed_to="trial",
            expires_at=trial_end,
        )
        _cached_license = info
        _cache_timestamp = time.time()
        return info

    # Trial expired, no valid key
    info = LicenseInfo(
        plan="FREE",
        is_trial=False,
        trial_days_left=0,
        licensed_to="unlicensed",
        expires_at=None,
    )
    _cached_license = info
    _cache_timestamp = time.time()
    return info


def is_tool_available(tool_name: str, license_info: LicenseInfo) -> bool:
    """Check if a tool is available for the current license.

    During trial: all tools available.
    With license: all tools available (plan-based gating via RLMEngine).
    Without license (post-trial): only FREE_TIER_TOOLS.
    """
    if license_info.is_trial:
        return True
    if license_info.plan != "FREE":
        return True  # Licensed users get plan-based gating via RLMEngine
    return tool_name in FREE_TIER_TOOLS
