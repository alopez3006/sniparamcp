"""Configuration module for RLM MCP Server."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str

    # Redis (for rate limiting) - optional, rate limiting disabled if not set
    redis_url: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS - comma-separated list of allowed origins
    # In production, this MUST be set explicitly (not "*")
    cors_allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Rate limiting
    rate_limit_requests: int = 100  # requests per minute per API key
    rate_limit_window: int = 60  # seconds

    # Demo key rate limiting (stricter, public key)
    demo_api_key_ids: str = ""  # comma-separated DB IDs of demo API keys
    demo_rate_limit_requests: int = 5  # requests per minute per IP for demo keys
    demo_rate_limit_window: int = 60  # seconds

    # IP-based rate limiting (secondary layer)
    ip_rate_limit_requests: int = 300  # requests per window per IP
    ip_rate_limit_window: int = 60  # seconds

    # Fail-closed: reject requests when Redis is unavailable in production
    rate_limit_fail_closed: bool = False

    # Security: Max JSON payload size for SSE params (in bytes)
    max_json_payload_size: int = 102400  # 100KB

    # Security: Regex execution timeout (in seconds)
    regex_timeout: float = 1.0

    # Security: Max regex pattern length
    max_regex_pattern_length: int = 500

    # Plan limits (queries per month)
    plan_limits: dict[str, int] = {
        "FREE": 100,
        "PRO": 5000,
        "TEAM": 20000,
        "ENTERPRISE": -1,  # unlimited
    }

    # Self-hosted license key (optional)
    snipara_license_key: str = ""

    # Sentry error tracking (optional)
    sentry_dsn: str = ""

    # Environment name for Sentry
    environment: str = "development"

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if self.cors_allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
