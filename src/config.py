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

    # Redis (for rate limiting)
    redis_url: str = "redis://localhost:6379"

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
