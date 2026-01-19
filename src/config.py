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

    # Rate limiting
    rate_limit_requests: int = 100  # requests per minute per API key
    rate_limit_window: int = 60  # seconds

    # Plan limits (queries per month)
    plan_limits: dict[str, int] = {
        "FREE": 100,
        "PRO": 5000,
        "TEAM": 20000,
        "ENTERPRISE": -1,  # unlimited
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
