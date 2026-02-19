"""
Centralized configuration with secure credential handling.
All secrets must be provided via environment variables.
"""
import os
from enum import Enum
from typing import Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class SupersetConfig:
    """Superset connection configuration."""
    url: str
    username: str
    password: str
    provider: str
    read_only: bool
    allowed_database_ids: Set[int]
    relationships_path: str


@dataclass
class LLMConfig:
    """LLM API configuration - unified for all providers."""
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5.2"
    # Claude (Anthropic)
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-5-20250514"
    # Gemini (Google)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    # Default provider
    default_provider: LLMProvider = LLMProvider.OPENAI

    def get_available_providers(self) -> list:
        """Get list of providers with configured API keys."""
        available = []
        if self.openai_api_key:
            available.append(LLMProvider.OPENAI)
        if self.claude_api_key:
            available.append(LLMProvider.CLAUDE)
        if self.gemini_api_key:
            available.append(LLMProvider.GEMINI)
        return available


@dataclass
class AppConfig:
    """Main application configuration."""
    superset: SupersetConfig
    llm: LLMConfig
    interactive_mode: bool
    cache_ttl_seconds: int


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


def _parse_database_ids(raw: str) -> Set[int]:
    """Parse comma-separated database IDs."""
    if not raw or not raw.strip():
        return set()
    return {int(x) for x in raw.split(",") if x.strip().isdigit()}


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """
    Load configuration from environment variables.

    Required environment variables:
    - SUPERSET_URL: Base URL for Superset instance
    - SUPERSET_USERNAME: Superset username
    - SUPERSET_PASSWORD: Superset password

    Optional environment variables:
    - SUPERSET_PROVIDER: Auth provider (default: "db")
    - READ_ONLY: Block write operations (default: "true")
    - ALLOWED_DATABASE_IDS: Comma-separated list of allowed DB IDs
    - RELATIONSHIPS_PATH: Path to relationships.yaml (default: "relationships.yaml")
    - OPENAI_API_KEY: OpenAI API key (required for SQL generation)
    - OPENAI_MODEL: OpenAI model name (default: "gpt-4")
    - GEMINI_API_KEY: Gemini API key (optional, for SQL critique)
    - GEMINI_MODEL: Gemini model name (default: "gemini-2.0-flash")
    - INTERACTIVE_MODE: Enable interactive prompts (default: "true")
    - CACHE_TTL_SECONDS: Dataset cache TTL (default: "300")

    Raises:
        ConfigurationError: If required variables are missing
    """
    # Required credentials - fail fast if missing
    superset_url = os.environ.get("SUPERSET_URL", "").rstrip("/")
    superset_username = os.environ.get("SUPERSET_USERNAME", "")
    superset_password = os.environ.get("SUPERSET_PASSWORD", "")

    missing = []
    if not superset_url:
        missing.append("SUPERSET_URL")
    if not superset_username:
        missing.append("SUPERSET_USERNAME")
    if not superset_password:
        missing.append("SUPERSET_PASSWORD")

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please set them in your environment or .env file."
        )

    # LLM keys - at least one should be set for full functionality
    openai_key = os.environ.get("OPENAI_API_KEY")
    claude_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not openai_key and not claude_key and not gemini_key:
        import warnings
        warnings.warn(
            "No LLM API key is set (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY). "
            "SQL generation features will be limited.",
            UserWarning
        )

    # Determine default provider based on available keys
    default_provider_str = os.environ.get("DEFAULT_LLM_PROVIDER", "openai").lower()
    try:
        default_provider = LLMProvider(default_provider_str)
    except ValueError:
        default_provider = LLMProvider.OPENAI

    return AppConfig(
        superset=SupersetConfig(
            url=superset_url,
            username=superset_username,
            password=superset_password,
            provider=os.environ.get("SUPERSET_PROVIDER", "db"),
            read_only=os.environ.get("READ_ONLY", "true").lower() == "true",
            allowed_database_ids=_parse_database_ids(
                os.environ.get("ALLOWED_DATABASE_IDS", "")
            ),
            relationships_path=os.environ.get("RELATIONSHIPS_PATH", "relationships.yaml"),
        ),
        llm=LLMConfig(
            openai_api_key=openai_key,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
            claude_api_key=claude_key,
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250514"),
            gemini_api_key=gemini_key,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            default_provider=default_provider,
        ),
        interactive_mode=os.environ.get("INTERACTIVE_MODE", "true").lower() == "true",
        cache_ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "300")),
    )


def get_superset_config() -> SupersetConfig:
    """Get Superset configuration."""
    return load_config().superset


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return load_config().llm


def is_interactive() -> bool:
    """Check if interactive mode is enabled."""
    return load_config().interactive_mode


def is_read_only() -> bool:
    """Check if read-only mode is enabled."""
    return load_config().superset.read_only