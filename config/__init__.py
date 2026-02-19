"""
Configuration module for the superset-mcp project.
"""

from .semantic_config_manager import SemanticConfigManager, get_config_manager

# Import app configuration from the root config module
# We need to import from the actual file path to avoid circular imports
import importlib.util
import os

_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
_spec = importlib.util.spec_from_file_location("app_config", _config_path)
_app_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app_config)

# Re-export app configuration
load_config = _app_config.load_config
ConfigurationError = _app_config.ConfigurationError
LLMProvider = _app_config.LLMProvider
LLMConfig = _app_config.LLMConfig
SupersetConfig = _app_config.SupersetConfig
AppConfig = _app_config.AppConfig
get_superset_config = _app_config.get_superset_config
get_llm_config = _app_config.get_llm_config
is_interactive = _app_config.is_interactive
is_read_only = _app_config.is_read_only

__all__ = [
    # Semantic config
    "SemanticConfigManager",
    "get_config_manager",
    # App config
    "load_config",
    "ConfigurationError",
    "LLMProvider",
    "LLMConfig",
    "SupersetConfig",
    "AppConfig",
    "get_superset_config",
    "get_llm_config",
    "is_interactive",
    "is_read_only",
]