"""
SQL validation module.

This module provides:
- SQL syntax validation
- Dangerous operation detection
- Column reference checking
"""

from .sql_validator import SQLValidator, ValidationResult

__all__ = [
    "SQLValidator",
    "ValidationResult",
]