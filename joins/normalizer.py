"""
Column name normalization for cross-convention matching.

Handles:
- snake_case (user_id)
- camelCase (userId)
- PascalCase (UserId)
- kebab-case (user-id)
- Mixed conventions
"""
import re
from typing import List, Tuple, Optional


class ColumnNormalizer:
    """
    Normalize column names across different naming conventions.

    This enables matching columns like:
    - assetId ↔ asset_id
    - userId ↔ user_id
    - CreatedByUserId ↔ created_by_user_id
    """

    # Common ID-like suffixes
    ID_SUFFIXES = frozenset({'id', 'uuid', 'guid', 'pk', 'key'})

    # Common FK patterns (suffix -> what it typically references)
    FK_PATTERNS = {
        'id': 'id',
        'uuid': 'uuid',
        'guid': 'guid',
        'key': 'id',
        'ref': 'id',
        'fk': 'id',
    }

    @staticmethod
    def to_parts(name: str) -> List[str]:
        """
        Split a column name into constituent parts.

        Handles camelCase, PascalCase, snake_case, and kebab-case.

        Examples:
            "assetId" -> ["asset", "id"]
            "asset_id" -> ["asset", "id"]
            "AssetID" -> ["asset", "id"]
            "user-name" -> ["user", "name"]
            "createdByUserId" -> ["created", "by", "user", "id"]
            "HTMLParser" -> ["html", "parser"]

        Args:
            name: Column name to split

        Returns:
            List of lowercase parts
        """
        if not name:
            return []

        # Handle kebab-case by replacing with underscore
        name = name.replace("-", "_")

        # Handle camelCase and PascalCase
        # Insert underscore before uppercase letters (but not consecutive ones)
        # First, handle transitions like "userId" -> "user_Id"
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)

        # Handle consecutive uppercase like "HTMLParser" -> "HTML_Parser"
        # or "userID" -> "user_ID"
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)

        # Split on underscores and lowercase
        parts = [p.lower() for p in name.split('_') if p]

        return parts

    @staticmethod
    def to_snake_case(name: str) -> str:
        """
        Convert any naming convention to snake_case.

        Examples:
            "assetId" -> "asset_id"
            "userId" -> "user_id"
            "CreatedByUserId" -> "created_by_user_id"

        Args:
            name: Column name to convert

        Returns:
            snake_case version
        """
        parts = ColumnNormalizer.to_parts(name)
        return "_".join(parts)

    @staticmethod
    def to_camel_case(name: str) -> str:
        """
        Convert any naming convention to camelCase.

        Examples:
            "asset_id" -> "assetId"
            "created_by_user_id" -> "createdByUserId"

        Args:
            name: Column name to convert

        Returns:
            camelCase version
        """
        parts = ColumnNormalizer.to_parts(name)
        if not parts:
            return name
        return parts[0] + "".join(p.capitalize() for p in parts[1:])

    @staticmethod
    def normalize_for_comparison(name: str) -> str:
        """
        Create a normalized form for comparison.

        Removes all case and separator differences by joining parts.

        Examples:
            "assetId" -> "assetid"
            "asset_id" -> "assetid"
            "Asset_ID" -> "assetid"

        Args:
            name: Column name to normalize

        Returns:
            Normalized string for comparison
        """
        return "".join(ColumnNormalizer.to_parts(name))

    @staticmethod
    def is_id_column(name: str) -> bool:
        """
        Check if column appears to be an ID/key column.

        Args:
            name: Column name to check

        Returns:
            True if column is likely an ID/key column
        """
        parts = ColumnNormalizer.to_parts(name)
        normalized = ColumnNormalizer.to_snake_case(name)

        # Direct ID columns
        if normalized in ("id", "uuid", "guid", "pk", "key"):
            return True

        # Ends with id/uuid/etc.
        if parts and parts[-1] in ColumnNormalizer.ID_SUFFIXES:
            return True

        return False

    @staticmethod
    def extract_entity_from_fk(name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the entity name and key type from a foreign key column.

        Examples:
            "assetId" -> ("asset", "id")
            "user_uuid" -> ("user", "uuid")
            "created_by_id" -> ("created_by", "id")
            "userId" -> ("user", "id")
            "customer_fk" -> ("customer", "fk")

        Args:
            name: Column name to analyze

        Returns:
            Tuple of (entity_name, key_type) or (None, None) if not a FK pattern
        """
        parts = ColumnNormalizer.to_parts(name)

        if not parts or len(parts) < 2:
            return (None, None)

        # Check if last part is an ID indicator
        if parts[-1] in ColumnNormalizer.ID_SUFFIXES:
            entity = "_".join(parts[:-1])
            key_type = parts[-1]
            return (entity, key_type)

        # Check for FK patterns
        if parts[-1] in ColumnNormalizer.FK_PATTERNS:
            entity = "_".join(parts[:-1])
            key_type = ColumnNormalizer.FK_PATTERNS[parts[-1]]
            return (entity, key_type)

        return (None, None)

    @staticmethod
    def get_potential_fk_names(entity: str, key_type: str = "id") -> List[str]:
        """
        Generate potential FK column names for an entity.

        Given an entity like "user", generates variations:
        - user_id (snake_case)
        - userId (camelCase)
        - UserId (PascalCase)
        - user_uuid (if key_type is uuid)

        Args:
            entity: Entity name (e.g., "user", "asset")
            key_type: Type of key ("id", "uuid", etc.)

        Returns:
            List of potential FK column names
        """
        variations = []

        # snake_case
        variations.append(f"{entity}_{key_type}")

        # camelCase
        variations.append(f"{entity}{key_type.capitalize()}")

        # Handle entity with underscores -> camelCase
        if "_" in entity:
            camel_entity = ColumnNormalizer.to_camel_case(entity)
            variations.append(f"{camel_entity}{key_type.capitalize()}")

        # Uppercase ID variant (userId, userID)
        if key_type.lower() == "id":
            variations.append(f"{entity}ID")
            if "_" in entity:
                camel_entity = ColumnNormalizer.to_camel_case(entity)
                variations.append(f"{camel_entity}ID")

        return variations

    @staticmethod
    def columns_match(col1: str, col2: str) -> bool:
        """
        Check if two column names match after normalization.

        Args:
            col1: First column name
            col2: Second column name

        Returns:
            True if columns match after normalization
        """
        return (ColumnNormalizer.normalize_for_comparison(col1) ==
                ColumnNormalizer.normalize_for_comparison(col2))