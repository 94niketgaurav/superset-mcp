"""
Unified LLM service for SQL generation with multiple provider support.

Supports:
- OpenAI (GPT-4, GPT-4o, GPT-5.2)
- Anthropic Claude (Sonnet 4.5, Opus)
- Google Gemini (for review/critique)
"""
import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

# Import centralized config from config package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from config package (not config.py directly to avoid name conflict)
from config import LLMProvider, LLMConfig, get_llm_config


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    sql: str
    explanation: str
    assumptions: List[str]
    suggested_title: str
    confidence: float
    provider: str
    model: str


class LLMService:
    """
    Unified LLM service for SQL generation and query understanding.

    Supports switching between OpenAI and Claude for comparison.
    """

    SQL_GENERATION_SYSTEM_PROMPT = """You are an expert SQL query generator for Apache Superset.
Your task is to convert natural language queries into SIMPLE, clean SQL.

CRITICAL RULES FOR SIMPLE SQL:
1. For single-table queries, write SIMPLE SQL like: SELECT email, name, created_at FROM users WHERE tenant = 'x' LIMIT 1000
2. DO NOT use table prefixes for single tables (wrong: user_details.email, right: email)
3. DO NOT use AS aliases for tables in single-table queries (wrong: FROM users AS u, right: FROM users)
4. DO NOT use AS aliases for columns unless renaming (wrong: email AS email, right: email)
5. Only use table aliases (t1, t2) when JOINing multiple tables
6. Use table names WITHOUT schema prefix - schema is set separately in Superset
7. Add LIMIT clause for safety (default 1000 unless specified)
8. For date/time filters, use proper SQL functions

SCHEMA AWARENESS:
- Known schema names (like 'jeeves', 'dexit', 'public') are DATABASE SCHEMAS, NOT filter values
- When user says "in jeeves" or "from jeeves", they mean the DATABASE SCHEMA, not a WHERE clause filter
- Do NOT add WHERE clauses like "tenant = 'jeeves'" when 'jeeves' is a schema name
- The schema is configured separately in Superset, so just use the table name

DATE/TIME FILTER HANDLING:
- If user requests time filtering (e.g., "last month", "yesterday", "last week"):
  - Check if the table has a date/time column (created_at, updated_at, timestamp, date, etc.)
  - If a date column exists, use it for filtering
  - If NO date column exists, note this in assumptions and ask user to clarify

DIALECT-SPECIFIC SYNTAX (CRITICAL - check DATABASE ENGINE in prompt):
For StarRocks:
  - INTERVAL syntax: INTERVAL 7 DAY (NOT INTERVAL '7 days')
  - DATE_SUB(NOW(), INTERVAL 7 DAY) for "last 7 days"
  - DATE_SUB(CURDATE(), INTERVAL 1 MONTH) for "last month"
  - Use CURDATE() or NOW() for current date/time
  - String concat: CONCAT(a, b) not a || b
For PostgreSQL:
  - INTERVAL syntax: INTERVAL '7 days' or '7 days'::interval
  - NOW() - INTERVAL '7 days' for "last 7 days"
  - Use CURRENT_DATE or NOW() for current date/time
  - String concat: a || b or CONCAT(a, b)

OUTPUT FORMAT (JSON):
{
    "sql": "SELECT ... FROM ...",
    "explanation": "Brief explanation of what this query does",
    "assumptions": ["List of assumptions made"],
    "suggested_title": "A descriptive title for this query",
    "confidence": 0.95,
    "needs_clarification": false,
    "clarification_question": null
}

Always respond with valid JSON only."""

    QUERY_EXPLANATION_PROMPT = """You are an expert SQL analyst. Analyze the given SQL query and provide a comprehensive explanation.

Your analysis should include:
1. PLAIN ENGLISH EXPLANATION: What does this query do in simple terms?
2. TABLE ANALYSIS: Which tables are used and how they relate
3. FILTER ANALYSIS: What filters are applied and what data is included/excluded
4. AGGREGATION ANALYSIS: Any grouping, counting, summing operations
5. PERFORMANCE CONSIDERATIONS: Potential bottlenecks or optimization opportunities
6. OPTIMIZATION SUGGESTIONS: Specific ways to improve the query

Consider the DATABASE ENGINE when suggesting optimizations (StarRocks vs PostgreSQL have different best practices).

OUTPUT FORMAT (JSON):
{
    "summary": "One-line summary of what query does",
    "plain_english": "Detailed explanation in plain English that a non-technical user can understand",
    "tables_used": ["list of tables"],
    "filters": [{"column": "col", "condition": "description"}],
    "aggregations": ["description of aggregations"],
    "joins": [{"type": "LEFT/INNER", "tables": "table1 -> table2", "on": "condition"}],
    "performance_notes": ["potential performance issues"],
    "optimizations": [
        {"issue": "problem description", "suggestion": "how to fix", "improved_sql": "optional improved SQL snippet"}
    ],
    "optimized_sql": "full optimized version of the query if improvements suggested",
    "dialect_issues": ["any syntax issues for the specified database engine"]
}

Always respond with valid JSON only."""

    CHART_SUGGESTION_PROMPT = """Based on the SQL query and its results structure, suggest the best chart type.

Consider:
- Single value: big_number or big_number_total
- Time series data: line, area, or bar (time-based)
- Categorical comparisons: bar, pie, or donut
- Geographic data: map visualizations
- Multiple dimensions: heatmap or treemap
- Distribution: histogram

OUTPUT FORMAT (JSON):
{
    "recommended_type": "bar",
    "alternatives": ["pie", "table"],
    "reason": "Why this chart type is best",
    "x_axis_column": "column_name",
    "y_axis_column": "column_name",
    "group_by_column": null
}"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM service."""
        self.config = config or self._load_config_from_env()
        self._openai_client = None
        self._claude_client = None
        self._gemini_client = None

    def _load_config_from_env(self) -> LLMConfig:
        """Load LLM configuration from environment variables."""
        # Use dataclass defaults
        defaults = LLMConfig()

        # Check for environment override of default provider
        default_provider_str = os.environ.get("LLM_PROVIDER", "").lower()
        if default_provider_str == "openai":
            default_provider = LLMProvider.OPENAI
        elif default_provider_str == "claude":
            default_provider = LLMProvider.CLAUDE
        elif default_provider_str == "gemini":
            default_provider = LLMProvider.GEMINI
        else:
            # Use dataclass default (CLAUDE)
            default_provider = defaults.default_provider
        return LLMConfig(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_model=os.environ.get("OPENAI_MODEL") or defaults.openai_model,
            claude_api_key=os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"),
            claude_model=os.environ.get("CLAUDE_MODEL") or defaults.claude_model,
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            gemini_model=os.environ.get("GEMINI_MODEL") or defaults.gemini_model,
            default_provider=default_provider
        )

    def _get_openai_client(self):
        """Get or create OpenAI client."""
        if self._openai_client is None:
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
        return self._openai_client

    def _get_claude_client(self):
        """Get or create Claude client."""
        if self._claude_client is None:
            if not self.config.claude_api_key:
                raise ValueError("Claude API key not configured. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY")
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=self.config.claude_api_key)
        return self._claude_client

    def _get_gemini_client(self):
        """Get or create Gemini client."""
        if self._gemini_client is None:
            if not self.config.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            from google import genai
            self._gemini_client = genai.Client(api_key=self.config.gemini_api_key)
        return self._gemini_client

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available (configured) LLM providers."""
        providers = []

        if self.config.openai_api_key:
            providers.append({
                "id": "openai",
                "name": f"OpenAI {self.config.openai_model}",
                "model": self.config.openai_model,
                "available": True
            })

        if self.config.claude_api_key:
            providers.append({
                "id": "claude",
                "name": "Claude Sonnet",
                "model": self.config.claude_model,
                "available": True
            })

        if self.config.gemini_api_key:
            providers.append({
                "id": "gemini",
                "name": "Google Gemini",
                "model": self.config.gemini_model,
                "available": True
            })

        return providers

    def generate_sql(
        self,
        query: str,
        context: Dict[str, Any],
        provider: Optional[LLMProvider] = None,
        examples: Optional[List[Dict]] = None,
        learned_patterns: Optional[str] = None
    ) -> SQLGenerationResult:
        """
        Generate SQL from natural language query.

        Args:
            query: Natural language query
            context: Schema context (tables, columns, joins)
            provider: LLM provider to use (default: from config)
            examples: Optional few-shot examples from similar queries
            learned_patterns: Optional description of learned patterns for these tables

        Returns:
            SQLGenerationResult with generated SQL and metadata
        """
        provider = provider or self.config.default_provider

        # Build the user prompt with context and examples
        user_prompt = self._build_sql_prompt(query, context, examples, learned_patterns)

        if provider == LLMProvider.OPENAI:
            return self._generate_sql_openai(query, user_prompt)
        elif provider == LLMProvider.CLAUDE:
            return self._generate_sql_claude(query, user_prompt)
        elif provider == LLMProvider.GEMINI:
            return self._generate_sql_gemini(query, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _build_sql_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        examples: Optional[List[Dict]] = None,
        learned_patterns: Optional[str] = None
    ) -> str:
        """Build the SQL generation prompt with context and few-shot examples.

        Args:
            query: User's natural language query
            context: Schema context (tables, joins, etc.)
            examples: Optional list of similar query examples for few-shot learning
            learned_patterns: Optional string describing learned patterns for these tables
        """
        tables_info = []
        date_columns_found = []
        for table in context.get("tables", []):
            cols = table.get("columns", [])[:20]  # Limit columns shown
            cols_str = ", ".join(cols)
            schema = table.get('schema', 'public')
            table_name = table['table_name']
            # Show schema as metadata, but emphasize table name is what goes in SQL
            tables_info.append(f"- {table_name} (schema: {schema}): {cols_str}")
            # Track date columns for time filter support
            for col in cols:
                col_lower = col.lower()
                if any(dt in col_lower for dt in ['date', 'time', 'created', 'updated', 'timestamp']):
                    date_columns_found.append(f"{table_name}.{col}")

        joins_info = context.get("available_joins", [])
        joins_str = "\n".join(joins_info) if joins_info else "None specified"

        # Get known schemas from context
        known_schemas = context.get("known_schemas", [])
        schemas_str = ", ".join(known_schemas) if known_schemas else "public"

        # Get extracted schema from context (if user mentioned a schema)
        extracted_schema = context.get("extracted_schema")
        schema_note = ""
        if extracted_schema:
            schema_note = f"\nNOTE: User mentioned '{extracted_schema}' - this is a SCHEMA NAME, not a filter value. Do NOT use it in WHERE clauses."

        # Date columns info
        date_cols_str = ", ".join(date_columns_found) if date_columns_found else "NONE FOUND"

        # State filters info (e.g., "not completed", "pending")
        state_filters = context.get("state_filters", [])
        state_filters_str = ""
        if state_filters:
            state_info_parts = []
            for sf in state_filters:
                state = sf.get('state', '')
                negated = sf.get('negated', False)
                target = sf.get('target_entity', '')
                op = "IS NOT" if negated else "IS"
                target_note = f" (applies to: {target})" if target else ""
                state_info_parts.append(f"- {state} {op} TRUE{target_note}")
            state_filters_str = f"""
STATE/STATUS FILTERS DETECTED:
{chr(10).join(state_info_parts)}
Convert these to appropriate WHERE clauses based on the TABLE being used:
- For 'Assignment Usage' table: pending = completion_date IS NULL, completed = completion_date IS NOT NULL
- For 'assignmentcompleted' table: presence in table = completed
- For other tables: use status column or similar (e.g., status='pending' or is_completed=0)
"""

        # Build few-shot examples section
        examples_section = ""
        if examples:
            examples_section = "\n\nSIMILAR SUCCESSFUL QUERIES (use as reference):\n"
            for i, ex in enumerate(examples[:3], 1):  # Limit to 3 examples
                ex_title = ex.get("title", "Query")
                ex_sql = ex.get("sql", "")
                ex_reasons = ex.get("match_reasons", [])
                reason_str = f" (relevant: {', '.join(ex_reasons[:2])})" if ex_reasons else ""
                examples_section += f"""
Example {i}: {ex_title}{reason_str}
```sql
{ex_sql}
```
"""

        # Add learned patterns section
        patterns_section = ""
        if learned_patterns:
            patterns_section = f"\n{learned_patterns}\n"

        return f"""DATABASE ENGINE: {context.get('engine', 'postgresql')}

KNOWN SCHEMA NAMES: {schemas_str}
These are database schemas - if mentioned in query, they indicate which schema to use, NOT filter values.{schema_note}
{patterns_section}
AVAILABLE TABLES:
{chr(10).join(tables_info)}

DATE/TIME COLUMNS AVAILABLE: {date_cols_str}
Use these for time-based filters (last month, yesterday, etc.)
{state_filters_str}{examples_section}
AVAILABLE JOINS:
{joins_str}

USER QUERY: {query}

Generate SIMPLE SQL. For single-table queries, write like this example:
SELECT email, tenant, username, first_name FROM user_details WHERE tenant = 'lcmc' LIMIT 1000

IMPORTANT:
- DO NOT use table prefixes (wrong: user_details.email) or aliases (wrong: FROM user_details AS ud)
- Schema names like '{schemas_str}' are NOT filter values - don't add WHERE clauses for them
- If time filter requested but no date column exists, set needs_clarification=true
- For state filters like 'not completed', look for status/state columns and add appropriate conditions
- Use the SIMILAR SUCCESSFUL QUERIES above as reference for correct SQL patterns
Return JSON only."""

    def _generate_sql_openai(self, query: str, user_prompt: str) -> SQLGenerationResult:
        """Generate SQL using OpenAI."""
        client = self._get_openai_client()

        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": self.SQL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return SQLGenerationResult(
            sql=result.get("sql", ""),
            explanation=result.get("explanation", ""),
            assumptions=result.get("assumptions", []),
            suggested_title=result.get("suggested_title", f"Query: {query[:50]}"),
            confidence=result.get("confidence", 0.8),
            provider="openai",
            model=self.config.openai_model
        )

    def _generate_sql_claude(self, query: str, user_prompt: str) -> SQLGenerationResult:
        """Generate SQL using Claude."""
        client = self._get_claude_client()

        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=4096,
            system=self.SQL_GENERATION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Parse the response - Claude returns text, we need to extract JSON
        content = response.content[0].text

        # Try to parse as JSON
        try:
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                # Fallback: treat entire response as SQL
                result = {
                    "sql": content,
                    "explanation": "Generated by Claude",
                    "assumptions": [],
                    "suggested_title": f"Query: {query[:50]}",
                    "confidence": 0.7
                }
        except json.JSONDecodeError:
            result = {
                "sql": content,
                "explanation": "Generated by Claude",
                "assumptions": [],
                "suggested_title": f"Query: {query[:50]}",
                "confidence": 0.7
            }

        return SQLGenerationResult(
            sql=result.get("sql", ""),
            explanation=result.get("explanation", ""),
            assumptions=result.get("assumptions", []),
            suggested_title=result.get("suggested_title", f"Query: {query[:50]}"),
            confidence=result.get("confidence", 0.8),
            provider="claude",
            model=self.config.claude_model
        )

    def _generate_sql_gemini(self, query: str, user_prompt: str) -> SQLGenerationResult:
        """Generate SQL using Gemini."""
        client = self._get_gemini_client()

        full_prompt = f"{self.SQL_GENERATION_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = client.models.generate_content(
            model=self.config.gemini_model,
            contents=full_prompt
        )

        content = response.text

        # Try to parse as JSON
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {
                    "sql": content,
                    "explanation": "Generated by Gemini",
                    "assumptions": [],
                    "suggested_title": f"Query: {query[:50]}",
                    "confidence": 0.7
                }
        except json.JSONDecodeError:
            result = {
                "sql": content,
                "explanation": "Generated by Gemini",
                "assumptions": [],
                "suggested_title": f"Query: {query[:50]}",
                "confidence": 0.7
            }

        return SQLGenerationResult(
            sql=result.get("sql", ""),
            explanation=result.get("explanation", ""),
            assumptions=result.get("assumptions", []),
            suggested_title=result.get("suggested_title", f"Query: {query[:50]}"),
            confidence=result.get("confidence", 0.8),
            provider="gemini",
            model=self.config.gemini_model
        )

    def explain_query(
        self,
        sql: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Explain a SQL query in plain English with analysis and optimization suggestions.

        Args:
            sql: The SQL query to explain
            context: Optional context with schema info, engine type, etc.
            provider: LLM provider to use

        Returns:
            Dict with explanation, analysis, and optimization suggestions
        """
        provider = provider or self.config.default_provider
        context = context or {}

        engine = context.get("engine", "starrocks")
        tables_info = context.get("tables", [])

        # Build schema context if available
        schema_context = ""
        if tables_info:
            schema_context = "\n\nAVAILABLE SCHEMA CONTEXT:\n"
            for table in tables_info[:5]:  # Limit to 5 tables
                schema_context += f"\nTable: {table.get('name', 'unknown')}\n"
                columns = table.get("columns", [])[:10]  # Limit columns
                if columns:
                    schema_context += f"Columns: {', '.join(columns)}\n"

        prompt = f"""DATABASE ENGINE: {engine}

SQL Query to analyze:
```sql
{sql}
```
{schema_context}

Analyze this query and provide a comprehensive explanation with optimization suggestions."""

        try:
            if provider == LLMProvider.OPENAI:
                client = self._get_openai_client()
                response = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": self.QUERY_EXPLANATION_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)

            elif provider == LLMProvider.CLAUDE:
                client = self._get_claude_client()
                response = client.messages.create(
                    model=self.config.claude_model,
                    max_tokens=4096,
                    system=self.QUERY_EXPLANATION_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    result = {"summary": content, "plain_english": content}

            elif provider == LLMProvider.GEMINI:
                client = self._get_gemini_client()
                full_prompt = f"{self.QUERY_EXPLANATION_PROMPT}\n\n{prompt}"
                response = client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=full_prompt
                )
                content = response.text
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    result = {"summary": content, "plain_english": content}
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return {
                "success": True,
                "summary": result.get("summary", ""),
                "plain_english": result.get("plain_english", ""),
                "tables_used": result.get("tables_used", []),
                "filters": result.get("filters", []),
                "aggregations": result.get("aggregations", []),
                "joins": result.get("joins", []),
                "performance_notes": result.get("performance_notes", []),
                "optimizations": result.get("optimizations", []),
                "optimized_sql": result.get("optimized_sql"),
                "dialect_issues": result.get("dialect_issues", []),
                "provider": provider.value if hasattr(provider, 'value') else str(provider),
                "model": getattr(self.config, f"{provider.value if hasattr(provider, 'value') else provider}_model", "unknown")
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": "Failed to analyze query",
                "plain_english": f"Could not analyze the query: {str(e)}"
            }

    def suggest_chart_type(
        self,
        sql: str,
        query: str,
        result_columns: List[str],
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Suggest the best chart type for a query result.

        Args:
            sql: The SQL query
            query: Original natural language query
            result_columns: List of column names in the result
            provider: LLM provider to use

        Returns:
            Chart type suggestion with reasoning
        """
        provider = provider or self.config.default_provider

        prompt = f"""SQL Query:
```sql
{sql}
```

Original Request: {query}

Result Columns: {', '.join(result_columns)}

{self.CHART_SUGGESTION_PROMPT}"""

        if provider == LLMProvider.OPENAI:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        elif provider == LLMProvider.CLAUDE:
            client = self._get_claude_client()
            response = client.messages.create(
                model=self.config.claude_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                return json.loads(content[start:end])
            except:
                return {"recommended_type": "table", "reason": "Default fallback"}

        else:
            return {"recommended_type": "table", "reason": "Provider not supported for chart suggestions"}

    def review_sql(
        self,
        sql: str,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review and potentially improve SQL using Gemini.

        Args:
            sql: SQL to review
            query: Original query
            context: Schema context

        Returns:
            Review result with suggestions
        """
        if not self.config.gemini_api_key:
            return {"reviewed": False, "reason": "Gemini not configured"}

        prompt = f"""Review this SQL query for correctness and optimization:

Original Request: {query}

SQL:
```sql
{sql}
```

Schema Context:
{json.dumps(context.get('tables', []), indent=2)}

Provide feedback in JSON format:
{{
    "is_correct": true/false,
    "issues": ["list of issues if any"],
    "suggestions": ["optimization suggestions"],
    "improved_sql": "improved SQL if changes needed, or null"
}}"""

        try:
            client = self._get_gemini_client()
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt
            )

            content = response.text
            start = content.find('{')
            end = content.rfind('}') + 1
            return json.loads(content[start:end])
        except Exception as e:
            return {"reviewed": False, "error": str(e)}


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service