"""
Chatbot API Server - REST API wrapper for the orchestrator.

This server exposes the orchestrator functionality via HTTP endpoints
so the chatbot widget can interact with it.

Run: python chatbot/api_server.py
"""
import os
import sys
import re
import json
import time
import logging
import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# Get parent directory path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for imports
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dotenv import load_dotenv
# Load .env from parent directory
load_dotenv(os.path.join(PARENT_DIR, '.env'))

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from openai import OpenAI

from config import load_config, ConfigurationError
import mcp_superset_server as mcp
from query_engine import get_query_engine
from llm import get_llm_service, LLMProvider
from nlu.semantic_matcher import SemanticMatcher
from nlu.table_intelligence import get_table_intelligence
from learning import ExampleGenerator, KnowledgeStore
from config.semantic_config_manager import get_config_manager

# Get the directory where this script is located
CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=CHATBOT_DIR)
CORS(app)  # Enable CORS for cross-origin requests from Superset


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (widget.js, etc.)"""
    return send_from_directory(CHATBOT_DIR, filename)


@app.route('/')
def index():
    """Serve the bookmarklet installation page."""
    return send_from_directory(CHATBOT_DIR, 'bookmarklet.html')


@app.route('/install')
def install():
    """Alias for bookmarklet installation page."""
    return send_from_directory(CHATBOT_DIR, 'bookmarklet.html')


# Session storage for conversation state
sessions = {}


@dataclass
class ChatSession:
    """Tracks state for a chat conversation."""
    session_id: str
    current_step: str = "idle"
    query: Optional[str] = None
    tables: list = None
    joins: list = None
    filters: list = None  # WHERE clause filters
    time_filters: list = None  # Time-based filters (last month, etc.)
    state_filters: list = None  # State filters like "not completed", "pending"
    pending_date_alternatives: list = None  # Alternative tables with date columns
    sql: Optional[str] = None
    database_id: Optional[int] = None
    schema: Optional[str] = None  # Database schema for SQL Lab
    extracted_schema: Optional[str] = None  # Schema name extracted from query (e.g., "jeeves")
    sqllab_url: Optional[str] = None
    pending_choices: list = None
    llm_provider: str = None  # Selected LLM provider (None = use default)
    suggested_title: Optional[str] = None
    chart_type: Optional[str] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.joins is None:
            self.joins = []
        if self.filters is None:
            self.filters = []
        if self.state_filters is None:
            self.state_filters = []
        if self.pending_choices is None:
            self.pending_choices = []


def get_session(session_id: str) -> ChatSession:
    """Get or create a session."""
    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id=session_id)
    return sessions[session_id]


def get_mcp_tools():
    """Get MCP tool functions."""
    return {
        "build_execution_plan": mcp.build_execution_plan.fn,
        "find_join_path": mcp.find_join_path.fn,
        "generate_sql_context": mcp.generate_sql_context.fn,
        "validate_sql": mcp.validate_sql.fn,
        "execute_and_save": mcp.execute_and_save.fn,
        "get_sqllab_url": mcp.get_sqllab_url.fn,
        "create_saved_query": mcp.create_saved_query.fn,
        "get_table_info": mcp.get_table_info.fn,
        "get_dataset": mcp.get_dataset.fn,
        "get_server_info": mcp.get_server_info.fn,
    }


def get_llm_config():
    """Get LLM configuration."""
    try:
        config = load_config()
        return config.llm
    except ConfigurationError:
        return None


def _is_sql_query(text: str) -> bool:
    """
    Check if the text appears to be a SQL query.

    Returns True if the text looks like a standalone SQL query.
    """
    text = text.strip()

    # SQL queries typically start with these keywords
    sql_start_keywords = [
        'SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE',
        'CREATE', 'ALTER', 'DROP', 'EXPLAIN'
    ]

    text_upper = text.upper()

    # Check if it starts with a SQL keyword
    starts_with_sql = any(text_upper.startswith(kw) for kw in sql_start_keywords)

    # Check for SQL structure indicators
    has_sql_structure = bool(re.search(r'\bFROM\s+\w+', text, re.IGNORECASE))
    has_select = bool(re.search(r'\bSELECT\b', text, re.IGNORECASE))

    # It's likely SQL if it starts with SQL keyword and has FROM clause
    return starts_with_sql and has_sql_structure and has_select


def _contains_sql(text: str) -> bool:
    """Check if the text contains SQL code (possibly in code blocks)."""
    # Check for SQL in code blocks
    if '```sql' in text.lower() or '```' in text:
        return True

    # Check for SQL keywords that suggest embedded SQL
    sql_patterns = [
        r'\bSELECT\b.*\bFROM\b',
        r'\bWITH\b.*\bAS\b.*\bSELECT\b',
    ]

    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in sql_patterns)


def _extract_sql_from_message(text: str) -> Optional[str]:
    """Extract SQL from a message that may contain explanatory text."""
    # Try to extract from code blocks first
    code_block_match = re.search(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find SELECT...FROM pattern
    sql_match = re.search(r'((?:WITH\b.*?)?\bSELECT\b.*?\bFROM\b.*?)(?:\n\n|$)', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    return None


def explain_sql_query(session: 'ChatSession', sql: str, tools: dict) -> Response:
    """
    Explain a SQL query using LLM.

    Provides:
    - Plain English explanation
    - Table and column analysis
    - Performance notes
    - Optimization suggestions
    """
    try:
        llm_service = get_llm_service()
        if not llm_service:
            return jsonify({
                "response": "LLM service not available for query explanation.",
                "step": "idle",
                "needs_input": True
            })

        # Get schema context for better analysis
        context = {"engine": "starrocks"}  # Default
        try:
            server_info = tools["get_server_info"]()
            context["engine"] = server_info.get("engine_hint", "starrocks")
        except Exception:
            pass

        # Get the LLM provider from session or use default
        provider = None
        if session.llm_provider:
            try:
                provider = LLMProvider(session.llm_provider)
            except ValueError:
                pass

        # Call the explanation service
        result = llm_service.explain_query(sql, context, provider)

        if not result.get("success", False):
            return jsonify({
                "response": f"Failed to analyze query: {result.get('error', 'Unknown error')}",
                "step": "idle",
                "needs_input": True
            })

        # Build response
        messages = []

        # Summary
        if result.get("summary"):
            messages.append(f"**Summary:** {result['summary']}")

        # Plain English explanation
        if result.get("plain_english"):
            messages.append(f"\n**What this query does:**\n{result['plain_english']}")

        # Tables used
        if result.get("tables_used"):
            tables = ", ".join(f"`{t}`" for t in result["tables_used"])
            messages.append(f"\n**Tables used:** {tables}")

        # Joins
        if result.get("joins"):
            messages.append("\n**Joins:**")
            for j in result["joins"]:
                messages.append(f"- {j.get('type', 'JOIN')}: {j.get('tables', '')} ON {j.get('on', '')}")

        # Filters
        if result.get("filters"):
            messages.append("\n**Filters:**")
            for f in result["filters"]:
                messages.append(f"- {f.get('column', '')}: {f.get('condition', '')}")

        # Performance notes
        if result.get("performance_notes"):
            messages.append("\n**Performance considerations:**")
            for note in result["performance_notes"]:
                messages.append(f"- {note}")

        # Dialect issues
        if result.get("dialect_issues"):
            messages.append(f"\n**Dialect issues ({context.get('engine', 'unknown')}):**")
            for issue in result["dialect_issues"]:
                messages.append(f"- ⚠️ {issue}")

        # Optimizations
        if result.get("optimizations"):
            messages.append("\n**Optimization suggestions:**")
            for opt in result["optimizations"]:
                messages.append(f"- **{opt.get('issue', 'Issue')}:** {opt.get('suggestion', '')}")
                if opt.get("improved_sql"):
                    messages.append(f"  ```sql\n  {opt['improved_sql']}\n  ```")

        # Optimized SQL
        optimized_sql = result.get("optimized_sql")
        actions = []
        if optimized_sql and optimized_sql.strip() != sql.strip():
            messages.append(f"\n**Optimized query:**\n```sql\n{optimized_sql}\n```")
            session.sql = optimized_sql
            session.current_step = "sql_ready"
            actions = ["execute", "save_only", "get_url", "cancel"]

        return jsonify({
            "response": "\n".join(messages),
            "step": session.current_step,
            "sql": optimized_sql if optimized_sql else None,
            "original_sql": sql,
            "analysis": result,
            "actions": actions if actions else None,
            "needs_input": True,
            "provider": result.get("provider"),
            "model": result.get("model")
        })

    except ImportError as e:
        return jsonify({
            "response": f"LLM service not available: {str(e)}",
            "step": "idle",
            "needs_input": True
        })
    except Exception as e:
        return jsonify({
            "response": f"Error analyzing query: {str(e)}",
            "step": "idle",
            "needs_input": True
        })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "superset-chatbot"})


@app.route("/api/info", methods=["GET"])
def info():
    """Get server and semantic matcher info."""
    try:
        tools = get_mcp_tools()
        server_info = tools["get_server_info"]()
        return jsonify(server_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/providers", methods=["GET"])
def get_providers():
    """Get available LLM providers."""
    try:
        llm_service = get_llm_service()
        providers = llm_service.get_available_providers()
        # Get actual default from config
        default_provider = llm_service.config.default_provider.value
        return jsonify({
            "providers": providers,
            "default": default_provider
        })
    except Exception as e:
        # Fallback if LLM service not available
        return jsonify({
            "providers": [
                {"id": "openai", "name": "OpenAI GPT-4", "available": True},
                {"id": "claude", "name": "Claude Sonnet", "available": False}
            ],
            "default": "openai",
            "error": str(e)
        })


@app.route("/api/set-provider", methods=["POST"])
def set_provider():
    """Set the LLM provider for a session."""
    data = request.json
    session_id = data.get("session_id")
    provider = data.get("provider", "openai")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    session = get_session(session_id)
    session.llm_provider = provider

    return jsonify({
        "session_id": session_id,
        "provider": provider,
        "message": f"LLM provider set to {provider}"
    })


@app.route("/api/chart-types", methods=["GET"])
def get_chart_types():
    """Get available chart types with descriptions."""
    chart_types = [
        {
            "id": "table",
            "name": "Table",
            "icon": "table",
            "description": "Display data in rows and columns",
            "best_for": "Detailed data inspection"
        },
        {
            "id": "big_number_total",
            "name": "Big Number",
            "icon": "number",
            "description": "Single large metric value",
            "best_for": "KPIs, single metrics"
        },
        {
            "id": "echarts_timeseries_line",
            "name": "Line Chart",
            "icon": "line-chart",
            "description": "Show trends over time",
            "best_for": "Time series, trends"
        },
        {
            "id": "echarts_timeseries_bar",
            "name": "Bar Chart",
            "icon": "bar-chart",
            "description": "Compare values across categories",
            "best_for": "Comparisons, rankings"
        },
        {
            "id": "pie",
            "name": "Pie Chart",
            "icon": "pie-chart",
            "description": "Show proportions",
            "best_for": "Part-to-whole (< 7 items)"
        },
        {
            "id": "echarts_area",
            "name": "Area Chart",
            "icon": "area-chart",
            "description": "Filled line chart",
            "best_for": "Cumulative data"
        },
        {
            "id": "heatmap",
            "name": "Heatmap",
            "icon": "heat-map",
            "description": "Color-coded matrix",
            "best_for": "Correlations, matrices"
        },
        {
            "id": "scatter",
            "name": "Scatter Plot",
            "icon": "scatter-plot",
            "description": "Two-variable relationship",
            "best_for": "Correlation analysis"
        }
    ]
    return jsonify({"chart_types": chart_types})


@app.route("/api/dashboards", methods=["GET"])
def list_dashboards():
    """List available dashboards."""
    try:
        result = mcp.list_dashboards.fn()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "dashboards": []})


@app.route("/api/create-dashboard", methods=["POST"])
def create_dashboard():
    """Create a new dashboard."""
    try:
        data = request.json
        title = data.get("title", "New Dashboard")
        result = mcp.create_dashboard.fn(dashboard_title=title)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Handles user messages and returns bot responses.

    Request body:
    {
        "session_id": "unique-session-id",
        "message": "user message",
        "action": "optional action (confirm, cancel, select)"
        "selection": "optional selection index for choices"
    }

    Response:
    {
        "response": "bot message",
        "step": "current step name",
        "choices": [...] if user needs to make a choice,
        "sql": "generated SQL if available",
        "sqllab_url": "URL to open in SQL Lab",
        "needs_input": true/false
    }
    """
    data = request.json
    session_id = data.get("session_id", "default")
    message = data.get("message", "").strip()
    action = data.get("action")
    selection = data.get("selection")

    session = get_session(session_id)
    tools = get_mcp_tools()

    try:
        # Handle text-based commands (cancel, reset, etc.)
        message_lower = message.lower().strip()
        if message_lower in ["cancel", "reset", "start over", "clear", "nevermind", "never mind"]:
            action = "cancel"

        # Check if user pasted a SQL query and wants explanation
        if _is_sql_query(message) and session.current_step == "idle":
            return explain_sql_query(session, message, tools)

        # Check if user is asking to explain, analyze, or optimize with SQL
        explain_keywords = ["explain", "analyze", "what does", "understand", "optimize", "improve", "help me understand"]
        if any(kw in message_lower for kw in explain_keywords) and _contains_sql(message):
            sql = _extract_sql_from_message(message)
            if sql:
                return explain_sql_query(session, sql, tools)

        # Handle numeric input as selection when in choosing state
        if message.isdigit() and session.current_step in ["choosing_table", "confirming_joins", "choosing_date_source"] and session.pending_choices:
            action = "select"
            selection = int(message)

        # Handle date source selection
        if message.isdigit() and session.current_step == "choosing_date_source":
            action = "select_date_source"
            selection = int(message)

        # Handle actions
        if action == "cancel":
            session.current_step = "idle"
            session.query = None
            session.tables = []
            session.joins = []
            session.filters = []
            session.time_filters = []
            session.sql = None
            session.pending_choices = []
            session.pending_date_alternatives = None
            session.extracted_schema = None
            return jsonify({
                "response": "Cancelled. What else would you like to query?",
                "step": "idle",
                "needs_input": True
            })

        if action == "select" and session.pending_choices:
            # User selected from choices
            try:
                idx = int(selection)
                if 0 <= idx < len(session.pending_choices):
                    selected = session.pending_choices[idx]
                    session.tables.append(selected["table_name"])
                    session.pending_choices = []

                    # Continue to next step
                    return process_after_table_selection(session, tools)
            except (ValueError, IndexError):
                return jsonify({
                    "response": "Invalid selection. Please choose a number from the list.",
                    "step": session.current_step,
                    "choices": session.pending_choices,
                    "needs_input": True
                })

        if action == "confirm_joins":
            # User confirmed joins
            session.joins = session.pending_choices
            session.pending_choices = []
            return generate_sql(session, tools)

        if action == "reject_joins":
            # User rejected joins
            session.joins = []
            session.pending_choices = []
            return generate_sql(session, tools)

        if action == "select_date_source":
            # User selected a date source option
            alternatives = getattr(session, 'pending_date_alternatives', [])
            try:
                idx = int(selection)
                if idx < len(alternatives):
                    # User chose an alternative table with date column
                    alt = alternatives[idx]
                    alt_table = alt['table']
                    date_column = alt['date_column']

                    # Replace or add the table
                    if alt_table.lower() in ['asset_creation_overview', 'asset_creation_overview2']:
                        # Use the pre-built view instead
                        session.tables = [alt_table]
                        session.pending_date_alternatives = None
                        return generate_sql(session, tools)
                    else:
                        # Add as a join (e.g., audithistory)
                        original_table = session.tables[0] if session.tables else 'asset'
                        session.tables.append(alt_table)
                        # Find join path
                        join_result = tools["find_join_path"](session.tables, max_depth=3)
                        session.joins = join_result.get("join_paths", [])
                        session.pending_date_alternatives = None
                        return generate_sql(session, tools)
                else:
                    # User chose to skip time filter
                    session.time_filters = []  # Clear time filters
                    session.pending_date_alternatives = None
                    return generate_sql(session, tools)

            except (ValueError, IndexError):
                return jsonify({
                    "response": "Invalid selection. Please choose a number from the options.",
                    "step": "choosing_date_source",
                    "needs_input": True
                })

        if action == "execute":
            # User wants to execute
            return execute_query(session, tools)

        if action == "save_only":
            # User wants to save without executing
            return save_query_only(session, tools)

        if action == "get_url":
            # User just wants the URL
            return get_url_only(session, tools)

        # Check if this is a modification request while SQL is ready
        if message and session.current_step == "sql_ready" and session.sql:
            modification_keywords = ['add', 'remove', 'change', 'modify', 'filter', 'sort', 'order',
                                     'include', 'exclude', 'limit', 'group', 'where', 'join']
            is_modification = any(kw in message.lower() for kw in modification_keywords)

            if is_modification:
                return modify_sql(session, tools, message)

        # New query - accept at any step (implicitly cancels current flow)
        if message:
            session.query = message
            session.tables = []
            session.joins = []
            session.filters = []
            session.sql = None
            session.pending_choices = []
            session.extracted_schema = None
            session.current_step = "analyzing"

            return analyze_query(session, tools)

        # No message and no action - show help
        return jsonify({
            "response": "I'm ready to help you query your data. What would you like to know?\n\nTry asking something like:\n• \"Show me all users\"\n• \"Get assets with their events\"\n• \"Count users by department\"",
            "step": "idle",
            "needs_input": True
        })

    except Exception as e:
        return jsonify({
            "response": f"Error: {str(e)}",
            "step": "error",
            "needs_input": True
        }), 500


def analyze_query(session: ChatSession, tools: dict) -> Response:
    """Analyze the query and discover tables with smart auto-selection.

    The system automatically selects the best tables based on:
    - Semantic matching (exact, alias, synonym, prefix matches)
    - TableIntelligence (date columns for time filters)
    - Learned patterns from previous queries

    User prompts are minimized - only asked when no suitable tables found.
    """
    # Use auto_select=True for smart, autonomous table selection
    plan = tools["build_execution_plan"](session.query, auto_select=True)

    table_matches = plan.get("table_matches", {})
    messages = []

    # Capture filters from the plan
    session.filters = plan.get("filters", [])
    session.time_filters = plan.get("time_filters", [])
    session.state_filters = plan.get("state_filters", [])
    time_filters = session.time_filters
    state_filters = session.state_filters

    # Capture extracted schema if present (e.g., "in jeeves" means schema, not filter)
    extracted_schema = plan.get("extracted_schema")
    if extracted_schema:
        session.extracted_schema = extracted_schema

    # Smart auto-selection: Use recommended tables directly without prompting
    tables_selected = []
    for entity, match_data in table_matches.items():
        recommended = match_data.get("recommended")
        candidates = match_data.get("candidates", [])

        if recommended:
            # Use the recommended table (already smartly selected by build_execution_plan)
            table_name = recommended.get("table_name")
            if table_name and table_name not in tables_selected:
                session.tables.append(table_name)
                tables_selected.append(table_name)
        elif candidates:
            # Fallback: auto-select top candidate
            top = candidates[0]
            table_name = top.get("table_name")
            if table_name and table_name not in tables_selected:
                session.tables.append(table_name)
                tables_selected.append(table_name)

    # Build informative message about what was selected
    if tables_selected:
        table_list = ', '.join(f'`{t}`' for t in tables_selected)
        messages.append(f"Using tables: {table_list}")

        # Show schema if extracted
        if extracted_schema:
            messages.append(f"Schema: **{extracted_schema}**")

        # Show detected filters in a concise way
        filter_info = []
        if session.filters:
            filter_info.extend([f"{f['column']}='{f['value']}'" for f in session.filters[:3]])
        if time_filters:
            for tf in time_filters[:2]:
                period = tf.get('period', '').replace('_', ' ')
                filter_info.append(f"time: {period}")
        if state_filters:
            for sf in state_filters[:2]:
                state = sf.get('state', '')
                negated = sf.get('negated', False)
                prefix = "NOT " if negated else ""
                filter_info.append(f"{prefix}{state}")
        if filter_info:
            messages.append(f"Filters: {', '.join(filter_info)}")

    # Only ask for user input if NO tables found at all
    if not session.tables:
        # Check if we have any candidates to offer
        all_candidates = []
        for entity, match_data in table_matches.items():
            all_candidates.extend(match_data.get("candidates", []))

        if all_candidates:
            # Offer top candidates as choices
            session.current_step = "choosing_table"
            session.pending_choices = all_candidates[:5]

            return jsonify({
                "response": "I found some possible matches but couldn't determine the best one. Please select:",
                "step": "choosing_table",
                "choices": [
                    {
                        "index": i,
                        "label": f"{c['table_name']} ({c.get('match_reason', 'match')})",
                        "table_name": c["table_name"],
                        "score": c.get("score", 0.5),
                        "reason": c.get("match_reason", "")
                    }
                    for i, c in enumerate(all_candidates[:5])
                ],
                "needs_input": True
            })
        else:
            return jsonify({
                "response": "I couldn't identify any tables from your query. Could you be more specific about what data you want?",
                "step": "idle",
                "needs_input": True
            })

    # All tables resolved, continue to joins and SQL generation
    return process_after_table_selection(session, tools, messages)


def process_after_table_selection(session: ChatSession, tools: dict, messages: list = None) -> Response:
    """Continue processing after table selection with smart auto-selection.

    This function minimizes user prompts by:
    - Auto-selecting best alternative tables for time filtering
    - Auto-confirming joins based on semantic patterns
    - Replacing virtual datasets with their underlying physical tables
    - Only asking for input when truly ambiguous
    """
    if not session.tables:
        return jsonify({
            "response": "I couldn't identify any tables from your query. Could you be more specific?",
            "step": "idle",
            "needs_input": True
        })

    if messages is None:
        messages = [f"Using tables: {', '.join(f'`{t}`' for t in session.tables)}"]

    # Resolve virtual datasets to their physical tables
    matcher = SemanticMatcher()
    resolved_tables, virtual_warnings = matcher.resolve_tables_for_sql(session.tables)

    if virtual_warnings:
        # Replace session tables with resolved physical tables
        original_tables = session.tables.copy()
        session.tables = [t.name for t in resolved_tables]

        # Update messages to show the replacement
        for warning in virtual_warnings:
            messages.append(f"Note: {warning}")

        if session.tables != original_tables:
            messages.append(f"Resolved to physical tables: {', '.join(f'`{t}`' for t in session.tables)}")

    # Smart time filter handling: Auto-select best table with date columns
    time_filters = getattr(session, 'time_filters', [])
    if time_filters and len(session.tables) == 1:
        intelligence = get_table_intelligence()

        table_name = session.tables[0]
        has_dates = intelligence.has_date_columns(table_name)

        if not has_dates:
            # Auto-select the best alternative with date columns
            alternatives = intelligence.find_time_filter_alternatives(table_name)

            if alternatives:
                # Pick the top alternative automatically
                best_alt = alternatives[0]
                if best_alt.date_columns:
                    date_col = best_alt.date_columns[0]

                    # Check if it's a standalone view (like asset_creation_overview)
                    # or needs to be joined
                    alt_table = best_alt.table_name
                    if any(kw in alt_table.lower() for kw in ['overview', 'unified', 'view', 'summary']):
                        # Use this view directly instead of original table
                        session.tables = [alt_table]
                        messages.append(f"Using `{alt_table}` (has `{date_col}` for time filtering)")
                    else:
                        # Add as a join
                        session.tables.append(alt_table)
                        messages.append(f"Joining `{alt_table}` for time filtering via `{date_col}`")

    # Smart join handling: Auto-confirm joins instead of asking
    if len(session.tables) >= 2:
        join_result = tools["find_join_path"](session.tables, max_depth=3)
        join_paths = join_result.get("join_paths", [])

        if join_paths:
            # Auto-confirm joins - don't ask user
            session.joins = join_paths

            # Add join info to messages
            join_info = []
            for j in join_paths:
                join_info.append(f"{j['left_table']} → {j['right_table']}")
            if join_info:
                messages.append(f"Joins: {', '.join(join_info)}")
        else:
            session.joins = []
    else:
        session.joins = []

    # Proceed directly to SQL generation
    return generate_sql(session, tools, messages)


def generate_sql(session: ChatSession, tools: dict, context_messages: list = None) -> Response:
    """Generate SQL using selected LLM provider (OpenAI or Claude).

    Args:
        session: Chat session with query context
        tools: MCP tools
        context_messages: Optional list of context messages to include in response
    """
    session.current_step = "generating_sql"

    # Get context (including extracted schema if user mentioned one like "in jeeves")
    context = tools["generate_sql_context"](
        session.tables,
        session.query,
        session.joins,
        getattr(session, 'extracted_schema', None),
        getattr(session, 'state_filters', None)
    )

    # Get database_id and schema from first table
    if session.tables:
        table_info = tools["get_table_info"](session.tables[0])

        # Prefer extracted_schema if user explicitly mentioned it (e.g., "in jeeves")
        if getattr(session, 'extracted_schema', None):
            session.schema = session.extracted_schema
        else:
            # Otherwise get schema from table_info
            session.schema = table_info.get("schema")

        # First try to get database_id directly from table_info (for physical tables)
        if table_info.get("database_id"):
            session.database_id = table_info.get("database_id")
        else:
            # Fall back to fetching from Superset API (for registered datasets)
            dataset_id = table_info.get("dataset_id")
            if dataset_id:
                try:
                    dataset_info = tools["get_dataset"](dataset_id)
                    session.database_id = dataset_info.get("database_id", 1)
                    if not session.schema:
                        session.schema = dataset_info.get("schema")
                except Exception:
                    session.database_id = 1
            else:
                session.database_id = 1

    # Try using the new unified LLM service
    try:
        llm_service = get_llm_service()

        # Map provider string to enum
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "claude": LLMProvider.CLAUDE,
            "gemini": LLMProvider.GEMINI
        }
        # Use session provider if set, otherwise use LLM service default
        if session.llm_provider:
            provider = provider_map.get(session.llm_provider, llm_service.config.default_provider)
        else:
            provider = llm_service.config.default_provider

        # Add filter instructions to context
        if session.filters:
            filter_clauses = [f"{f['column']} = '{f['value']}'" for f in session.filters]
            context["filter_instructions"] = filter_clauses

        # Get few-shot examples for improved SQL generation
        examples = []
        learned_patterns = None
        try:
            knowledge_store = KnowledgeStore()
            config_manager = get_config_manager()
            semantic_config = config_manager.config if config_manager else {}

            example_gen = ExampleGenerator(knowledge_store, semantic_config)

            # Build query intent from session data
            query_intent = {
                "entities": [],
                "aggregations": [],
                "time_filters": session.time_filters or [],
                "state_filters": session.state_filters or [],
                "join_hints": []
            }

            examples = example_gen.get_relevant_examples(
                user_query=session.query,
                tables=session.tables,
                query_intent=query_intent,
                max_examples=3
            )

            # Get learned patterns for these tables
            if session.tables:
                learned_patterns = example_gen.get_learned_patterns_for_tables(session.tables)

        except Exception as e:
            # Don't fail SQL generation if examples can't be fetched
            logging.getLogger(__name__).debug(f"Could not get examples: {e}")

        # Generate SQL with examples
        result = llm_service.generate_sql(
            session.query,
            context,
            provider,
            examples=examples,
            learned_patterns=learned_patterns
        )

        sql = result.sql
        session.sql = sql
        session.suggested_title = result.suggested_title
        session.current_step = "sql_ready"

        # Validate
        validation = tools["validate_sql"](sql, database_id=session.database_id, strict=False)

        # Build response with context info first, then SQL
        messages = []

        # Include context messages (tables, joins, filters) if provided
        if context_messages:
            messages.extend(context_messages)
            messages.append("")  # Empty line before SQL

        # Add the generated SQL
        messages.append(f"**Generated SQL:**\n```sql\n{sql}\n```")

        # Only show explanation if it adds value
        if result.explanation and len(result.explanation) > 10:
            messages.append(f"\n{result.explanation}")

        # Show warnings if any (skip assumptions to reduce verbosity)
        if validation.get("warnings"):
            warnings_str = ", ".join(validation["warnings"][:2])
            messages.append(f"\n⚠️ {warnings_str}")

        return jsonify({
            "response": "\n".join(messages),
            "step": "sql_ready",
            "sql": sql,
            "database_id": session.database_id,
            "suggested_title": result.suggested_title,
            "explanation": result.explanation,
            "provider": result.provider,
            "model": result.model,
            "validation": validation,
            "actions": ["execute", "save_only", "get_url", "create_chart", "cancel"],
            "needs_input": True
        })

    except ImportError:
        # Fallback to direct OpenAI if LLM service not available
        return generate_sql_openai_fallback(session, tools, context)

    except Exception as e:
        return jsonify({
            "response": f"Error generating SQL: {str(e)}",
            "step": "error",
            "needs_input": True
        })


def generate_sql_openai_fallback(session: ChatSession, tools: dict, context: dict) -> Response:
    """Fallback SQL generation using OpenAI directly."""
    llm_config = get_llm_config()

    if not llm_config or not llm_config.openai_api_key:
        return jsonify({
            "response": "No LLM API key configured. Cannot generate SQL.",
            "step": "error",
            "needs_input": True
        })

    try:
        client = OpenAI(api_key=llm_config.openai_api_key)

        filter_instructions = ""
        if session.filters:
            filter_clauses = [f"{f['column']} = '{f['value']}'" for f in session.filters]
            filter_instructions = f"\nApply these WHERE filters:\n- " + "\n- ".join(filter_clauses)

        system_prompt = f"""You are a SQL generator for Apache Superset.
Engine: {context.get('engine', 'postgresql')}

Tables:
{json.dumps(context.get('tables', []), indent=2)}

Available joins:
{json.dumps(context.get('available_joins', []), indent=2)}

Rules:
{json.dumps(context.get('rules', []), indent=2)}
{filter_instructions}

Return ONLY the SQL query, no explanations."""

        response = client.chat.completions.create(
            model=llm_config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": session.query}
            ],
            temperature=0.1
        )

        sql = response.choices[0].message.content.strip()

        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        session.sql = sql
        session.current_step = "sql_ready"

        validation = tools["validate_sql"](sql, database_id=session.database_id, strict=False)

        return jsonify({
            "response": f"**Generated SQL:**\n```sql\n{sql}\n```\n\n**What would you like to do?**",
            "step": "sql_ready",
            "sql": sql,
            "database_id": session.database_id,
            "provider": "openai",
            "validation": validation,
            "actions": ["execute", "save_only", "get_url", "cancel"],
            "needs_input": True
        })

    except Exception as e:
        return jsonify({
            "response": f"Error generating SQL: {str(e)}",
            "step": "error",
            "needs_input": True
        })


def modify_sql(session: ChatSession, tools: dict, modification_request: str) -> Response:
    """Modify existing SQL based on user's request using LLM."""
    session.current_step = "modifying_sql"

    try:
        llm_service = get_llm_service()

        # Get provider
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "claude": LLMProvider.CLAUDE,
            "gemini": LLMProvider.GEMINI
        }
        if session.llm_provider:
            provider = provider_map.get(session.llm_provider, llm_service.config.default_provider)
        else:
            provider = llm_service.config.default_provider

        # Build modification prompt
        modify_prompt = f"""You are modifying an existing SQL query based on user feedback.

CURRENT SQL:
```sql
{session.sql}
```

USER'S MODIFICATION REQUEST: {modification_request}

RULES:
1. Apply the requested change to the existing SQL
2. Keep all other parts of the query unchanged
3. Maintain the same style (no unnecessary aliases for single tables)
4. Return ONLY valid JSON with the modified SQL

OUTPUT FORMAT (JSON):
{{
    "sql": "SELECT ... FROM ...",
    "explanation": "What was changed",
    "change_summary": "Brief summary of the modification"
}}"""

        # Call LLM
        if provider == LLMProvider.OPENAI:
            client = llm_service._get_openai_client()
            response = client.chat.completions.create(
                model=llm_service.config.openai_model,
                messages=[{"role": "user", "content": modify_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        elif provider == LLMProvider.CLAUDE:
            client = llm_service._get_claude_client()
            response = client.messages.create(
                model=llm_service.config.claude_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": modify_prompt}]
            )
            content = response.content[0].text
            start = content.find('{')
            end = content.rfind('}') + 1
            result = json.loads(content[start:end])
        else:
            result = {"sql": session.sql, "explanation": "Provider not supported for modification"}

        # Update session
        new_sql = result.get("sql", session.sql)
        session.sql = new_sql
        session.current_step = "sql_ready"

        # Validate
        validation = tools["validate_sql"](new_sql, database_id=session.database_id, strict=False)

        change_summary = result.get("change_summary", result.get("explanation", "Query modified"))

        messages = [
            f"**Modified SQL** ({change_summary}):\n```sql\n{new_sql}\n```"
        ]

        if result.get("explanation"):
            messages.append(f"\n**Changes:** {result['explanation']}")

        messages.append("\n**What would you like to do?**")
        messages.append("\n_Or describe more changes, or type a new query_")

        return jsonify({
            "response": "\n".join(messages),
            "step": "sql_ready",
            "sql": new_sql,
            "database_id": session.database_id,
            "suggested_title": session.suggested_title,
            "provider": provider.value if hasattr(provider, 'value') else str(provider),
            "model": llm_service.config.openai_model if provider == LLMProvider.OPENAI else llm_service.config.claude_model,
            "validation": validation,
            "actions": ["execute", "save_only", "get_url", "create_chart", "cancel"],
            "needs_input": True
        })

    except Exception as e:
        return jsonify({
            "response": f"Error modifying SQL: {str(e)}\n\nYou can try again or type a new query.",
            "step": "sql_ready",
            "sql": session.sql,
            "actions": ["execute", "save_only", "get_url", "cancel"],
            "needs_input": True
        })


def execute_query(session: ChatSession, tools: dict) -> Response:
    """Execute the query and save to SQL Lab."""
    if not session.sql or not session.database_id:
        return jsonify({
            "response": "No SQL query ready to execute.",
            "step": "idle",
            "needs_input": True
        })

    try:
        result = tools["execute_and_save"](
            database_id=session.database_id,
            sql=session.sql,
            label=f"Chat: {session.query[:40]}",
            schema=session.schema,
            description=f"Generated from chat: {session.query}",
            limit=1000
        )

        execution = result.get("execution", {})
        saved = result.get("saved_query", {})

        session.sqllab_url = saved.get("sqllab_url")
        session.current_step = "completed"

        row_count = execution.get("row_count", 0)
        columns = execution.get("columns", [])

        # Record successful query feedback for learning
        try:
            knowledge_store = KnowledgeStore()
            knowledge_store.record_query_feedback(
                natural_language=session.query or "",
                generated_sql=session.sql,
                feedback_type="success",
                row_count=row_count
            )
        except Exception:
            pass  # Don't fail if feedback recording fails

        messages = [
            f"**Query executed successfully!**",
            f"• Rows: {row_count}",
            f"• Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}",
            f"\n**Open in SQL Lab:** [Click here]({session.sqllab_url})"
        ]

        return jsonify({
            "response": "\n".join(messages),
            "step": "completed",
            "sqllab_url": session.sqllab_url,
            "row_count": row_count,
            "columns": columns,
            "redirect": True,
            "needs_input": True
        })

    except Exception as e:
        # Record failed query feedback for learning
        try:
            knowledge_store = KnowledgeStore()
            knowledge_store.record_query_feedback(
                natural_language=session.query or "",
                generated_sql=session.sql or "",
                feedback_type="failure",
                error_message=str(e)
            )
        except Exception:
            pass  # Don't fail if feedback recording fails

        return jsonify({
            "response": f"Execution failed: {str(e)}",
            "step": "error",
            "needs_input": True
        })


def save_query_only(session: ChatSession, tools: dict) -> Response:
    """Save the query without executing."""
    if not session.sql or not session.database_id:
        return jsonify({
            "response": "No SQL query ready to save.",
            "step": "idle",
            "needs_input": True
        })

    try:
        result = tools["create_saved_query"](
            database_id=session.database_id,
            sql=session.sql,
            label=f"Chat: {session.query[:40]}",
            description=f"Generated from chat: {session.query}"
        )

        session.sqllab_url = result.get("sqllab_url")
        session.current_step = "completed"

        return jsonify({
            "response": f"**Query saved!**\n\n[Open in SQL Lab]({session.sqllab_url})",
            "step": "completed",
            "sqllab_url": session.sqllab_url,
            "redirect": True,
            "needs_input": True
        })

    except Exception as e:
        return jsonify({
            "response": f"Save failed: {str(e)}",
            "step": "error",
            "needs_input": True
        })


def get_url_only(session: ChatSession, tools: dict) -> Response:
    """Get SQL Lab URL without saving (with database and schema pre-selected)."""
    if not session.sql or not session.database_id:
        return jsonify({
            "response": "No SQL query ready.",
            "step": "idle",
            "needs_input": True
        })

    try:
        # Pass schema to pre-select it in SQL Lab
        result = tools["get_sqllab_url"](
            database_id=session.database_id,
            sql=session.sql,
            schema=session.schema
        )

        session.sqllab_url = result.get("sqllab_url")
        session.current_step = "completed"

        return jsonify({
            "response": f"**Open in SQL Lab:**\n\n[Click here]({session.sqllab_url})",
            "step": "completed",
            "sqllab_url": session.sqllab_url,
            "redirect": True,
            "needs_input": True
        })

    except Exception as e:
        return jsonify({
            "response": f"Error: {str(e)}",
            "step": "error",
            "needs_input": True
        })


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset a chat session."""
    data = request.json
    session_id = data.get("session_id", "default")

    if session_id in sessions:
        del sessions[session_id]

    return jsonify({"status": "ok", "message": "Session reset"})


# ===========================================================================
# Query Learning Endpoints
# ===========================================================================

@app.route("/api/train-queries", methods=["POST"])
def train_queries():
    """
    Train the query learning system from existing Superset queries, charts, and datasets.

    POST body (all optional):
    {
        "include_saved_queries": true,
        "include_charts": true,
        "include_datasets": true,
        "since_date": "2026-01-01"
    }
    """
    data = request.json or {}

    tools = get_mcp_tools()
    result = tools.get("train_from_queries")

    if result is None:
        # Call the tool directly if not in tools dict
        result = mcp.train_from_queries.fn(
            include_saved_queries=data.get("include_saved_queries", True),
            include_charts=data.get("include_charts", True),
            include_datasets=data.get("include_datasets", True),
            since_date=data.get("since_date")
        )
    else:
        result = result(
            include_saved_queries=data.get("include_saved_queries", True),
            include_charts=data.get("include_charts", True),
            include_datasets=data.get("include_datasets", True),
            since_date=data.get("since_date")
        )

    return jsonify(result)


@app.route("/api/query-examples", methods=["GET"])
def get_query_examples():
    """
    Get stored query examples.

    Query params:
    - tables: comma-separated list of table names
    - keywords: comma-separated keywords
    - type: query type (simple, join, aggregation)
    - limit: max results (default 10)
    """
    tables = request.args.get("tables", "").split(",") if request.args.get("tables") else None
    keywords = request.args.get("keywords", "").split(",") if request.args.get("keywords") else None
    query_type = request.args.get("type")
    limit = int(request.args.get("limit", 10))

    # Filter out empty strings
    if tables:
        tables = [t.strip() for t in tables if t.strip()]
    if keywords:
        keywords = [k.strip() for k in keywords if k.strip()]

    result = mcp.get_query_examples.fn(
        tables=tables or None,
        keywords=keywords or None,
        query_type=query_type,
        limit=limit
    )

    return jsonify(result)


@app.route("/api/query-learning-stats", methods=["GET"])
def query_learning_stats():
    """Get statistics about the query learning system."""
    result = mcp.get_query_learning_stats.fn()
    return jsonify(result)


@app.route("/api/validate-examples", methods=["POST"])
def validate_examples():
    """Validate query examples against current schema."""
    result = mcp.validate_query_examples.fn()
    return jsonify(result)


@app.route("/api/architecture", methods=["GET"])
def get_architecture():
    """
    Get complete architecture documentation of the Superset MCP system.

    Returns detailed information about:
    - System components and their roles
    - Data flow and processing pipeline
    - Training process
    - API endpoints
    - Configuration options
    """
    architecture = {
        "name": "Superset MCP Chatbot",
        "version": "1.0.0",
        "description": "Natural language to SQL query generator for Apache Superset",

        "components": {
            "chatbot_api": {
                "path": "chatbot/api_server.py",
                "description": "Flask REST API server that handles chat conversations and orchestrates the query generation pipeline",
                "port": 5050,
                "endpoints": [
                    {"path": "/api/chat", "method": "POST", "description": "Main chat endpoint for query generation"},
                    {"path": "/api/health", "method": "GET", "description": "Health check"},
                    {"path": "/api/providers", "method": "GET", "description": "List available LLM providers"},
                    {"path": "/api/train-queries", "method": "POST", "description": "Train from existing queries"},
                    {"path": "/api/query-examples", "method": "GET", "description": "Get stored query examples"},
                    {"path": "/api/architecture", "method": "GET", "description": "This endpoint - system architecture"},
                    {"path": "/api/query-logs", "method": "GET", "description": "Get query logs for training"}
                ]
            },
            "mcp_server": {
                "path": "mcp_superset_server.py",
                "description": "MCP (Model Context Protocol) server providing tools for Superset interaction",
                "tools": [
                    "build_execution_plan - Analyze NL query and discover tables",
                    "generate_sql_context - Build context for SQL generation",
                    "validate_sql - Validate SQL syntax and dialect",
                    "execute_and_save - Execute query and save to SQL Lab",
                    "get_sqllab_url - Generate SQL Lab URL",
                    "train_from_queries - Train from existing Superset queries"
                ]
            },
            "llm_service": {
                "path": "llm/llm_service.py",
                "description": "Unified LLM service supporting multiple providers",
                "providers": ["OpenAI (GPT-4/5)", "Anthropic Claude", "Google Gemini"],
                "functions": [
                    "generate_sql - Generate SQL from natural language",
                    "explain_query - Explain and optimize SQL queries"
                ]
            },
            "nlu": {
                "path": "nlu/",
                "description": "Natural Language Understanding components",
                "modules": {
                    "semantic_matcher.py": "Table/column matching using synonyms, aliases, fuzzy matching",
                    "table_intelligence.py": "Date column detection, time filter handling",
                    "intent_parser.py": "Query intent extraction (filters, aggregations, time ranges)"
                }
            },
            "learning": {
                "path": "learning/",
                "description": "Query learning and few-shot example system",
                "modules": {
                    "query_harvester.py": "Harvest queries from Superset saved queries and charts",
                    "query_analyzer.py": "Analyze SQL patterns using sqlglot",
                    "knowledge_store.py": "SQLite store for query examples and feedback",
                    "example_generator.py": "Generate few-shot examples for LLM prompts"
                }
            },
            "validation": {
                "path": "validation/sql_validator.py",
                "description": "SQL validation with dialect-specific rules",
                "dialects": ["StarRocks", "PostgreSQL"]
            },
            "config": {
                "path": "config/",
                "description": "Configuration management",
                "files": {
                    "semantic_config.json": "Table schemas, synonyms, join patterns",
                    "llm_config.py": "LLM provider configuration"
                }
            }
        },

        "data_flow": {
            "1_user_input": "User sends natural language query via chat API",
            "2_intent_parsing": "NLU extracts entities, filters, time ranges, state filters",
            "3_table_discovery": "Semantic matcher finds relevant tables using synonyms, aliases",
            "4_virtual_resolution": "Virtual datasets resolved to physical tables",
            "5_join_discovery": "Join paths found between multiple tables",
            "6_context_building": "Schema context built with columns, joins, examples",
            "7_sql_generation": "LLM generates SQL using context and few-shot examples",
            "8_validation": "SQL validated for syntax and dialect compatibility",
            "9_execution": "Query executed in Superset and results returned"
        },

        "training_process": {
            "description": "System learns from existing Superset queries to improve generation",
            "steps": {
                "1_harvest": "Fetch saved queries and chart SQL from Superset API",
                "2_analyze": "Parse SQL to extract patterns (tables, joins, filters, aggregations)",
                "3_store": "Save analyzed examples to knowledge store (SQLite)",
                "4_generate": "Select relevant examples based on user query similarity",
                "5_enhance": "Add examples to LLM prompt for few-shot learning",
                "6_feedback": "Track success/failure to refine example selection"
            },
            "trigger": "POST /api/train-queries"
        },

        "deployment": {
            "docker": {
                "dockerfile": "Dockerfile",
                "compose": "docker-compose.yml",
                "port": 5050,
                "command": "docker-compose up -d"
            },
            "local": {
                "command": "python chatbot/api_server.py",
                "port": 5050
            },
            "bookmarklet": {
                "install": "http://localhost:5050/install",
                "description": "Drag bookmarklet to toolbar, click on any Superset page"
            }
        },

        "configuration": {
            "env_vars": {
                "SUPERSET_URL": "Superset instance URL",
                "SUPERSET_USERNAME": "Superset admin username",
                "SUPERSET_PASSWORD": "Superset admin password",
                "OPENAI_API_KEY": "OpenAI API key",
                "ANTHROPIC_API_KEY": "Anthropic API key (optional)",
                "GOOGLE_API_KEY": "Google API key (optional)",
                "ENGINE_HINT": "Database dialect (starrocks/postgresql)",
                "DEFAULT_DATABASE": "Default database ID",
                "DEFAULT_SCHEMA": "Default schema name"
            }
        }
    }

    return jsonify(architecture)


# Query logging for LLM training
QUERY_LOG_FILE = os.path.join(PARENT_DIR, "query_logs", "query_log.txt")


def log_query_for_training(user_query: str, generated_sql: str, llm_response: dict,
                           success: bool, error: str = None):
    """Log query interactions for future LLM training."""
    os.makedirs(os.path.dirname(QUERY_LOG_FILE), exist_ok=True)

    timestamp = datetime.datetime.now().isoformat()

    log_entry = f"""
================================================================================
TIMESTAMP: {timestamp}
SUCCESS: {success}
--------------------------------------------------------------------------------
USER QUERY:
{user_query}
--------------------------------------------------------------------------------
GENERATED SQL:
{generated_sql}
--------------------------------------------------------------------------------
LLM RESPONSE:
{json.dumps(llm_response, indent=2) if llm_response else 'N/A'}
--------------------------------------------------------------------------------
ERROR:
{error or 'None'}
================================================================================

"""

    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to log query: {e}")


@app.route("/api/query-logs", methods=["GET"])
def get_query_logs():
    """
    Get query logs for LLM training.

    Query params:
    - limit: Max number of recent entries (default 100)
    - format: 'json' or 'text' (default 'json')
    """
    limit = int(request.args.get("limit", 100))
    fmt = request.args.get("format", "json")

    if not os.path.exists(QUERY_LOG_FILE):
        return jsonify({"logs": [], "count": 0, "message": "No query logs found"})

    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        if fmt == "text":
            return Response(content, mimetype="text/plain")

        # Parse log entries
        entries = content.split("=" * 80)
        logs = []

        for entry in entries[-limit:]:
            entry = entry.strip()
            if not entry:
                continue

            log = {}
            sections = entry.split("-" * 80)

            for section in sections:
                section = section.strip()
                if section.startswith("TIMESTAMP:"):
                    log["timestamp"] = section.replace("TIMESTAMP:", "").strip().split("\n")[0]
                    if "SUCCESS:" in section:
                        log["success"] = "True" in section
                elif section.startswith("USER QUERY:"):
                    log["user_query"] = section.replace("USER QUERY:", "").strip()
                elif section.startswith("GENERATED SQL:"):
                    log["generated_sql"] = section.replace("GENERATED SQL:", "").strip()
                elif section.startswith("ERROR:"):
                    error_text = section.replace("ERROR:", "").strip()
                    log["error"] = error_text if error_text != "None" else None

            if log:
                logs.append(log)

        return jsonify({
            "logs": logs,
            "count": len(logs),
            "total_in_file": len(entries) - 1,
            "log_file": QUERY_LOG_FILE
        })

    except Exception as e:
        return jsonify({"error": str(e), "logs": []})


def _cleanup_old_backups(log_dir: str, keep_count: int = 1) -> list:
    """Delete old backup files, keeping only the most recent ones."""
    deleted = []
    if os.path.exists(log_dir):
        backup_files = sorted([
            f for f in os.listdir(log_dir)
            if f.startswith("query_log_backup_") and f.endswith(".txt")
        ])
        for old_backup in backup_files[:-keep_count] if keep_count > 0 else backup_files:
            old_path = os.path.join(log_dir, old_backup)
            os.remove(old_path)
            deleted.append(old_backup)
    return deleted


@app.route("/api/query-logs/clear", methods=["POST"])
def clear_query_logs():
    """Clear query logs (for testing/reset). Keeps only the last 1 backup."""
    try:
        log_dir = os.path.dirname(QUERY_LOG_FILE)
        deleted_backups = _cleanup_old_backups(log_dir, keep_count=1)

        if os.path.exists(QUERY_LOG_FILE):
            # Create new backup
            backup_file = QUERY_LOG_FILE.replace(".txt", f"_backup_{int(time.time())}.txt")
            os.rename(QUERY_LOG_FILE, backup_file)

            # Delete old backups after creating new one (keep only 1)
            deleted_backups.extend(_cleanup_old_backups(log_dir, keep_count=1))

            return jsonify({
                "message": "Logs cleared",
                "backup": backup_file,
                "deleted_old_backups": deleted_backups
            })
        return jsonify({"message": "No logs to clear", "deleted_old_backups": deleted_backups})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("CHATBOT_PORT", 5050))
    print(f"\n{'='*60}")
    print("Superset Chatbot API Server")
    print(f"{'='*60}")
    print(f"Running on: http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/api/health")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=port, debug=True)