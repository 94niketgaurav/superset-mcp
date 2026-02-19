# Superset-MCP User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Training the Semantic Matcher](#training-the-semantic-matcher)
6. [Using the Query Builder](#using-the-query-builder)
7. [Understanding Matching and Joins](#understanding-matching-and-joins)
8. [Running Tests](#running-tests)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Introduction

Superset-MCP is an intelligent natural language to SQL query builder for Apache Superset. Simply describe what data you want in plain English, and the system will:

1. **Understand your intent** - Extract entities like "assets" and "events"
2. **Find the right tables** - Match your words to actual Superset datasets
3. **Figure out joins** - Automatically detect how tables connect (e.g., `assetId` → `assets.id`)
4. **Generate SQL** - Create proper SQL with correct joins
5. **Validate** - Check for errors before running
6. **Execute** - Run the query and return results

### Key Features

- **Natural Language Queries**: "Show me assets and their events for last week"
- **Semantic Matching**: Understands "asset" matches "assets", "assetversion", "assethistory"
- **Smart Joins**: Detects `assetId` → `assets.id` even with different naming conventions
- **Auto-Learning**: Improves matching over time by learning your datasets
- **Interactive Confirmation**: Review and approve before executing

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Access to an Apache Superset instance
- OpenAI API key (for SQL generation)
- Optional: Gemini API key (for SQL review)

### Steps

```bash
# 1. Clone or navigate to the project
cd superset-mcp

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your configuration file
cp .env.example .env

# 5. Edit .env with your credentials
nano .env  # or use your preferred editor
```

---

## Configuration

### Required Environment Variables

Edit your `.env` file with these required settings:

```bash
# Superset Connection (REQUIRED)
SUPERSET_URL=https://your-superset-instance.com
SUPERSET_USERNAME=your_username
SUPERSET_PASSWORD=your_password

# LLM API Key (at least one required)
OPENAI_API_KEY=sk-your-openai-key
```

### Optional Environment Variables

```bash
# Superset Settings
SUPERSET_PROVIDER=db              # Authentication provider
READ_ONLY=true                    # Block write operations
ALLOWED_DATABASE_IDS=1,2,3        # Restrict to specific databases

# LLM Settings
OPENAI_MODEL=gpt-4                # OpenAI model to use
GEMINI_API_KEY=your-gemini-key    # For SQL review (optional)
GEMINI_MODEL=gemini-2.0-flash     # Gemini model to use

# Application Settings
INTERACTIVE_MODE=true             # Enable confirmation prompts
CACHE_TTL_SECONDS=300             # Dataset cache duration
ENGINE_HINT=postgresql            # SQL dialect hint

# Relationships
RELATIONSHIPS_PATH=relationships.yaml  # Path to relationship definitions
```

### Defining Relationships (Optional)

Create a `relationships.yaml` file to define explicit join relationships:

```yaml
relationships:
  - left_dataset_id: 1
    right_dataset_id: 4
    left_column: id
    right_column: assetId
    type: inner

  - left_dataset_id: 6
    right_dataset_id: 1
    left_column: id
    right_column: userId
    type: left
```

---

## Quick Start

### Interactive Mode

```bash
# Run the query builder
python orchestrator.py
```

You'll see:

```
============================================================
Superset MCP - Natural Language Query Builder
============================================================

What do you want to query? >
```

Enter your query in plain English:

```
What do you want to query? > show me assets and their events

[Step 1] Analyzing query and discovering datasets...

  Entity: 'assets'
    -> Matched: assets (score: 1.00)

  Entity: 'events'
    Multiple candidates found:
      1. posthogevents (score: 0.68, reason: suffix match)
      2. auditevents (score: 0.65, reason: suffix match)
    Select dataset number: 1

[Step 3] Analyzing join relationships...
  Suggested joins:
    1. posthogevents.assetId INNER JOIN assets.id
       Confidence: 0.82, Reason: FK pattern: assetId -> assets.id

  Accept these joins? (Y/n) > y

[Step 4] Generating SQL...
[Step 5] Reviewing SQL with Gemini...
[Step 6] Validating SQL...
  Validation passed.

============================================================
GENERATED SQL:
============================================================
SELECT
    a.id AS asset_id,
    a.name AS asset_name,
    e.event_type,
    e.timestamp
FROM assets a
INNER JOIN posthogevents e ON e.assetId = a.id
LIMIT 1000
============================================================

Execute this query? (Y/n) > y

Executing...
```

### Non-Interactive Mode

Run with environment variables:

```bash
USER_REQUEST="show me assets and events" \
INTERACTIVE_MODE=false \
python orchestrator.py
```

---

## Training the Semantic Matcher

The semantic matcher can learn your datasets to improve matching accuracy. The training pipeline automatically:
1. Logs into Superset using your credentials
2. Discovers all databases and their schemas
3. Lists all datasets organized by database → schema → table
4. Learns column information and patterns
5. Discovers potential join patterns

### Full Training (Recommended First Time)

```bash
# Full training with sample value analysis
python train_semantic_matcher.py
```

This will output:

```
============================================================
Superset-MCP Semantic Matcher Training Pipeline
============================================================
Started at: 2024-01-15T14:30:00
Connection Settings:
  Superset URL: https://your-superset-instance.com
  Username: your_username

[Step 1] Logging into Superset...
  ✓ Successfully authenticated with Superset

[Step 2] Discovering databases...
  ✓ Found 3 database(s)
    • Production DB (ID: 1, Backend: postgresql)
      Schemas: public, analytics, reporting
    • Data Warehouse (ID: 2, Backend: snowflake)
      Schemas: raw, staging, mart

[Step 3] Fetching all datasets...
  ✓ Found 45 dataset(s)

============================================================
SUPERSET DATABASE HIERARCHY
============================================================

📦 Production DB (ID: 1, Backend: postgresql)
   └── 2 schema(s), 30 table(s)
       📁 public (15 tables)
          └── users
          └── orders
          └── products
          └── ... and 12 more
       📁 analytics (15 tables)
          └── user_metrics
          └── ... and 14 more

[Step 4] Learning dataset metadata and columns...
  Processing database: Production DB
    Schema: public (15 tables)
      [1/45] Learning: users (12 columns)
      [2/45] Learning: orders (8 columns)
      ...

[Step 5] Analyzing join patterns...
  ✓ Found 23 potential join patterns

  Top join patterns discovered:
    • orders.user_id → users.id (confidence: 0.95)
    • order_items.order_id → orders.id (confidence: 0.92)
    ...

============================================================
LEARNING COMPLETE
============================================================
  Databases: 3
  Datasets: 45/45
  Join patterns: 23
  Errors: 0
  Time: 45.3s
```

### Quick Training

```bash
# Quick training without sample values (faster but less accurate joins)
python train_semantic_matcher.py --quick
```

### View Training Status

```bash
# Show knowledge store statistics
python train_semantic_matcher.py --stats
```

Output:
```
==================================================
KNOWLEDGE STORE STATISTICS
==================================================

Storage: /home/user/.superset-mcp/knowledge.db
Last training: 2024-01-15T14:30:00

📊 Data Summary:
   Databases:       3
   Schemas:         8
   Datasets:        45
   Total columns:   523

🔗 Relationships:
   Join patterns:   23
   Synonyms:        12
   Column patterns: 156

📦 Databases:
   • Production DB (ID: 1, Backend: postgresql, Schemas: 3)
   • Data Warehouse (ID: 2, Backend: snowflake, Schemas: 4)
   • Analytics (ID: 3, Backend: bigquery, Schemas: 1)

==================================================
```

### List Learned Datasets

```bash
# List all datasets grouped by database and schema
python train_semantic_matcher.py --list
```

Output:
```
============================================================
LEARNED DATASETS
============================================================

📦 Production DB (ID: 1)
   2 schema(s), 30 table(s)

   📁 public
      └── users (12 cols, 2 keys)
      └── orders (8 cols, 3 keys)
      └── products (6 cols, 1 keys)
      ...

   📁 analytics
      └── user_metrics (15 cols, 2 keys)
      ...

============================================================
Total: 45 tables across 3 database(s)
============================================================
```

### Test the Trained Matcher

```bash
python train_semantic_matcher.py --test
```

Output:
```
Testing trained semantic matcher...
----------------------------------------
Loaded 45 datasets
Loaded 23 join patterns
Loaded 12 synonyms

Entity: 'assets'
  -> Recommended: assets (score: 1.00)
  -> Learned joins:
     - id -> posthogevents.assetId (confidence: 0.82)
     - id -> assethistory.assetId (confidence: 0.78)

Entity: 'events'
  -> Top candidates:
     - posthogevents (score: 0.68, reason: suffix match)
     - auditevents (score: 0.65, reason: suffix match)
```

### Export/Import Knowledge

```bash
# Export to JSON
python train_semantic_matcher.py --export knowledge_backup.json

# Import from JSON
python train_semantic_matcher.py --import knowledge_backup.json
```

### Schedule Daily Training (Cron)

Add to crontab (`crontab -e`):

```bash
# Run training daily at 2 AM
0 2 * * * cd /path/to/superset-mcp && /path/to/venv/bin/python train_semantic_matcher.py >> /var/log/superset-mcp-train.log 2>&1
```

---

## Using the Query Builder

### Example Queries

| Query | What It Does |
|-------|--------------|
| "show me assets" | Lists all assets |
| "assets and events" | Joins assets with events |
| "users with their orders" | Joins users with orders |
| "count all products" | Counts products |
| "events from last week" | Events with time filter |
| "top 10 customers" | Limited results |
| "assets by assetId" | Explicit join key hint |

### Query Tips

1. **Use plural nouns** for tables: "assets" not "asset"
2. **Mention relationships**: "and", "with", "by"
3. **Specify join keys** if needed: "by assetId"
4. **Add time filters**: "last week", "since January"
5. **Request aggregations**: "count", "sum", "average"

### Handling Ambiguous Matches

When multiple tables match your query, you'll be prompted:

```
  Entity: 'events'
    Multiple candidates found:
      1. posthogevents (score: 0.68, reason: suffix match)
      2. auditevents (score: 0.65, reason: suffix match)
      3. historyevents (score: 0.62, reason: suffix match)
    Select dataset number:
```

Enter the number of your choice (e.g., `1` for posthogevents).

---

## Understanding Matching and Joins

### How Dataset Matching Works

The semantic matcher uses multiple strategies (in order of priority):

| Strategy | Score | Example |
|----------|-------|---------|
| Exact match | 1.00 | "assets" → "assets" |
| Learned alias | 0.98 | "item" → "assets" (learned) |
| Learned synonym | 0.96 | "resource" → "assets" (learned) |
| Built-in synonym | 0.95 | "user" → "users" |
| Prefix match | 0.70-0.90 | "asset" → "assetversion" |
| Suffix match | 0.65-0.80 | "events" → "posthogevents" |
| Contains match | 0.50-0.65 | "log" → "audit_logs" |
| Fuzzy match | varies | "assts" → "assets" (typo) |

### How Join Detection Works

The enhanced join reasoner detects joins using:

1. **YAML Relationships** (confidence: 1.0)
   - Explicitly defined in `relationships.yaml`

2. **Normalized Column Match** (confidence: 0.75-0.85)
   - `assetId` ↔ `asset_id` (same after normalization)

3. **FK → PK Pattern** (confidence: 0.65-0.80)
   - `assetId` in events → `id` in assets
   - Infers from column name + table name match

4. **Learned Patterns** (confidence: varies)
   - Patterns discovered during training
   - Based on value overlap analysis

### Column Name Normalization

The system handles different naming conventions:

| Original | Normalized |
|----------|------------|
| `assetId` | `asset_id` |
| `userId` | `user_id` |
| `CreatedByUserId` | `created_by_user_id` |
| `user-name` | `user_name` |

This allows matching `assetId` (camelCase) with `asset_id` (snake_case).

---

## Running Tests

### Run All Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# NLU tests only
pytest tests/test_happy_flow.py::TestEntityExtractor -v
pytest tests/test_happy_flow.py::TestSemanticMatcher -v

# Join tests
pytest tests/test_happy_flow.py::TestColumnNormalizer -v
pytest tests/test_happy_flow.py::TestEnhancedJoinReasoner -v

# Validation tests
pytest tests/test_happy_flow.py::TestSQLValidator -v

# Integration tests
pytest tests/test_happy_flow.py::TestHappyFlowIntegration -v
```

### Test Output Example

```
tests/test_happy_flow.py::TestEntityExtractor::test_extract_simple_entities PASSED
tests/test_happy_flow.py::TestEntityExtractor::test_extract_join_hints PASSED
tests/test_happy_flow.py::TestSemanticMatcher::test_exact_match PASSED
tests/test_happy_flow.py::TestSemanticMatcher::test_prefix_match PASSED
tests/test_happy_flow.py::TestColumnNormalizer::test_camel_to_snake PASSED
tests/test_happy_flow.py::TestSQLValidator::test_valid_select PASSED
tests/test_happy_flow.py::TestHappyFlowIntegration::test_full_query_building_flow PASSED

==================== 15 passed in 0.45s ====================
```

---

## Troubleshooting

### Common Issues

#### 1. "Configuration Error: Missing required environment variables"

**Solution**: Ensure your `.env` file has the required variables:
```bash
SUPERSET_URL=https://...
SUPERSET_USERNAME=...
SUPERSET_PASSWORD=...
OPENAI_API_KEY=sk-...
```

#### 2. "No datasets selected"

**Cause**: The semantic matcher couldn't find any matching tables.

**Solution**:
- Run training: `python train_semantic_matcher.py`
- Check if datasets exist in Superset
- Try more specific table names
- Use `--stats` to verify knowledge store has data

#### 3. "Datasets span multiple databases"

**Cause**: Selected tables are in different Superset databases (can't join).

**Solution**:
- Choose tables from the same database
- Check `ALLOWED_DATABASE_IDS` setting

#### 4. "No automatic joins detected"

**Cause**: System couldn't find join columns.

**Solution**:
- Define relationships in `relationships.yaml`
- Run training to discover patterns
- Mention join key in query: "by assetId"

#### 5. "READ_ONLY is enabled; write SQL is blocked"

**Cause**: The generated SQL contains INSERT/UPDATE/DELETE.

**Solution**:
- This is a safety feature
- Set `READ_ONLY=false` in `.env` if you need write operations
- Be careful with write operations!

#### 6. LLM Errors

**OpenAI Error**: Check `OPENAI_API_KEY` is valid and has credits
**Gemini Error**: Gemini is optional; SQL will still generate without it

### Debug Mode

For more detailed output, run Python with verbose logging:

```bash
PYTHONVERBOSE=1 python orchestrator.py
```

### Checking Knowledge Store

```bash
# View statistics
python train_semantic_matcher.py --stats

# Export and inspect
python train_semantic_matcher.py --export debug.json
cat debug.json | python -m json.tool | less
```

---

## Advanced Usage

### Using as MCP Server

The system can run as an MCP server for Claude or other MCP clients:

```bash
# Run MCP server (stdio mode)
python mcp_superset_server.py
```

### Programmatic Usage

```python
from nlu.semantic_matcher import create_trained_matcher
from nlu.entity_extractor import EntityExtractor
from joins.enhanced_reasoner import EnhancedJoinReasoner

# Create trained matcher
matcher = create_trained_matcher()

# Extract entities from query
extractor = EntityExtractor()
intent = extractor.extract("show me assets and events")

# Match to datasets
for entity in intent.entities:
    result = matcher.match_entity(entity.name)
    print(f"{entity.name} -> {result.recommended.table_name if result.recommended else 'no match'}")

# Get learned joins
joins = matcher.get_learned_joins("assets")
for join in joins:
    print(f"  {join.left_column} -> {join.right_table}.{join.right_column}")
```

### Custom Knowledge Store Location

```python
from learning.knowledge_store import KnowledgeStore

# Use custom location
store = KnowledgeStore("/custom/path/knowledge.db")
```

### API Integration

The MCP tools can be called directly:

```python
from mcp_superset_server import (
    extract_query_entities,
    discover_datasets,
    suggest_enhanced_joins,
    validate_sql,
    build_execution_plan,
    execute_sql
)

# Build complete execution plan
plan = build_execution_plan("show me assets and events", auto_select=True)

# Execute if ready
if plan["ready_for_sql"]:
    # Generate SQL (would need LLM call)
    # Validate
    validation = validate_sql(sql, plan["recommended_dataset_ids"])
    # Execute
    if validation["is_valid"]:
        results = execute_sql(database_id, sql)
```

---

## Support

For issues or questions:
1. Check this documentation
2. Run `python train_semantic_matcher.py --test` to verify setup
3. Review the [Architecture Documentation](ARCHITECTURE.md)
4. Check test examples in `tests/test_happy_flow.py`