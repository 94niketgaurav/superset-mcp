#!/usr/bin/env python3
"""
Training pipeline for the semantic matcher and query learning system.

This script:
1. Connects to Superset and fetches all datasets
2. Analyzes columns and sample values
3. Discovers join patterns
4. Stores knowledge for the semantic matcher
5. Harvests saved queries, charts, and dataset SQL for few-shot learning
6. Analyzes query patterns for improved SQL generation

Usage:
    # Full training (datasets + queries)
    python train_semantic_matcher.py

    # Quick training (no sample values)
    python train_semantic_matcher.py --quick

    # Train only query learning (skip dataset discovery)
    python train_semantic_matcher.py --queries-only

    # Skip query learning (only dataset discovery)
    python train_semantic_matcher.py --skip-queries

    # Export knowledge to JSON
    python train_semantic_matcher.py --export knowledge.json

    # Import knowledge from JSON
    python train_semantic_matcher.py --import knowledge.json

    # Show statistics
    python train_semantic_matcher.py --stats

Schedule this script to run daily with cron:
    0 2 * * * cd /path/to/superset-mcp && python train_semantic_matcher.py >> /var/log/superset-mcp-train.log 2>&1
"""
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_banner():
    """Print a banner."""
    print("=" * 60)
    print("Superset-MCP Semantic Matcher Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()


def check_environment():
    """Check that required environment variables are set."""
    required = ["SUPERSET_URL", "SUPERSET_USERNAME", "SUPERSET_PASSWORD"]
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        print("ERROR: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these variables or create a .env file.")
        sys.exit(1)

    # Show connection info (masked)
    superset_url = os.environ.get("SUPERSET_URL", "")
    superset_user = os.environ.get("SUPERSET_USERNAME", "")
    print("Connection Settings:")
    print(f"  Superset URL: {superset_url}")
    print(f"  Username: {superset_user}")
    print()


def get_knowledge_store():
    """Get the appropriate knowledge store based on configuration."""
    # Check if PostgreSQL is configured
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_db = os.environ.get("POSTGRES_DB")

    if postgres_host or postgres_db:
        try:
            from learning.pg_knowledge_store import PostgresKnowledgeStore
            return PostgresKnowledgeStore(), "postgresql"
        except ImportError as e:
            print(f"WARNING: PostgreSQL configured but psycopg2 not installed: {e}")
            print("Falling back to SQLite...")

    from learning.knowledge_store import KnowledgeStore
    return KnowledgeStore(), "sqlite"


def train_full(fetch_samples: bool = True, skip_queries: bool = False):
    """
    Run full training pipeline.

    Args:
        fetch_samples: Whether to fetch sample values (slower but better results)
        skip_queries: Whether to skip query learning (only run dataset discovery)
    """
    from learning.dataset_learner import DatasetLearner
    from config.semantic_config_manager import get_config_manager

    print("Initializing knowledge store...")
    store, store_type = get_knowledge_store()

    if store_type == "postgresql":
        print(f"Using PostgreSQL: {os.environ.get('POSTGRES_HOST')}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DB')}")
    else:
        print(f"Using SQLite: {store.storage_path}")
    print()

    # Progress callback
    def progress_callback(progress):
        percent = progress.progress_percent
        elapsed = progress.elapsed_time
        print(f"\r  Progress: {percent:.1f}% ({progress.processed_datasets}/{progress.total_datasets}) - {elapsed:.1f}s", end="")

    print("Starting training...")
    learner = DatasetLearner(
        knowledge_store=store,
        sample_size=100 if fetch_samples else 0,
        progress_callback=progress_callback
    )

    stats = learner.learn_all(
        fetch_samples=fetch_samples,
        learn_joins=True
    )

    # Save learned data to JSON config
    print("\n[Step 7] Updating JSON configuration file...")
    try:
        config_manager = get_config_manager()

        # Get all learned data
        all_datasets = store.get_all_datasets()
        tables = []
        synonyms = {}

        for ds in all_datasets:
            # Include full column info with types
            columns_with_types = []
            if ds.columns:
                for col_name, col_info in ds.columns.items():
                    columns_with_types.append({
                        "name": col_name,
                        "type": col_info.column_type or "string",
                        "is_pk": col_info.is_primary_key,
                        "is_fk": col_info.is_foreign_key,
                        "references": col_info.fk_references
                    })

            tables.append({
                "dataset_id": ds.dataset_id,
                "table_name": ds.table_name,
                "schema": ds.schema,
                "database_id": ds.database_id,
                "columns": columns_with_types,
                "aliases": ds.aliases or []
            })

            # Build synonyms from aliases
            for alias in (ds.aliases or []):
                if alias.lower() not in synonyms:
                    synonyms[alias.lower()] = []
                if ds.table_name.lower() not in synonyms[alias.lower()]:
                    synonyms[alias.lower()].append(ds.table_name.lower())

        # Get database info
        db_info_str = store.get_metadata("databases")
        databases = json.loads(db_info_str) if db_info_str else {}

        # Get join patterns
        join_patterns = []
        for jp in store.get_join_patterns(min_confidence=0.3):
            join_patterns.append({
                "left_table": jp.left_table,
                "left_column": jp.left_column,
                "right_table": jp.right_table,
                "right_column": jp.right_column,
                "join_type": jp.join_type,
                "confidence": jp.confidence
            })

        # Update config
        config_manager.update_from_training(
            tables=tables,
            databases=databases,
            synonyms=synonyms,
            join_patterns=join_patterns
        )

        print(f"  ✓ Saved {len(tables)} tables, {len(synonyms)} synonyms, {len(join_patterns)} joins to config")
        print(f"  ✓ Config file: {config_manager.config_path}")

    except Exception as e:
        print(f"  Warning: Could not save to JSON config: {e}")

    print("\n")
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Datasets processed: {stats['datasets_successful']}/{stats['datasets_processed']}")
    print(f"Join patterns found: {stats['join_patterns_found']}")
    print(f"Synonyms created: {stats.get('synonyms_created', 0)}")
    print(f"Errors: {len(stats['errors'])}")
    print(f"Time elapsed: {stats['elapsed_seconds']:.1f}s")

    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")

    # Run query learning if not skipped
    if not skip_queries:
        query_stats = train_query_learning(
            include_saved_queries=True,
            include_charts=True,
            include_datasets=True
        )
        stats['query_learning'] = query_stats
    else:
        print("\n[Skipped] Query learning (--skip-queries)")

    return stats


def train_query_learning(
    include_saved_queries: bool = True,
    include_charts: bool = True,
    include_datasets: bool = True,
    since_date: str = None
):
    """
    Train the query learning system from existing Superset queries.

    This harvests:
    - Saved queries from SQL Lab
    - Charts with SQL configurations
    - Virtual datasets with SQL queries

    Args:
        include_saved_queries: Whether to harvest SQL Lab saved queries
        include_charts: Whether to harvest chart configurations
        include_datasets: Whether to harvest virtual datasets with SQL
        since_date: Only harvest items modified since this date (ISO format)

    Returns:
        Training statistics
    """
    from mcp_superset_server import SupersetClient
    from learning.query_harvester import QueryHarvester
    from learning.query_analyzer import QueryAnalyzer
    from learning.knowledge_store import KnowledgeStore
    from config.semantic_config_manager import get_config_manager

    print("\n" + "=" * 60)
    print("QUERY LEARNING PIPELINE")
    print("=" * 60)

    # Initialize client (uses env vars SUPERSET_URL, SUPERSET_USERNAME, SUPERSET_PASSWORD)
    superset_url = os.environ.get("SUPERSET_URL")

    print("\n[Step 1] Connecting to Superset...")
    client = SupersetClient(superset_url)

    # Get semantic config for validation
    print("[Step 2] Loading semantic configuration...")
    try:
        config_manager = get_config_manager()
        semantic_config = config_manager._config  # Access internal config dict
        if not semantic_config:
            semantic_config = {"schemas": {}}
        print(f"  Loaded config with {len(semantic_config.get('schemas', {}))} schemas")
    except Exception as e:
        print(f"  Warning: Could not load semantic config: {e}")
        semantic_config = {"schemas": {}}

    # Initialize components
    print("[Step 3] Initializing learning components...")
    harvester = QueryHarvester(client)
    analyzer = QueryAnalyzer(semantic_config=semantic_config)
    knowledge_store = KnowledgeStore()

    # Harvest queries
    print("[Step 4] Harvesting queries from Superset...")
    sources = []
    if include_saved_queries:
        sources.append("saved queries")
    if include_charts:
        sources.append("charts")
    if include_datasets:
        sources.append("datasets")
    print(f"  Sources: {', '.join(sources)}")

    def progress_callback(stage, current, total):
        total_str = str(total) if total > 0 else "?"
        print(f"\r  Harvesting {stage}: {current}/{total_str}", end="")

    harvested = harvester.harvest_all(
        include_saved_queries=include_saved_queries,
        include_charts=include_charts,
        include_datasets=include_datasets,
        since_date=since_date,
        progress_callback=progress_callback
    )
    print(f"\n  Total harvested: {len(harvested)} items")

    # Analyze and store queries
    print("\n[Step 5] Analyzing query patterns...")
    stats = {
        "harvested": len(harvested),
        "analyzed": 0,
        "stored": 0,
        "duplicates": 0,
        "errors": 0,
        "by_source": {},
        "by_type": {}
    }

    for i, query in enumerate(harvested, 1):
        if i % 10 == 0 or i == len(harvested):
            print(f"\r  Analyzing: {i}/{len(harvested)}", end="")

        try:
            # Analyze the query
            example = analyzer.analyze(
                sql=query.sql,
                title=query.title,
                description=query.description or ""
            )

            if example is None:
                stats["errors"] += 1
                continue

            stats["analyzed"] += 1

            # Store the example
            import json as json_module
            example_id = knowledge_store.save_query_example(
                title=example.title,
                sql=example.sql,
                normalized_sql=example.normalized_sql,
                pattern_json=json_module.dumps(example.pattern.to_dict()) if example.pattern else "{}",
                keywords=example.keywords,
                tables=example.pattern.tables if example.pattern else [],
                source=query.source,
                source_id=query.source_id,
                description=example.description,
                schema_name=query.schema,
                database_id=query.database_id,
                dialect=example.dialect,
                query_type=example.pattern.query_type if example.pattern else "simple",
                complexity_score=example.pattern.complexity_score if example.pattern else 0.0
            )

            if example_id:
                stats["stored"] += 1
                source = query.source
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

                # Track query types
                query_type = example.pattern.query_type if example.pattern else "simple"
                stats["by_type"][query_type] = stats["by_type"].get(query_type, 0) + 1
            else:
                stats["duplicates"] += 1

        except Exception as e:
            stats["errors"] += 1

    print("\n")
    print("\n[Step 6] Query Learning Results:")
    print(f"  Harvested:   {stats['harvested']}")
    print(f"  Analyzed:    {stats['analyzed']}")
    print(f"  Stored:      {stats['stored']}")
    print(f"  Duplicates:  {stats['duplicates']}")
    print(f"  Errors:      {stats['errors']}")

    if stats["by_source"]:
        print("\n  By Source:")
        for source, count in stats["by_source"].items():
            print(f"    - {source}: {count}")

    if stats["by_type"]:
        print("\n  By Query Type:")
        for qtype, count in sorted(stats["by_type"].items()):
            print(f"    - {qtype}: {count}")

    # Get total examples in store
    all_examples = knowledge_store.get_examples_by_tables([], limit=10000)
    stats["total_examples"] = len(all_examples)
    print(f"\n  Total examples in knowledge base: {stats['total_examples']}")

    return stats


def show_statistics():
    """Show knowledge store statistics."""
    store, store_type = get_knowledge_store()
    stats = store.get_statistics()

    print("\n" + "=" * 50)
    print("KNOWLEDGE STORE STATISTICS")
    print("=" * 50)
    print(f"\nStorage: {stats['storage_path']}")
    print(f"Last training: {stats['last_training'] or 'Never'}")

    print("\n📊 Data Summary:")
    print(f"   Databases:       {stats.get('databases', 0)}")
    print(f"   Schemas:         {stats.get('schemas', 0)}")
    print(f"   Datasets:        {stats['datasets']}")
    print(f"   Total columns:   {stats.get('total_columns', 0)}")

    print("\n🔗 Relationships:")
    print(f"   Join patterns:   {stats['join_patterns']}")
    print(f"   Synonyms:        {stats['synonyms']}")
    print(f"   Column patterns: {stats['column_patterns']}")

    # Show database info if available
    db_info = store.get_metadata("databases")
    if db_info:
        try:
            databases = json.loads(db_info)
            print(f"\n📦 Databases:")
            for db_id, info in databases.items():
                schema_count = len(info.get('schemas', []))
                print(f"   • {info['name']} (ID: {db_id}, Backend: {info['backend']}, Schemas: {schema_count})")
        except json.JSONDecodeError:
            pass

    # Show query learning statistics (query examples are stored in SQLite)
    print("\n📝 Query Learning:")
    try:
        from learning.knowledge_store import KnowledgeStore
        sqlite_store = KnowledgeStore()
        examples = sqlite_store.get_examples_by_tables([], limit=10000)
        total_examples = len(examples)
        print(f"   Query examples: {total_examples}")

        if total_examples > 0:
            # Count by source
            by_source = {}
            by_type = {}
            for ex in examples:
                source = ex.get("source", "unknown")
                by_source[source] = by_source.get(source, 0) + 1
                qtype = ex.get("query_type", "simple")
                by_type[qtype] = by_type.get(qtype, 0) + 1

            if by_source:
                print("   By source:")
                for source, count in sorted(by_source.items()):
                    print(f"      - {source}: {count}")

            if by_type:
                print("   By query type:")
                for qtype, count in sorted(by_type.items()):
                    print(f"      - {qtype}: {count}")
    except Exception as e:
        print(f"   (Could not fetch query stats: {e})")

    print("\n" + "=" * 50)


def list_learned_datasets():
    """List all learned datasets grouped by database and schema."""
    from collections import defaultdict

    store, _ = get_knowledge_store()
    datasets = store.get_all_datasets()

    if not datasets:
        print("No datasets learned yet. Run training first:")
        print("  python train_semantic_matcher.py")
        return

    # Group by database_id and schema
    organized = defaultdict(lambda: defaultdict(list))
    for ds in datasets:
        db_id = ds.database_id or 0
        schema = ds.schema or "(default)"
        organized[db_id][schema].append(ds)

    # Get database names
    db_info = store.get_metadata("databases")
    db_names = {}
    if db_info:
        try:
            db_data = json.loads(db_info)
            db_names = {int(k): v.get('name', f'Database {k}') for k, v in db_data.items()}
        except json.JSONDecodeError:
            pass

    print("\n" + "=" * 60)
    print("LEARNED DATASETS")
    print("=" * 60)

    total_tables = 0
    for db_id, schemas in sorted(organized.items()):
        db_name = db_names.get(db_id, f"Database {db_id}")
        db_table_count = sum(len(tables) for tables in schemas.values())
        total_tables += db_table_count

        print(f"\n📦 {db_name} (ID: {db_id})")
        print(f"   {len(schemas)} schema(s), {db_table_count} table(s)")

        for schema_name, tables in sorted(schemas.items()):
            print(f"\n   📁 {schema_name}")
            for ds in sorted(tables, key=lambda x: x.table_name):
                col_count = len(ds.columns)
                id_cols = [c for c, info in ds.columns.items()
                          if info.is_primary_key or info.is_foreign_key]
                print(f"      └── {ds.table_name} ({col_count} cols, {len(id_cols)} keys)")

    print(f"\n" + "=" * 60)
    print(f"Total: {total_tables} tables across {len(organized)} database(s)")
    print("=" * 60)


def export_knowledge(filepath: str):
    """Export knowledge to JSON file."""
    store, _ = get_knowledge_store()
    store.export_to_json(filepath)
    print(f"Knowledge exported to: {filepath}")


def import_knowledge(filepath: str):
    """Import knowledge from JSON file."""
    store, _ = get_knowledge_store()
    store.import_from_json(filepath)
    print(f"Knowledge imported from: {filepath}")


def test_matcher():
    """Test the trained matcher with sample queries."""
    from nlu.semantic_matcher import create_trained_matcher

    print("Testing trained semantic matcher...")
    print("-" * 40)

    matcher = create_trained_matcher()
    stats = matcher.get_statistics()

    print(f"Loaded {stats.get('total_tables', 0)} tables")
    print(f"Loaded {stats.get('total_columns', 0)} columns")
    print(f"Loaded {stats.get('databases', 0)} databases")
    print(f"Loaded {len(stats.get('schemas', []))} schemas")
    print(f"Loaded {stats.get('join_patterns', 0)} join patterns")
    print(f"Loaded {stats.get('synonyms', 0)} synonyms")
    print()

    # Test queries
    test_entities = ["assets", "events", "users", "orders", "sessions"]

    for entity in test_entities:
        matches = matcher.match_table(entity)
        print(f"Entity: '{entity}'")

        if matches:
            top_match = matches[0]
            print(f"  -> Best match: {top_match.table_name} (score: {top_match.score:.2f}, reason: {top_match.match_reason})")

            # Show table info
            table_info = matcher.get_table(top_match.table_name)
            if table_info:
                print(f"     Schema: {table_info.schema}")
                print(f"     Columns: {len(table_info.columns)}")

                # Show joinable columns
                joinable = matcher.get_joinable_columns(top_match.table_name)
                if joinable:
                    print(f"     Joinable columns: {', '.join(j['column'] for j in joinable[:5])}")

            if len(matches) > 1:
                print(f"  -> Other candidates:")
                for m in matches[1:4]:
                    print(f"     - {m.table_name} (score: {m.score:.2f})")
        else:
            print(f"  -> No matches found")

        # Show join paths for the matched table
        if matches:
            top_match = matches[0]
            join_paths = matcher.find_join_path([top_match.table_name], max_depth=1)
            if join_paths:
                print(f"  -> Join paths:")
                for jp in join_paths[:2]:
                    print(f"     - {jp.left_table}.{jp.left_column} -> {jp.right_table}.{jp.right_column}")

        print()


def show_schema():
    """Show full schema information loaded in the semantic matcher."""
    from nlu.semantic_matcher import create_trained_matcher

    matcher = create_trained_matcher()
    matcher.print_schema_summary()


def clear_knowledge():
    """Clear all knowledge (with confirmation)."""
    response = input("Are you sure you want to clear all knowledge? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    store, _ = get_knowledge_store()
    store.clear_all()
    print("Knowledge store cleared.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train the semantic matcher and query learning system with Superset knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_semantic_matcher.py              # Full training (datasets + queries)
  python train_semantic_matcher.py --quick      # Quick training (no samples)
  python train_semantic_matcher.py --queries-only   # Train only query learning
  python train_semantic_matcher.py --skip-queries   # Skip query learning
  python train_semantic_matcher.py --stats      # Show statistics
  python train_semantic_matcher.py --list       # List learned datasets
  python train_semantic_matcher.py --test       # Test the trained matcher
  python train_semantic_matcher.py --export k.json  # Export to JSON
  python train_semantic_matcher.py --import k.json  # Import from JSON
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training without fetching sample values"
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Train only query learning (skip dataset discovery)"
    )
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Skip query learning (only run dataset discovery)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show knowledge store statistics"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all learned datasets grouped by database and schema"
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Show full schema information (tables, columns, keys)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the trained matcher with sample queries"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export knowledge to JSON file"
    )
    parser.add_argument(
        "--import",
        dest="import_file",
        type=str,
        metavar="FILE",
        help="Import knowledge from JSON file"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all knowledge (with confirmation)"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.stats:
        show_statistics()
        return

    if args.list:
        list_learned_datasets()
        return

    if args.schema:
        show_schema()
        return

    if args.test:
        test_matcher()
        return

    if args.export:
        export_knowledge(args.export)
        return

    if args.import_file:
        import_knowledge(args.import_file)
        return

    if args.clear:
        clear_knowledge()
        return

    # Default: run training
    print_banner()
    check_environment()

    try:
        # Handle queries-only mode
        if args.queries_only:
            print("Running query learning only (skipping dataset discovery)...")
            query_stats = train_query_learning(
                include_saved_queries=True,
                include_charts=True,
                include_datasets=True
            )
            print("\n" + "=" * 60)
            print("Query Learning Complete!")
            print("=" * 60)
            print(f"Total examples in knowledge base: {query_stats.get('total_examples', 0)}")
        else:
            # Full training (dataset discovery + query learning)
            stats = train_full(
                fetch_samples=not args.quick,
                skip_queries=args.skip_queries
            )

            # Also run a quick test
            print("\n")
            test_matcher()

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE - READY FOR PRODUCTION")
            print("=" * 60)

            # Show comprehensive summary
            print("\n📊 Dataset Discovery:")
            print(f"   Datasets processed: {stats.get('datasets_successful', 0)}/{stats.get('datasets_processed', 0)}")
            print(f"   Join patterns: {stats.get('join_patterns_found', 0)}")
            print(f"   Synonyms: {stats.get('synonyms_created', 0)}")

            if 'query_learning' in stats:
                ql = stats['query_learning']
                print(f"\n📝 Query Learning:")
                print(f"   Harvested: {ql.get('harvested', 0)}")
                print(f"   Stored: {ql.get('stored', 0)}")
                print(f"   Total examples: {ql.get('total_examples', 0)}")
                if ql.get('by_source'):
                    print(f"   By source: {ql.get('by_source')}")
                if ql.get('by_type'):
                    print(f"   By type: {ql.get('by_type')}")

            print("\n🔗 Integration Points:")
            print("   ✓ Semantic config updated: config/semantic_config.json")
            print("   ✓ Knowledge store updated: SQLite/PostgreSQL")
            print("   ✓ Query examples stored for few-shot learning")
            print("\n🚀 The chatbot API is now ready at: http://localhost:5050")
            print("   POST /api/chat - Natural language query")
            print("   GET  /api/query-learning-stats - View learning statistics")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()