#!/usr/bin/env python3
"""
Database migration runner for superset-mcp.

Usage:
    python migrations/migrate.py              # Run all pending migrations
    python migrations/migrate.py --status     # Show migration status
    python migrations/migrate.py --rollback   # Rollback last migration (manual)

Environment variables:
    POSTGRES_HOST     - PostgreSQL host (default: localhost)
    POSTGRES_PORT     - PostgreSQL port (default: 5432)
    POSTGRES_DB       - Database name (default: superset_mcp)
    POSTGRES_USER     - Database user (default: postgres)
    POSTGRES_PASSWORD - Database password (required)
"""
import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


def get_db_config():
    """Get database configuration from environment."""
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "database": os.environ.get("POSTGRES_DB", "superset_mcp"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
    }


def get_connection(config=None, database=None):
    """Get a database connection."""
    if config is None:
        config = get_db_config()

    conn_params = config.copy()
    if database:
        conn_params["database"] = database

    return psycopg2.connect(**conn_params)


def ensure_database_exists():
    """Ensure the target database exists."""
    config = get_db_config()
    target_db = config["database"]

    # Connect to postgres database to check/create target database
    try:
        conn = get_connection(config, database="postgres")
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (target_db,)
        )
        exists = cursor.fetchone()

        if not exists:
            print(f"Creating database: {target_db}")
            cursor.execute(f'CREATE DATABASE "{target_db}"')
            print(f"  ✓ Database created")
        else:
            print(f"Database '{target_db}' already exists")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"ERROR: Failed to connect to PostgreSQL: {e}")
        return False


def get_applied_migrations(conn):
    """Get list of applied migration versions."""
    cursor = conn.cursor()

    # Check if migrations table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'superset_mcp'
            AND table_name = 'migrations'
        )
    """)

    if not cursor.fetchone()[0]:
        return set()

    cursor.execute("SELECT version FROM superset_mcp.migrations ORDER BY version")
    return {row[0] for row in cursor.fetchall()}


def get_pending_migrations(conn):
    """Get list of pending migration files."""
    migrations_dir = Path(__file__).parent
    migration_files = sorted(glob.glob(str(migrations_dir / "*.sql")))

    applied = get_applied_migrations(conn)
    pending = []

    for filepath in migration_files:
        filename = Path(filepath).name
        # Extract version from filename (e.g., "001_initial_schema.sql" -> "001")
        version = filename.split("_")[0]

        if version not in applied:
            pending.append({
                "version": version,
                "filename": filename,
                "filepath": filepath
            })

    return pending


def run_migration(conn, migration):
    """Run a single migration."""
    print(f"\n  Running migration: {migration['filename']}")

    with open(migration["filepath"], "r") as f:
        sql = f.read()

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        print(f"    ✓ Migration {migration['version']} applied successfully")
        return True
    except psycopg2.Error as e:
        conn.rollback()
        print(f"    ✗ Migration {migration['version']} failed: {e}")
        return False
    finally:
        cursor.close()


def run_all_migrations():
    """Run all pending migrations."""
    print("\n" + "=" * 60)
    print("SUPERSET-MCP DATABASE MIGRATION")
    print("=" * 60)

    config = get_db_config()
    print(f"\nDatabase: {config['host']}:{config['port']}/{config['database']}")
    print(f"User: {config['user']}")

    # Ensure database exists
    if not ensure_database_exists():
        return False

    # Connect to target database
    try:
        conn = get_connection()
    except psycopg2.Error as e:
        print(f"ERROR: Failed to connect to database: {e}")
        return False

    # Get pending migrations
    pending = get_pending_migrations(conn)

    if not pending:
        print("\n✓ All migrations are up to date!")
        conn.close()
        return True

    print(f"\nPending migrations: {len(pending)}")
    for m in pending:
        print(f"  - {m['filename']}")

    # Run migrations
    print("\nApplying migrations...")
    success_count = 0

    for migration in pending:
        if run_migration(conn, migration):
            success_count += 1
        else:
            print(f"\nMigration stopped due to error.")
            break

    conn.close()

    print("\n" + "=" * 60)
    print(f"Migrations complete: {success_count}/{len(pending)} applied")
    print("=" * 60)

    return success_count == len(pending)


def show_status():
    """Show migration status."""
    print("\n" + "=" * 60)
    print("MIGRATION STATUS")
    print("=" * 60)

    config = get_db_config()
    print(f"\nDatabase: {config['host']}:{config['port']}/{config['database']}")

    try:
        conn = get_connection()
    except psycopg2.Error as e:
        print(f"ERROR: Cannot connect to database: {e}")
        return

    applied = get_applied_migrations(conn)
    pending = get_pending_migrations(conn)

    print(f"\nApplied migrations: {len(applied)}")
    for version in sorted(applied):
        print(f"  ✓ {version}")

    print(f"\nPending migrations: {len(pending)}")
    for m in pending:
        print(f"  ○ {m['filename']}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Database migration runner for superset-mcp"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback last migration (prints SQL, does not execute)"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.rollback:
        print("Rollback is not automated. Please manually revert changes.")
        print("Check the migration files for the schema changes.")
    else:
        success = run_all_migrations()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()