"""Project Meezan — database migration script (docs/FRD.md, Section 5).

Applies db/schema.sql to a SQLite database. Any existing tables in the
target file are dropped first, so this is safe to run against a fresh
database or one still carrying the prototype's old schema.
"""

import argparse
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "meezan.db"


def _drop_existing_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    for (table,) in cursor.fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')


def migrate(db_path: Path) -> None:
    schema_sql = SCHEMA_PATH.read_text()
    conn = sqlite3.connect(db_path)
    try:
        _drop_existing_tables(conn)
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the Project Meezan DB schema.")
    parser.add_argument(
        "db_path",
        nargs="?",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    migrate(Path(args.db_path))
    print(f"Migrated schema -> {args.db_path}")


if __name__ == "__main__":
    main()
