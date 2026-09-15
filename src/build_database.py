"""Build a local SQLite database from the cleaned HDB resale CSV."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.csv"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "hdb_resale.db"
TABLE_NAME = "resale_transactions"

REQUIRED_COLUMNS = {
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
    "year",
    "month_number",
    "price_per_sqm",
    "flat_age",
    "storey_mid",
    "remaining_lease_months",
}

SQLITE_TYPES = {
    "month": "TEXT",
    "town": "TEXT",
    "flat_type": "TEXT",
    "block": "TEXT",
    "street_name": "TEXT",
    "storey_range": "TEXT",
    "floor_area_sqm": "REAL",
    "flat_model": "TEXT",
    "lease_commence_date": "INTEGER",
    "remaining_lease": "TEXT",
    "resale_price": "REAL",
    "year": "INTEGER",
    "month_number": "INTEGER",
    "price_per_sqm": "REAL",
    "flat_age": "INTEGER",
    "storey_mid": "REAL",
    "remaining_lease_months": "INTEGER",
}


def build_database(input_path: Path, database_path: Path) -> int:
    """Replace the analytics table and return its verified row count."""
    df = pd.read_csv(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Clean data is missing required columns: " + ", ".join(missing_columns)
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            df.to_sql(
                TABLE_NAME,
                connection,
                if_exists="replace",
                index=False,
                dtype=SQLITE_TYPES,
                chunksize=500,
                method="multi",
            )
            connection.executescript(
                f"""
                CREATE INDEX idx_resale_year
                    ON {TABLE_NAME} (year);
                CREATE INDEX idx_resale_town
                    ON {TABLE_NAME} (town);
                CREATE INDEX idx_resale_flat_type
                    ON {TABLE_NAME} (flat_type);
                CREATE INDEX idx_resale_town_year
                    ON {TABLE_NAME} (town, year);
                """
            )
            inserted_rows = connection.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME}"
            ).fetchone()[0]

    if inserted_rows != len(df):
        raise RuntimeError(
            f"Database row count mismatch: expected {len(df)}, got {inserted_rows}"
        )
    return inserted_rows


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Clean CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"SQLite path (default: {DEFAULT_DATABASE_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Build the local analytics database."""
    args = parse_args()
    inserted_rows = build_database(args.input, args.database)
    print(f"Loaded {inserted_rows:,} rows into {TABLE_NAME}")
    print(f"Database: {args.database}")


if __name__ == "__main__":
    main()
