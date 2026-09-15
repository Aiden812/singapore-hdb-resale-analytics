"""Tests for the HDB resale cleaning and SQLite pipeline."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.build_database import build_database
from src.clean_data import (
    clean_data,
    get_storey_midpoint,
    parse_remaining_lease_months,
    write_snapshot_metadata,
)
from src.download_data import validate_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_record(**overrides: object) -> dict[str, object]:
    """Return one valid source record with optional field overrides."""
    record: dict[str, object] = {
        "month": "2025-04",
        "town": " TEST TOWN ",
        "flat_type": "4 ROOM",
        "block": "1",
        "street_name": " TEST ROAD ",
        "storey_range": "04 TO 06",
        "floor_area_sqm": "100",
        "flat_model": " Model A ",
        "lease_commence_date": "2000",
        "remaining_lease": "74 years",
        "resale_price": "500000",
    }
    record.update(overrides)
    return record


class CleanDataTests(unittest.TestCase):
    def test_engineers_expected_values(self) -> None:
        clean_df = clean_data(pd.DataFrame([source_record()]))

        self.assertEqual(get_storey_midpoint("04 TO 06"), 5)
        self.assertEqual(clean_df.loc[0, "town"], "TEST TOWN")
        self.assertEqual(clean_df.loc[0, "year"], 2025)
        self.assertEqual(clean_df.loc[0, "month_number"], 4)
        self.assertEqual(clean_df.loc[0, "price_per_sqm"], 5000)
        self.assertEqual(clean_df.loc[0, "flat_age"], 25)
        self.assertEqual(clean_df.loc[0, "storey_mid"], 5)
        self.assertEqual(clean_df.loc[0, "remaining_lease_months"], 888)

    def test_parses_remaining_lease_with_and_without_months(self) -> None:
        self.assertEqual(parse_remaining_lease_months("74 years"), 888)
        self.assertEqual(parse_remaining_lease_months("74 years 03 months"), 891)

    def test_rejects_invalid_remaining_lease(self) -> None:
        df = pd.DataFrame([source_record(remaining_lease="74 years 12 months")])

        with self.assertRaisesRegex(ValueError, "Invalid remaining lease"):
            clean_data(df)

    def test_preserves_fully_identical_rows(self) -> None:
        record = source_record()
        clean_df = clean_data(pd.DataFrame([record, record]))

        self.assertEqual(len(clean_df), 2)
        self.assertEqual(int(clean_df.duplicated().sum()), 1)

    def test_rejects_missing_required_column(self) -> None:
        df = pd.DataFrame([source_record()]).drop(columns="town")

        with self.assertRaisesRegex(ValueError, "missing required columns: town"):
            clean_data(df)

    def test_rejects_invalid_month(self) -> None:
        df = pd.DataFrame([source_record(month="not-a-month")])

        with self.assertRaisesRegex(ValueError, "month=1"):
            clean_data(df)

    def test_rejects_invalid_storey_range(self) -> None:
        df = pd.DataFrame([source_record(storey_range="LEVEL FIVE")])

        with self.assertRaisesRegex(ValueError, "Invalid storey range"):
            clean_data(df)

    def test_rejects_nonpositive_floor_area(self) -> None:
        df = pd.DataFrame([source_record(floor_area_sqm="0")])

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            clean_data(df)

    def test_rejects_nonpositive_resale_price(self) -> None:
        df = pd.DataFrame([source_record(resale_price="0")])

        with self.assertRaisesRegex(ValueError, "resale_price"):
            clean_data(df)

    def test_rejects_fractional_lease_year(self) -> None:
        df = pd.DataFrame([source_record(lease_commence_date="2000.5")])

        with self.assertRaisesRegex(ValueError, "whole years"):
            clean_data(df)

    def test_rejects_blank_text(self) -> None:
        df = pd.DataFrame([source_record(town="   ")])

        with self.assertRaisesRegex(ValueError, "blank text values"):
            clean_data(df)


class DownloadValidationTests(unittest.TestCase):
    def test_accepts_expected_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "valid.csv"
            pd.DataFrame([source_record()]).to_csv(path, index=False)

            validate_download(path)

    def test_rejects_unexpected_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.csv"
            path.write_text("message\nnot a dataset\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                validate_download(path)

    def test_rejects_header_only_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "empty.csv"
            pd.DataFrame(columns=source_record().keys()).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "no data rows"):
                validate_download(path)


class SnapshotMetadataTests(unittest.TestCase):
    def test_writes_hashes_and_snapshot_coverage(self) -> None:
        clean_df = clean_data(pd.DataFrame([source_record()]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            raw_path = temporary_path / "raw.csv"
            clean_path = temporary_path / "clean.csv"
            metadata_path = temporary_path / "metadata.json"
            pd.DataFrame([source_record()]).to_csv(raw_path, index=False)
            clean_df.to_csv(clean_path, index=False)

            metadata = write_snapshot_metadata(
                raw_path=raw_path,
                clean_path=clean_path,
                metadata_path=metadata_path,
                clean_df=clean_df,
            )

            self.assertEqual(metadata["row_count"], 1)
            self.assertEqual(metadata["month_min"], "2025-04")
            self.assertEqual(metadata["month_max"], "2025-04")
            self.assertEqual(len(str(metadata["source_sha256"])), 64)
            self.assertTrue(metadata_path.exists())


class DatabaseTests(unittest.TestCase):
    def test_builds_indexed_database_and_runs_all_queries(self) -> None:
        records = [
            source_record(),
            source_record(
                month="2025-05",
                town="ANOTHER TOWN",
                block="2",
                floor_area_sqm="80",
                resale_price="400000",
            ),
        ]
        clean_df = clean_data(pd.DataFrame(records))

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            csv_path = temporary_path / "clean.csv"
            database_path = temporary_path / "test.db"
            clean_df.to_csv(csv_path, index=False)

            inserted_rows = build_database(csv_path, database_path)
            self.assertEqual(inserted_rows, len(clean_df))

            sql = (PROJECT_ROOT / "sql" / "analysis_queries.sql").read_text(
                encoding="utf-8"
            )
            statements = [part.strip() for part in sql.split(";") if part.strip()]
            self.assertEqual(len(statements), 9)

            with closing(sqlite3.connect(database_path)) as connection:
                self.assertEqual(
                    connection.execute("PRAGMA quick_check").fetchone()[0],
                    "ok",
                )
                index_names = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA index_list('resale_transactions')"
                    )
                }
                self.assertEqual(
                    index_names,
                    {
                        "idx_resale_year",
                        "idx_resale_town",
                        "idx_resale_flat_type",
                        "idx_resale_town_year",
                    },
                )
                for statement in statements:
                    connection.execute(statement).fetchall()


if __name__ == "__main__":
    unittest.main()
