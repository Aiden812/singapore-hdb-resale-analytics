"""Focused tests for the advanced descriptive and robustness analysis."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.advanced_analysis import (
    RPI_DATASET_ID,
    annual_price_quantiles,
    download_rpi_dataset,
    duplicate_removal_sensitivity,
    million_dollar_share,
    normalise_rpi_data,
    parse_remaining_lease_months,
    partial_month_disclosure,
    prepare_transactions,
    quarterly_rpi_benchmark,
    reconcile_python_sql,
    remaining_lease_bands,
    run_analysis,
    sha256_file,
    town_endpoint_growth,
)


def transaction(
    month: str,
    *,
    town: str = "ALPHA",
    price: float = 500_000,
    lease: str = "70 years",
) -> dict[str, object]:
    """Return the minimal valid transaction used by this test module."""
    return {
        "month": month,
        "town": town,
        "remaining_lease": lease,
        "resale_price": price,
    }


def two_complete_years() -> pd.DataFrame:
    """Return two full calendar years for two towns."""
    rows: list[dict[str, object]] = []
    for year, alpha_price, beta_price in [
        (2020, 400_000, 600_000),
        (2021, 500_000, 660_000),
    ]:
        for month in range(1, 13):
            rows.append(
                transaction(
                    f"{year}-{month:02d}",
                    town="ALPHA",
                    price=alpha_price + month,
                    lease="70 years 06 months",
                )
            )
            rows.append(
                transaction(
                    f"{year}-{month:02d}",
                    town="BETA",
                    price=beta_price + month,
                    lease="64 years",
                )
            )
    return pd.DataFrame(rows)


class LeaseParserTests(unittest.TestCase):
    def test_parses_year_and_month_variants(self) -> None:
        self.assertEqual(parse_remaining_lease_months("61 years 04 months"), 736)
        self.assertEqual(parse_remaining_lease_months("1 year 01 month"), 13)
        self.assertEqual(parse_remaining_lease_months("70 years"), 840)

    def test_rejects_invalid_month_component(self) -> None:
        with self.assertRaisesRegex(ValueError, "month component"):
            parse_remaining_lease_months("61 years 12 months")

    def test_rejects_unrecognised_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid remaining_lease"):
            parse_remaining_lease_months("about sixty years")


class DescriptiveAnalysisTests(unittest.TestCase):
    def test_discloses_latest_month_as_provisional_from_matching_manifest(self) -> None:
        frame = pd.DataFrame(
            [transaction("2021-02"), transaction("2021-03"), transaction("2021-03")]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "clean.csv"
            frame.to_csv(input_path, index=False)
            digest = sha256_file(input_path)
            (input_path.parent / "snapshot_metadata.json").write_text(
                json.dumps(
                    {
                        "processed_sha256": digest,
                        "source_modified_at_utc": "2021-02-28T16:30:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            disclosure = partial_month_disclosure(
                input_path,
                digest,
                prepare_transactions(frame),
            )

        self.assertTrue(disclosure["latest_month_is_provisional"])
        self.assertEqual(disclosure["latest_month"], "2021-03")
        self.assertEqual(disclosure["provisional_rows"], 2)

    def test_annual_quantiles_disclose_partial_year(self) -> None:
        frame = pd.DataFrame(
            [
                transaction("2020-01", price=100),
                transaction("2020-02", price=200),
                transaction("2021-01", price=1_000_000),
            ]
        )
        result = annual_price_quantiles(frame)

        self.assertEqual(result["year"].tolist(), [2020, 2021])
        self.assertEqual(result["p50"].tolist(), [150.0, 1_000_000.0])
        self.assertEqual(result["months_observed"].tolist(), [2, 1])
        self.assertFalse(result["is_complete_year"].any())

    def test_million_dollar_share_by_year_and_town(self) -> None:
        frame = pd.DataFrame(
            [
                transaction("2020-01", town="A", price=999_999),
                transaction("2020-02", town="A", price=1_000_000),
                transaction("2020-03", town="B", price=1_500_000),
            ]
        )
        result = million_dollar_share(frame, ["year", "town"])

        alpha = result.loc[result["town"].eq("A")].iloc[0]
        self.assertEqual(alpha["million_dollar_transactions"], 1)
        self.assertEqual(alpha["transaction_count"], 2)
        self.assertAlmostEqual(alpha["million_dollar_share_pct"], 50.0)

    def test_remaining_lease_bands_include_counts_and_median(self) -> None:
        frame = pd.DataFrame(
            [
                transaction("2020-01", price=400_000, lease="60 years"),
                transaction("2020-02", price=500_000, lease="64 years 11 months"),
                transaction("2020-03", price=800_000, lease="65 years"),
            ]
        )
        result = remaining_lease_bands(frame, bin_years=5)

        self.assertEqual(
            result["remaining_lease_band"].tolist(), ["60-64 years", "65-69 years"]
        )
        self.assertEqual(result["transaction_count"].tolist(), [2, 1])
        self.assertEqual(result["median_resale_price"].tolist(), [450_000, 800_000])

    def test_town_growth_enforces_endpoint_sample_guard(self) -> None:
        frame = two_complete_years()
        # Leave BETA below a two-record endpoint minimum in the first year.
        beta_first = (pd.to_datetime(frame["month"]).dt.year.eq(2020)) & frame[
            "town"
        ].eq("BETA")
        frame = frame.loc[~beta_first | frame["month"].eq("2020-01")]

        result = town_endpoint_growth(frame, min_sample=2)
        alpha = result.loc[result["town"].eq("ALPHA")].iloc[0]
        beta = result.loc[result["town"].eq("BETA")].iloc[0]

        self.assertTrue(alpha["meets_minimum_sample"])
        self.assertGreater(alpha["median_price_growth_pct"], 20)
        self.assertFalse(beta["meets_minimum_sample"])
        self.assertTrue(pd.isna(beta["median_price_growth_pct"]))
        self.assertEqual(beta["first_year_count"], 1)

    def test_duplicate_removal_is_reported_as_sensitivity(self) -> None:
        duplicated = transaction("2020-01", price=100)
        frame = pd.DataFrame(
            [duplicated, duplicated.copy(), transaction("2020-02", price=400)]
        )
        annual, overall = duplicate_removal_sensitivity(frame)

        self.assertEqual(overall["rows_removed"], 1)
        self.assertEqual(overall["median_with_duplicates"], 100)
        self.assertEqual(overall["median_without_duplicates"], 250)
        self.assertEqual(annual.loc[0, "rows_removed"], 1)


class BenchmarkTests(unittest.TestCase):
    def test_normalises_and_sorts_official_rpi(self) -> None:
        result = normalise_rpi_data(
            pd.DataFrame({"Quarter": ["2020-Q2", "2020-Q1"], "Index": [132.0, 130.0]})
        )
        self.assertEqual(result["quarter"].tolist(), ["2020-Q1", "2020-Q2"])
        self.assertEqual(result["index"].tolist(), [130.0, 132.0])

    def test_quarterly_benchmark_rebases_both_series(self) -> None:
        transactions = pd.DataFrame(
            [
                transaction("2020-01", price=400_000),
                transaction("2020-04", price=440_000),
            ]
        )
        rpi = pd.DataFrame({"quarter": ["2020-Q1", "2020-Q2"], "index": [100.0, 105.0]})
        result = quarterly_rpi_benchmark(transactions, rpi)

        self.assertEqual(result.loc[0, "base_quarter"], "2020-Q1")
        self.assertEqual(result.loc[0, "raw_median_rebased"], 100)
        self.assertEqual(result.loc[0, "official_rpi_rebased"], 100)
        self.assertAlmostEqual(result.loc[1, "raw_median_rebased"], 110)
        self.assertAlmostEqual(result.loc[1, "official_rpi_rebased"], 105)

    def test_downloads_paginated_rpi_snapshot(self) -> None:
        first_response = Mock()
        first_response.json.return_value = {
            "success": True,
            "result": {
                "total": 2,
                "records": [{"_id": 1, "quarter": "2020-Q1", "index": "100"}],
            },
        }
        second_response = Mock()
        second_response.json.return_value = {
            "success": True,
            "result": {
                "total": 2,
                "records": [{"_id": 2, "quarter": "2020-Q2", "index": "102"}],
            },
        }
        session = Mock()
        session.get.side_effect = [first_response, second_response]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "rpi.csv"
            rows = download_rpi_dataset(output, page_size=1, session=session)
            saved = pd.read_csv(output)

        self.assertEqual(rows, 2)
        self.assertEqual(saved["quarter"].tolist(), ["2020-Q1", "2020-Q2"])
        self.assertEqual(session.get.call_count, 2)
        first_response.raise_for_status.assert_called_once()
        second_response.raise_for_status.assert_called_once()

    def test_reconciles_python_and_sql_annual_medians(self) -> None:
        frame = pd.DataFrame(
            [
                transaction("2020-01", price=100),
                transaction("2020-02", price=300),
                transaction("2021-01", price=500),
            ]
        )
        database_frame = frame.assign(year=pd.to_datetime(frame["month"]).dt.year)[
            ["year", "resale_price"]
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "test.db"
            with closing(sqlite3.connect(database_path)) as connection:
                database_frame.to_sql("resale_transactions", connection, index=False)
                connection.commit()
            result = reconcile_python_sql(frame, database_path)

        self.assertTrue(result["matches_within_tolerance"].all())
        self.assertEqual(result["absolute_difference_sgd"].tolist(), [0, 0])


class EndToEndTests(unittest.TestCase):
    def test_run_analysis_writes_reproducible_artifacts_and_metadata(self) -> None:
        frame = two_complete_years()
        rpi = pd.DataFrame(
            {
                "quarter": [
                    f"{year}-Q{quarter}"
                    for year in (2020, 2021)
                    for quarter in range(1, 5)
                ],
                "index": [100, 101, 102, 103, 104, 105, 106, 107],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "clean.csv"
            rpi_path = root / "rpi.csv"
            reports = root / "reports"
            images = root / "images"
            frame.to_csv(input_path, index=False)
            rpi.to_csv(rpi_path, index=False)

            metadata = run_analysis(
                input_path,
                reports_dir=reports,
                images_dir=images,
                database_path=None,
                rpi_input_path=rpi_path,
                min_town_sample=2,
            )

            saved_metadata = json.loads(
                (reports / "advanced_analysis_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue((reports / "advanced_analysis_report.md").exists())
            self.assertTrue((reports / "annual_price_quantiles.csv").exists())
            self.assertTrue((reports / "quarterly_rpi_benchmark.csv").exists())
            self.assertTrue((images / "raw_median_vs_official_rpi.png").exists())
            self.assertEqual(
                saved_metadata["official_rpi"]["dataset_id"], RPI_DATASET_ID
            )
            self.assertEqual(metadata["transaction_source"]["row_count"], len(frame))


if __name__ == "__main__":
    unittest.main()
