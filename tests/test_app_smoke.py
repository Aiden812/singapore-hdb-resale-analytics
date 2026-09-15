"""End-to-end smoke test for the portfolio Streamlit app."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=60,
        ).run(timeout=60)

    def test_app_has_no_uncaught_exceptions(self) -> None:
        self.assertEqual(list(self.app.exception), [])

    def test_app_exposes_the_five_portfolio_tabs(self) -> None:
        self.assertEqual(
            [tab.label for tab in self.app.tabs],
            [
                "Overview",
                "Market vs mix",
                "Price & mix",
                "Property profiles",
                "Data notes",
            ],
        )

    def test_app_renders_the_expected_chart_set(self) -> None:
        self.assertGreaterEqual(len(self.app.get("plotly_chart")), 10)

    def test_csv_fallback_is_recognised_as_the_model_snapshot(self) -> None:
        snapshot = json.loads(
            (PROJECT_ROOT / "data" / "processed" / "snapshot_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        transactions = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            clean_csv = Path(temporary_directory) / "hdb_resale_clean.csv"
            transactions.to_csv(
                clean_csv,
                index=False,
                date_format="%Y-%m-%d",
                lineterminator="\n",
            )
            digest = hashlib.sha256(clean_csv.read_bytes()).hexdigest()
            self.assertEqual(digest, snapshot["processed_sha256"])
            with patch.dict(os.environ, {"HDB_DATA_PATH": str(clean_csv)}):
                app = AppTest.from_file(
                    str(PROJECT_ROOT / "app.py"),
                    default_timeout=60,
                ).run(timeout=60)

        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "Adjusted growth since Jan 2017",
            [metric.label for metric in app.metric],
        )


if __name__ == "__main__":
    unittest.main()
