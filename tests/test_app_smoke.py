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

    def test_app_exposes_the_portfolio_tabs(self) -> None:
        self.assertEqual(
            [tab.label for tab in self.app.tabs],
            [
                "Overview",
                "Comparable sales",
                "Market vs mix",
                "Price & mix",
                "Property profiles",
                "Data notes",
            ],
        )

    def test_app_renders_the_expected_chart_set(self) -> None:
        self.assertGreaterEqual(len(self.app.get("plotly_chart")), 11)

    def test_app_exposes_comparable_sales_controls_and_guardrail(self) -> None:
        self.assertIn(
            "Recent period",
            [selectbox.label for selectbox in self.app.selectbox],
        )
        self.assertIn(
            "Price basis",
            [radio.label for radio in self.app.radio],
        )
        self.assertIn(
            "Comparable blocks and MRT context",
            [expander.label for expander in self.app.expander],
        )
        self.assertIn(
            "Download comparable transactions (CSV)",
            [button.label for button in self.app.get("download_button")],
        )
        self.assertIn(
            "Rolling coverage",
            [metric.label for metric in self.app.metric],
        )
        visible_text = " ".join(
            element.value
            for element_type in ("markdown", "warning")
            for element in self.app.get(element_type)
            if isinstance(element.value, str)
        )
        self.assertIn("not a formal valuation", visible_text)

    def test_inflation_adjusted_comparables_render_without_errors(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=60,
        ).run(timeout=60)
        price_basis = next(radio for radio in app.radio if radio.label == "Price basis")
        price_basis.set_value("Inflation-adjusted")
        monthly_metric = next(
            radio for radio in app.radio if radio.label == "Monthly trend metric"
        )
        monthly_metric.set_value("Median inflation-adjusted price")
        app.run(timeout=60)

        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "Observed median",
            [metric.label for metric in app.metric],
        )

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
