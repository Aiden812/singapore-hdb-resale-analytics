"""Focused tests for the Streamlit dashboard calculations."""

from __future__ import annotations

import unittest

import pandas as pd

from src.dashboard_data import (
    DashboardDataError,
    annual_trend,
    binned_price_profile,
    coverage_by_year,
    filter_transactions,
    parse_remaining_lease_months,
    prepare_dashboard_data,
)


def sample_data() -> pd.DataFrame:
    """Return a small compatible transaction extract."""
    return pd.DataFrame(
        {
            "month": ["2024-01", "2024-12", "2025-01", "2025-02"],
            "town": ["TAMPINES", "TAMPINES", "BEDOK", "TAMPINES"],
            "flat_type": ["4 ROOM", "5 ROOM", "4 ROOM", "4 ROOM"],
            "storey_range": ["04 TO 06", "10 TO 12", "07 TO 09", "04 TO 06"],
            "floor_area_sqm": [100, 120, 90, 100],
            "remaining_lease": [
                "74 years 03 months",
                "70 years",
                "65 years 11 months",
                "73 years 06 months",
            ],
            "resale_price": [600_000, 1_100_000, 540_000, 1_000_000],
        }
    )


class DashboardDataTests(unittest.TestCase):
    def test_parses_remaining_lease_variants(self) -> None:
        parsed = parse_remaining_lease_months(
            pd.Series(["74 years 03 months", "70 years", "65", None])
        )
        self.assertEqual(parsed.iloc[0], 891)
        self.assertEqual(parsed.iloc[1], 840)
        self.assertEqual(parsed.iloc[2], 780)
        self.assertTrue(pd.isna(parsed.iloc[3]))

    def test_prepares_dashboard_fields(self) -> None:
        prepared = prepare_dashboard_data(sample_data())

        self.assertEqual(prepared["year"].tolist(), [2024, 2024, 2025, 2025])
        self.assertEqual(prepared.loc[0, "storey_mid"], 5)
        self.assertEqual(prepared.loc[0, "price_per_sqm"], 6_000)
        self.assertEqual(
            prepared["is_million_dollar"].tolist(), [False, True, False, True]
        )

    def test_filters_and_computes_annual_statistics(self) -> None:
        prepared = prepare_dashboard_data(sample_data())
        filtered = filter_transactions(
            prepared,
            years=[2025],
            towns=["TAMPINES"],
            flat_types=["4 ROOM"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["resale_price"], 1_000_000)

        annual = annual_trend(prepared)
        self.assertEqual(annual.loc[annual["year"].eq(2025), "transactions"].iloc[0], 2)
        self.assertAlmostEqual(
            annual.loc[annual["year"].eq(2025), "million_dollar_share"].iloc[0],
            0.5,
        )

    def test_flags_incomplete_years(self) -> None:
        prepared = prepare_dashboard_data(sample_data())
        coverage = coverage_by_year(prepared)
        self.assertTrue(coverage["is_partial"].all())
        self.assertEqual(
            coverage.loc[coverage["year"].eq(2025), "months_observed"].iloc[0], 2
        )

    def test_binned_profile_preserves_transaction_count(self) -> None:
        prepared = prepare_dashboard_data(sample_data())
        profile = binned_price_profile(prepared, "floor_area_sqm", bin_width=10)
        self.assertEqual(int(profile["transactions"].sum()), len(prepared))
        self.assertIn("median_price_per_sqm", profile.columns)

    def test_rejects_missing_schema_and_bad_numeric_values(self) -> None:
        with self.assertRaisesRegex(
            DashboardDataError, "missing required columns: town"
        ):
            prepare_dashboard_data(sample_data().drop(columns="town"))
        with self.assertRaisesRegex(DashboardDataError, "positive values"):
            prepare_dashboard_data(sample_data().assign(resale_price=0))


if __name__ == "__main__":
    unittest.main()
