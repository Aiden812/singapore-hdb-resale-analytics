"""Tests for deterministic, citation-ready copilot evidence packets."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.copilot_tools import (
    EvidenceInputError,
    build_comparable_sales_evidence,
    build_market_summary_evidence,
    build_model_diagnostics_evidence,
    build_town_comparison_evidence,
)


def market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": [
                "2024-01",
                "2024-12",
                "2025-01",
                "2025-02",
                "2025-03",
                "2025-04",
                "2025-05",
            ],
            "town": [
                "TAMPINES",
                "TAMPINES",
                "TAMPINES",
                "TAMPINES",
                "BEDOK",
                "BEDOK",
                "BEDOK",
            ],
            "flat_type": ["4 ROOM"] * 7,
            "floor_area_sqm": [100] * 7,
            "resale_price": [
                500_000,
                600_000,
                700_000,
                1_100_000,
                400_000,
                600_000,
                800_000,
            ],
        }
    )


def comparable_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": [
                "2025-07",
                "2025-06",
                "2025-05",
                "2025-04",
                "2025-03",
                "2025-02",
                "2025-01",
            ],
            "town": ["TAMPINES"] * 7,
            "flat_type": ["4 ROOM"] * 7,
            "block": list("ABCDEFG"),
            "street_name": ["TEST STREET"] * 7,
            "floor_area_sqm": [100, 100, 101, 100, 100, 102, 104],
            "storey_mid": [8, 8, 8, 8, 9, 8, 8],
            "remaining_lease_months": [
                72 * 12,
                72 * 12,
                72 * 12,
                73 * 12,
                72 * 12,
                72 * 12,
                72 * 12,
            ],
            "resale_price": [
                700_000,
                690_000,
                680_000,
                670_000,
                660_000,
                650_000,
                640_000,
            ],
        }
    )


def model_metrics() -> dict[str, object]:
    return {
        "holdout_mae_improvement_vs_baseline_pct": 74.4,
        "holdout_mae_improvement_vs_prior_12m_segment_baseline_pct": 35.5,
        "holdout_metrics": {
            "hedonic_ridge": {
                "mae": 52_844.25,
                "rmse": 69_635.5,
                "r2": 0.893,
                "mape_pct": 8.225,
            }
        },
        "holdout_prediction_interval": {
            "empirical_coverage": 0.75,
            "target_coverage": 0.8,
            "median_interval_width": 148_136.0,
        },
        "rolling_backtests": {
            "folds": 4,
            "mean_empirical_interval_coverage": 0.707,
            "metrics_by_model": {"hedonic_ridge": {"mean_mae": 52_347.5}},
        },
        "index_summary": {
            "base_month": "2017-01",
            "latest_month": "2026-08",
            "quality_adjusted_change_from_base_pct": 56.6,
        },
        "split": {
            "holdout_start_month": "2025-09",
            "holdout_end_month": "2026-08",
            "holdout_rows": 24_592,
        },
        "input_provenance": {
            "source_month_min": "2017-01",
            "source_month_max": "2026-09",
            "source_rows": 239_977,
            "sha256": "abc123",
        },
    }


def fact(packet: dict[str, object], fact_id: str) -> dict[str, object]:
    facts = packet["facts"]
    assert isinstance(facts, list)
    return next(item for item in facts if item["id"] == fact_id)


class CopilotEvidenceTests(unittest.TestCase):
    def assert_packet_is_citable_and_json_safe(
        self, packet: dict[str, object]
    ) -> None:
        json.dumps(packet, allow_nan=False)
        self.assertEqual(packet["schema_version"], "1.0")
        evidence_ids = {row["id"] for row in packet["evidence_rows"]}
        source_ids = {source["id"] for source in packet["sources"]}
        for item in packet["facts"]:
            self.assertRegex(item["id"], r"^fact\.")
            self.assertIn("raw_value", item)
            self.assertIn("display_value", item)
            self.assertIn("unit", item)
            self.assertTrue(set(item["evidence_ids"]).issubset(evidence_ids))
            self.assertTrue(set(item["source_ids"]).issubset(source_ids))

    def test_market_summary_has_exact_filters_numbers_and_citations(self) -> None:
        packet = build_market_summary_evidence(
            market_data(),
            towns="tampines",
            flat_types=["4 room"],
            start_month="2024-01-20",
            end_month="2025-02",
            provenance={"observed_at": pd.Timestamp("2026-09-08")},
        )

        self.assertEqual(packet["status"], "ok")
        self.assertEqual(packet["filters"]["towns"], ["TAMPINES"])
        self.assertEqual(packet["filters"]["flat_types"], ["4 ROOM"])
        self.assertEqual(packet["filters"]["start_month"], "2024-01")
        self.assertEqual(packet["snapshot"]["rows"], 7)
        self.assertEqual(
            fact(packet, "fact.market.transaction_count")["raw_value"], 4
        )
        self.assertEqual(
            fact(packet, "fact.market.median_resale_price")["raw_value"],
            650_000.0,
        )
        self.assertEqual(
            fact(packet, "fact.market.price_q25")["raw_value"], 575_000.0
        )
        self.assertEqual(
            fact(packet, "fact.market.price_q75")["raw_value"], 800_000.0
        )
        self.assertEqual(
            fact(packet, "fact.market.million_dollar_share")["raw_value"], 0.25
        )
        self.assertEqual(
            fact(
                packet, "fact.market.first_to_latest_monthly_median_change"
            )["raw_value"],
            120.00000000000001,
        )
        self.assertEqual(
            fact(packet, "fact.market.latest_observed_month")["evidence_ids"],
            ["row.market.endpoint.latest"],
        )
        rows = {row["id"]: row for row in packet["evidence_rows"]}
        self.assertEqual(
            rows["row.market.endpoint.first"]["median_resale_price_sgd"],
            500_000.0,
        )
        self.assertEqual(
            rows["row.market.endpoint.latest"]["median_resale_price_sgd"],
            1_100_000.0,
        )
        self.assertEqual(
            fact(packet, "fact.market.latest_month_median_price")["raw_value"],
            1_100_000.0,
        )
        self.assert_packet_is_citable_and_json_safe(packet)

    def test_market_summary_returns_insufficient_for_empty_or_ood_selection(self) -> None:
        out_of_domain = build_market_summary_evidence(
            market_data(), towns=["MARS CENTRAL"]
        )
        empty = build_market_summary_evidence(pd.DataFrame())

        self.assertEqual(out_of_domain["status"], "insufficient_evidence")
        self.assertEqual(
            fact(out_of_domain, "fact.market.transaction_count")["raw_value"], 0
        )
        self.assertEqual(empty["status"], "insufficient_evidence")
        self.assertEqual(empty["snapshot"]["rows"], 0)
        self.assert_packet_is_citable_and_json_safe(out_of_domain)
        self.assert_packet_is_citable_and_json_safe(empty)

    def test_market_summary_rejects_bad_schema_and_reversed_period(self) -> None:
        with self.assertRaisesRegex(EvidenceInputError, "missing required columns"):
            build_market_summary_evidence(market_data().drop(columns="town"))
        with self.assertRaisesRegex(EvidenceInputError, "cannot be after"):
            build_market_summary_evidence(
                market_data(), start_month="2025-02", end_month="2025-01"
            )

    def test_comparables_are_closest_five_and_stable_across_input_order(self) -> None:
        kwargs = {
            "town": "Tampines",
            "flat_type": "4 Room",
            "floor_area_sqm": 100,
            "storey_mid": 8,
            "remaining_lease_years": 72,
            "minimum_transactions": 5,
        }
        first = build_comparable_sales_evidence(comparable_data(), **kwargs)
        shuffled = build_comparable_sales_evidence(
            comparable_data().sample(frac=1, random_state=42), **kwargs
        )
        first_rows = first["evidence_rows"][1:]
        shuffled_rows = shuffled["evidence_rows"][1:]

        self.assertEqual(first["status"], "ok")
        self.assertEqual(len(first_rows), 5)
        self.assertEqual([row["block"] for row in first_rows], list("ABCDE"))
        self.assertEqual(first_rows[0]["distance_score"], 0.0)
        self.assertEqual(first_rows[1]["distance_score"], 0.0)
        self.assertEqual(
            [row["id"] for row in first_rows],
            [row["id"] for row in shuffled_rows],
        )
        self.assertEqual(
            fact(first, "fact.comparable.selected_median_price")["raw_value"],
            680_000.0,
        )
        self.assertEqual(
            fact(first, "fact.comparable.selected_price_q25")["raw_value"],
            670_000.0,
        )
        self.assertEqual(
            fact(first, "fact.comparable.selected_price_q75")["raw_value"],
            690_000.0,
        )
        self.assert_packet_is_citable_and_json_safe(first)

    def test_comparables_validate_inputs_and_expose_insufficient_evidence(self) -> None:
        packet = build_comparable_sales_evidence(
            comparable_data(),
            town="WOODLANDS",
            flat_type="4 ROOM",
            floor_area_sqm=100,
            storey_mid=8,
            remaining_lease_years=72,
        )
        self.assertEqual(packet["status"], "insufficient_evidence")
        self.assertEqual(len(packet["evidence_rows"]), 1)
        self.assertEqual(
            fact(packet, "fact.comparable.matching_pool_size")["raw_value"], 0
        )
        with self.assertRaisesRegex(EvidenceInputError, "must be positive"):
            build_comparable_sales_evidence(
                comparable_data(),
                town="TAMPINES",
                flat_type="4 ROOM",
                floor_area_sqm=-1,
            )
        with self.assertRaisesRegex(EvidenceInputError, "between 0 and 99"):
            build_comparable_sales_evidence(
                comparable_data(),
                town="TAMPINES",
                flat_type="4 ROOM",
                floor_area_sqm=100,
                remaining_lease_years=120,
            )
        self.assert_packet_is_citable_and_json_safe(packet)

    def test_town_comparison_has_exact_numbers_and_handles_ood_town(self) -> None:
        packet = build_town_comparison_evidence(
            market_data(), towns=["TAMPINES", "BEDOK"], flat_type="4 room"
        )
        thin = build_town_comparison_evidence(
            market_data(), towns=["TAMPINES", "ATLANTIS"], flat_type="4 room"
        )

        self.assertEqual(packet["status"], "ok")
        self.assertEqual(
            fact(packet, "fact.town.tampines.median_resale_price")["raw_value"],
            650_000.0,
        )
        self.assertEqual(
            fact(packet, "fact.town.bedok.median_resale_price")["raw_value"],
            600_000.0,
        )
        self.assertEqual(
            fact(
                packet, "fact.town.highest_to_lowest_median_price_gap"
            )["raw_value"],
            50_000.0,
        )
        self.assertEqual(
            fact(packet, "fact.town.highest_median_price_town")["raw_value"],
            "TAMPINES",
        )
        self.assertEqual(thin["status"], "insufficient_evidence")
        self.assert_packet_is_citable_and_json_safe(packet)
        self.assert_packet_is_citable_and_json_safe(thin)

    def test_town_comparison_requires_two_distinct_towns(self) -> None:
        with self.assertRaisesRegex(EvidenceInputError, "at least two"):
            build_town_comparison_evidence(
                market_data(), towns=["TAMPINES", "tampines"]
            )

    def test_model_diagnostics_preserve_exact_metrics_and_provenance(self) -> None:
        packet = build_model_diagnostics_evidence(model_metrics())

        self.assertEqual(packet["status"], "ok")
        self.assertEqual(packet["snapshot"]["rows"], 239_977)
        self.assertEqual(
            packet["snapshot"]["coverage"],
            {
                "start_month": "2017-01",
                "end_month": "2026-09",
                "months_observed": 117,
                "partial_years": [2026],
            },
        )
        self.assertEqual(
            fact(packet, "fact.model.holdout_mae")["raw_value"], 52_844.25
        )
        self.assertEqual(
            fact(packet, "fact.model.interval_empirical_coverage")[
                "display_value"
            ],
            "75.0%",
        )
        self.assertEqual(
            fact(
                packet,
                "fact.model.mae_improvement_vs_prior_12m_segment_baseline",
            )["display_value"],
            "35.5%",
        )
        self.assertTrue(any("below target" in text for text in packet["caveats"]))
        self.assert_packet_is_citable_and_json_safe(packet)

    def test_model_diagnostics_missing_metrics_are_insufficient_and_json_safe(
        self,
    ) -> None:
        packet = build_model_diagnostics_evidence({})

        self.assertEqual(packet["status"], "insufficient_evidence")
        self.assertEqual(packet["snapshot"]["rows"], 0)
        self.assertEqual(len(packet["evidence_rows"]), 4)
        self.assert_packet_is_citable_and_json_safe(packet)


if __name__ == "__main__":
    unittest.main()
