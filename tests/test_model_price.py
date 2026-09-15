"""Focused tests for the quality-adjusted HDB price model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.model_price import (
    add_trend_features,
    chronological_holdout,
    derive_quality_adjusted_index,
    evaluate_chronological_holdout,
    evaluate_rolling_backtests,
    exclude_provisional_latest_month,
    expanding_window_splits,
    fit_calibrated_interval_model,
    fit_hedonic_model,
    parse_remaining_lease,
    prediction_interval_metrics,
    prepare_model_data,
    run_analysis,
    segment_median_baseline,
)


def synthetic_transactions(month_count: int = 30) -> pd.DataFrame:
    """Create transactions with a known trend and a changing sales mix."""
    records: list[dict[str, object]] = []
    months = pd.date_range("2021-01-01", periods=month_count, freq="MS")
    for time_index, month in enumerate(months):
        profiles: list[tuple[int, int, int]] = [
            (town, flat_type, flat_model)
            for town in range(2)
            for flat_type in range(2)
            for flat_model in range(2)
        ]
        # The core sample keeps every profile observable in every month. Extra
        # lower-priced profiles early and higher-priced profiles later create
        # a deliberate composition shift in the unadjusted monthly median.
        mix_profile = (0, 0, 0) if time_index < month_count // 2 else (1, 1, 1)
        profiles.extend([mix_profile] * 6)

        for row_number, (town, flat_type, flat_model) in enumerate(profiles):
            floor_area = 72.0 + flat_type * 34.0 + flat_model * 3.0
            remaining_lease_months = 720 + flat_model * 180 + town * 24
            storey_mid = 5.0 + town * 6.0 + (row_number % 2) * 3.0
            log_price = (
                np.log(315_000.0)
                + 0.008 * time_index
                + 0.62 * np.log(floor_area / 80.0)
                + 0.0035 * (remaining_lease_months / 12.0 - 60.0)
                + 0.010 * (storey_mid - 5.0)
                + 0.12 * town
                + 0.08 * flat_type
                + 0.035 * flat_model
            )
            records.append(
                {
                    "month": month.strftime("%Y-%m-%d"),
                    "town": f"TOWN {town}",
                    "flat_type": "3 ROOM" if flat_type == 0 else "5 ROOM",
                    "flat_model": "MODEL A" if flat_model == 0 else "MODEL B",
                    "floor_area_sqm": floor_area,
                    "remaining_lease_months": remaining_lease_months,
                    "remaining_lease": "not used when month count is present",
                    "storey_mid": storey_mid,
                    "resale_price": float(np.exp(log_price)),
                }
            )
    return pd.DataFrame.from_records(records)


class ModelDataTests(unittest.TestCase):
    def test_parses_remaining_lease_strings(self) -> None:
        parsed = parse_remaining_lease(
            pd.Series(["61 years 04 months", "72 years", "1 year 01 month"])
        )

        np.testing.assert_allclose(parsed, [61 + 4 / 12, 72, 1 + 1 / 12])
        with self.assertRaisesRegex(ValueError, "invalid values"):
            parse_remaining_lease(pd.Series(["unknown"]))

    def test_uses_validated_remaining_lease_months_when_available(self) -> None:
        prepared = prepare_model_data(synthetic_transactions(month_count=3))

        self.assertAlmostEqual(prepared.loc[0, "remaining_lease_years"], 60.0)
        self.assertEqual(prepared.loc[0, "transaction_month"], "2021-01")

    def test_chronological_holdout_keeps_months_wholly_separate(self) -> None:
        prepared = prepare_model_data(synthetic_transactions(month_count=10))
        train, holdout, cutoff = chronological_holdout(prepared, holdout_months=3)

        self.assertEqual(cutoff, pd.Timestamp("2021-08-01"))
        self.assertEqual(train["month"].nunique(), 7)
        self.assertEqual(holdout["month"].nunique(), 3)
        self.assertLess(train["month"].max(), holdout["month"].min())

    def test_excludes_only_the_incomplete_as_of_month(self) -> None:
        source = synthetic_transactions(month_count=3)

        filtered, completeness = exclude_provisional_latest_month(
            source,
            "2021-02-28T16:30:00+00:00",
        )

        self.assertEqual(filtered["month"].nunique(), 2)
        self.assertEqual(completeness["excluded_month"], "2021-03")
        self.assertEqual(completeness["excluded_rows"], 14)
        self.assertTrue(completeness["provisional_latest_month_excluded"])

        unchanged, complete = exclude_provisional_latest_month(
            source,
            "2021-04-01",
        )
        self.assertEqual(len(unchanged), len(source))
        self.assertFalse(complete["provisional_latest_month_excluded"])

    def test_preprocessor_is_fit_on_training_categories_only(self) -> None:
        source = synthetic_transactions(month_count=10)
        source.loc[source["month"].ge("2021-09"), "town"] = "FUTURE TOWN"
        prepared = prepare_model_data(source)
        train, holdout, _ = chronological_holdout(prepared, holdout_months=2)
        origin = train["month"].min()
        train = add_trend_features(train, origin)
        holdout = add_trend_features(holdout, origin)

        model = fit_hedonic_model(train, time_mode="trend", ridge_alpha=1e-6)
        encoder = model.estimator.named_steps["preprocess"].named_transformers_[
            "categorical"
        ]

        self.assertNotIn("FUTURE TOWN", encoder.categories_[0])
        with self.assertWarnsRegex(UserWarning, "unknown categories"):
            predictions = model.predict_price(holdout)
        self.assertTrue(np.isfinite(predictions).all())

    def test_expanding_window_splits_are_ordered_and_non_overlapping(self) -> None:
        prepared = prepare_model_data(synthetic_transactions(month_count=42))
        splits = expanding_window_splits(
            prepared,
            test_months=6,
            max_splits=3,
            min_train_months=18,
        )

        self.assertEqual(len(splits), 3)
        self.assertEqual([fold for fold, *_ in splits], [1, 2, 3])
        self.assertEqual(
            [train["month"].nunique() for _, train, _, _ in splits], [24, 30, 36]
        )
        for _, train, test, cutoff in splits:
            self.assertLess(train["month"].max(), test["month"].min())
            self.assertEqual(test["month"].nunique(), 6)
            self.assertEqual(cutoff, test["month"].min())

    def test_segment_median_baseline_uses_safe_fallback_hierarchy(self) -> None:
        train = pd.DataFrame(
            {
                "month": ["2022-01-01"] * 4,
                "town": ["A", "A", "A", "B"],
                "flat_type": ["3 ROOM", "3 ROOM", "4 ROOM", "3 ROOM"],
                "resale_price": [100.0, 200.0, 300.0, 400.0],
            }
        )
        test = pd.DataFrame(
            {
                "month": ["2023-01-01"] * 4,
                "town": ["A", "A", "C", "C"],
                "flat_type": ["3 ROOM", "5 ROOM", "3 ROOM", "5 ROOM"],
                "resale_price": [1.0] * 4,
            }
        )

        predictions, metadata = segment_median_baseline(train, test)

        np.testing.assert_allclose(predictions, [150.0, 200.0, 200.0, 250.0])
        self.assertEqual(
            metadata["fallback_counts"],
            {
                "flat_type": 1,
                "global": 1,
                "town": 1,
                "town_x_flat_type": 1,
            },
        )


class HedonicModelTests(unittest.TestCase):
    def test_model_beats_baseline_and_reports_grouped_importance(self) -> None:
        evaluation = evaluate_chronological_holdout(
            synthetic_transactions(),
            holdout_months=6,
            ridge_alpha=1e-6,
            permutation_repeats=2,
            importance_sample_size=500,
        )

        baseline = evaluation.metrics["training_median_baseline"]
        model = evaluation.metrics["hedonic_ridge"]
        self.assertLess(model["mae"], baseline["mae"] * 0.15)
        self.assertGreater(model["r2"], 0.98)
        self.assertEqual(evaluation.split["training_end_month"], "2022-12")
        self.assertEqual(evaluation.split["holdout_start_month"], "2023-01")
        self.assertEqual(
            set(evaluation.permutation_importance["feature_group"]),
            {
                "transaction_month",
                "floor_area",
                "remaining_lease",
                "storey",
                "town",
                "flat_type",
                "flat_model",
            },
        )

    def test_calibrated_quantiles_are_ordered_and_report_coverage(self) -> None:
        prepared = prepare_model_data(synthetic_transactions(month_count=30))
        train, holdout, _ = chronological_holdout(prepared, holdout_months=6)
        origin = train["month"].min()
        train = add_trend_features(train, origin)
        holdout = add_trend_features(holdout, origin)
        point_model = fit_hedonic_model(
            train,
            time_mode="trend",
            ridge_alpha=1e-6,
        )

        interval_model = fit_calibrated_interval_model(
            train,
            point_model=point_model,
            ridge_alpha=1e-6,
            calibration_months=4,
        )
        predictions = interval_model.predict_quantiles(holdout)
        metrics = prediction_interval_metrics(
            holdout["resale_price"],
            predictions,
            lower_quantile=interval_model.lower_quantile,
            upper_quantile=interval_model.upper_quantile,
        )

        self.assertTrue(
            (predictions["lower_price"] <= predictions["median_price"]).all()
        )
        self.assertTrue(
            (predictions["median_price"] <= predictions["upper_price"]).all()
        )
        np.testing.assert_allclose(
            predictions["median_price"],
            point_model.predict_price(holdout),
            rtol=1e-12,
        )
        self.assertEqual(metrics["target_coverage"], 0.8)
        self.assertGreaterEqual(metrics["empirical_coverage"], 0.0)
        self.assertLessEqual(metrics["empirical_coverage"], 1.0)
        self.assertEqual(interval_model.calibration["calibration_months"], 4)

    def test_rolling_backtests_report_every_model_and_interval_fold(self) -> None:
        backtests = evaluate_rolling_backtests(
            synthetic_transactions(month_count=42),
            test_months=6,
            max_splits=3,
            min_train_months=18,
            ridge_alpha=1e-6,
            interval_calibration_months=4,
        )

        self.assertEqual(backtests.fold_metrics["fold"].nunique(), 3)
        self.assertEqual(len(backtests.interval_metrics), 3)
        self.assertEqual(
            set(backtests.fold_metrics["model"]),
            {
                "training_median_baseline",
                "town_flat_type_median_baseline",
                "prior_12m_town_flat_type_median_baseline",
                "hedonic_ridge",
            },
        )
        self.assertTrue(
            backtests.interval_metrics["empirical_coverage"].between(0, 1).all()
        )
        self.assertTrue(
            (
                backtests.interval_metrics["training_end_month"]
                < backtests.interval_metrics["test_start_month"]
            ).all()
        )
        self.assertTrue(
            backtests.interval_metrics["median_model"].eq("hedonic_ridge").all()
        )

    def test_adjusted_index_is_deterministic_and_recovers_known_trend(self) -> None:
        source = synthetic_transactions()
        first, _ = derive_quality_adjusted_index(source, ridge_alpha=1e-6)
        second, _ = derive_quality_adjusted_index(source, ridge_alpha=1e-6)

        assert_frame_equal(first, second)
        expected_last_index = np.exp(0.008 * 29) * 100.0
        self.assertAlmostEqual(
            first["quality_adjusted_price_index"].iloc[-1],
            expected_last_index,
            delta=1.0,
        )
        self.assertGreater(
            first["raw_price_index"].iloc[-1]
            - first["quality_adjusted_price_index"].iloc[-1],
            20.0,
        )
        self.assertEqual(first["raw_price_index"].iloc[0], 100.0)
        self.assertEqual(first["quality_adjusted_price_index"].iloc[0], 100.0)

    def test_run_analysis_writes_all_small_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "clean.csv"
            reports_dir = root / "reports"
            images_dir = root / "images"
            synthetic_transactions(month_count=18).to_csv(input_path, index=False)

            summary = run_analysis(
                input_path=input_path,
                reports_dir=reports_dir,
                images_dir=images_dir,
                holdout_months=4,
                ridge_alpha=1e-6,
                permutation_repeats=1,
                importance_sample_size=100,
                as_of="2022-06-15",
            )

            expected_paths = {
                reports_dir / "price_model_metrics.json",
                reports_dir / "price_model_permutation_importance.csv",
                reports_dir / "quality_adjusted_price_index.csv",
                reports_dir / "price_model_backtests.csv",
                reports_dir / "price_model_error_slices.csv",
                reports_dir / "price_model_interval_metrics.csv",
                images_dir / "raw_vs_quality_adjusted_price_index.png",
            }
            self.assertTrue(
                all(path.is_file() and path.stat().st_size for path in expected_paths)
            )
            stored_summary = json.loads(
                (reports_dir / "price_model_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                stored_summary["split"]["holdout_start_month"],
                summary["split"]["holdout_start_month"],
            )
            self.assertIn("not causal", stored_summary["methodology"]["interpretation"])
            self.assertEqual(stored_summary["methodology"]["index_ridge_alpha"], 0.0)
            self.assertIn(
                "unpenalized",
                stored_summary["methodology"]["index_estimator"],
            )
            self.assertEqual(
                stored_summary["input_provenance"]["excluded_month"],
                "2022-06",
            )
            self.assertEqual(stored_summary["index_summary"]["latest_month"], "2022-05")
            self.assertEqual(
                stored_summary["input_provenance"]["sha256"],
                summary["input_provenance"]["sha256"],
            )
            self.assertEqual(stored_summary["rolling_backtests"]["folds"], 1)
            self.assertEqual(
                stored_summary["holdout_prediction_interval"]["target_coverage"],
                0.8,
            )
            self.assertIn(
                "prior_12m_town_flat_type_median_baseline",
                stored_summary["holdout_metrics"],
            )
            error_slices = pd.read_csv(reports_dir / "price_model_error_slices.csv")
            self.assertEqual(
                set(error_slices["slice_dimension"]),
                {"town", "flat_type", "price_band", "remaining_lease_band"},
            )


if __name__ == "__main__":
    unittest.main()
