"""Evaluate a hedonic price model and build a quality-adjusted time index.

The module deliberately separates two modelling tasks:

* an out-of-time evaluation model, trained only on observations before the
  holdout period and using a smooth time trend that can extrapolate; and
* a full-sample monthly fixed-effects model used to describe a mix-adjusted
  price index.

Both are descriptive models of transaction prices.  Their coefficients and
indices should not be interpreted as causal effects or formal valuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"

RANDOM_SEED = 42
SINGAPORE_TIME_ZONE = ZoneInfo("Asia/Singapore")
DEFAULT_HOLDOUT_MONTHS = 12
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_INDEX_RIDGE_ALPHA = 0.0
DEFAULT_PERMUTATION_REPEATS = 3
DEFAULT_IMPORTANCE_SAMPLE_SIZE = 30_000

DEFAULT_BACKTEST_SPLITS = 4
DEFAULT_BACKTEST_MIN_TRAIN_MONTHS = 24
DEFAULT_INTERVAL_CALIBRATION_MONTHS = 6
DEFAULT_INTERVAL_QUANTILES = (0.10, 0.50, 0.90)

SEGMENT_COLUMNS = ("town", "flat_type")
PRICE_BAND_EDGES = (-np.inf, 300_000, 400_000, 500_000, 700_000, 1_000_000, np.inf)
PRICE_BAND_LABELS = (
    "Below $300k",
    "$300k-$399k",
    "$400k-$499k",
    "$500k-$699k",
    "$700k-$999k",
    "$1m and above",
)
LEASE_BAND_EDGES = (-np.inf, 50, 60, 70, 80, 90, np.inf)
LEASE_BAND_LABELS = (
    "Below 50 years",
    "50-59 years",
    "60-69 years",
    "70-79 years",
    "80-89 years",
    "90 years and above",
)

BASE_REQUIRED_COLUMNS = {
    "month",
    "town",
    "flat_type",
    "flat_model",
    "floor_area_sqm",
    "storey_mid",
    "resale_price",
}
LEASE_COLUMNS = {"remaining_lease", "remaining_lease_months"}

QUALITY_NUMERIC_FEATURES = (
    "log_floor_area_sqm",
    "log_floor_area_sqm_squared",
    "remaining_lease_years",
    "remaining_lease_years_squared",
    "storey_mid",
    "storey_mid_squared",
)
QUALITY_CATEGORICAL_FEATURES = ("town", "flat_type", "flat_model")
TREND_FEATURES = (
    "time_index",
    "season_sin",
    "season_cos",
)

PERMUTATION_GROUPS = (
    ("transaction_month", TREND_FEATURES),
    (
        "floor_area",
        ("log_floor_area_sqm", "log_floor_area_sqm_squared"),
    ),
    (
        "remaining_lease",
        ("remaining_lease_years", "remaining_lease_years_squared"),
    ),
    ("storey", ("storey_mid", "storey_mid_squared")),
    ("town", ("town",)),
    ("flat_type", ("flat_type",)),
    ("flat_model", ("flat_model",)),
)

TimeMode = Literal["trend", "monthly_fixed_effects"]


@dataclass
class HedonicPriceModel:
    """A fitted log-price pipeline plus its retransformation adjustment."""

    estimator: Pipeline
    smearing_factor: float
    feature_columns: tuple[str, ...]
    time_mode: TimeMode

    def predict_log_price(self, df: pd.DataFrame) -> np.ndarray:
        """Predict natural-log resale prices."""
        return np.asarray(
            self.estimator.predict(df.loc[:, self.feature_columns]),
            dtype="float64",
        )

    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """Predict resale prices on the dollar scale."""
        log_prediction = self.predict_log_price(df)
        return np.exp(np.clip(log_prediction, -50.0, 50.0)) * self.smearing_factor


@dataclass
class ModelEvaluation:
    """Results from the chronological holdout evaluation."""

    model: HedonicPriceModel
    metrics: dict[str, dict[str, float]]
    permutation_importance: pd.DataFrame
    split: dict[str, object]
    error_slices: pd.DataFrame
    interval_metrics: dict[str, object]


@dataclass
class CalibratedIntervalModel:
    """A hedonic point model with time-ordered residual quantile offsets."""

    model: HedonicPriceModel
    lower_quantile: float
    median_quantile: float
    upper_quantile: float
    lower_log_offset: float
    median_log_offset: float
    upper_log_offset: float
    calibration: dict[str, object]

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return lower, median and upper prices in ascending order."""
        predicted_log_price = self.model.predict_log_price(df)
        return pd.DataFrame(
            {
                "lower_price": np.exp(
                    np.clip(predicted_log_price + self.lower_log_offset, -50.0, 50.0)
                ),
                "median_price": np.exp(
                    np.clip(predicted_log_price + self.median_log_offset, -50.0, 50.0)
                ),
                "upper_price": np.exp(
                    np.clip(predicted_log_price + self.upper_log_offset, -50.0, 50.0)
                ),
            },
            index=df.index,
        )


@dataclass
class BacktestEvaluation:
    """Results from non-overlapping expanding-window backtest folds."""

    fold_metrics: pd.DataFrame
    interval_metrics: pd.DataFrame


def parse_remaining_lease(values: pd.Series) -> pd.Series:
    """Convert HDB lease strings such as ``61 years 04 months`` to years."""
    if pd.api.types.is_numeric_dtype(values):
        numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
        invalid = ~np.isfinite(numeric_values) | numeric_values.le(0)
        if invalid.any():
            raise ValueError(
                f"remaining_lease contains {int(invalid.sum())} invalid values"
            )
        return numeric_values

    lease_text = values.astype("string").str.strip()
    parts = lease_text.str.extract(
        r"^(?P<years>\d+(?:\.\d+)?)\s+years?"
        r"(?:\s+(?P<months>\d+)\s+months?)?$"
    )
    years = pd.to_numeric(parts["years"], errors="coerce")
    months = pd.to_numeric(parts["months"], errors="coerce").fillna(0)
    invalid = years.isna() | months.lt(0) | months.gt(11)
    if invalid.any():
        examples = lease_text.loc[invalid].dropna().unique()[:3]
        example_text = ", ".join(repr(value) for value in examples)
        suffix = f"; examples: {example_text}" if example_text else ""
        raise ValueError(
            f"remaining_lease contains {int(invalid.sum())} invalid values{suffix}"
        )
    return (years + months / 12.0).astype("float64")


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input fields and create deterministic model features."""
    missing = sorted(BASE_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Model data is missing required columns: " + ", ".join(missing)
        )
    if LEASE_COLUMNS.isdisjoint(df.columns):
        raise ValueError(
            "Model data requires remaining_lease_months or remaining_lease"
        )

    model_df = df.copy()
    model_df["month"] = pd.to_datetime(model_df["month"], errors="coerce")
    if model_df["month"].isna().any():
        raise ValueError(
            f"month contains {int(model_df['month'].isna().sum())} invalid values"
        )
    model_df["month"] = model_df["month"].dt.to_period("M").dt.to_timestamp()

    for column in ("floor_area_sqm", "storey_mid", "resale_price"):
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
        invalid = ~np.isfinite(model_df[column])
        if invalid.any():
            raise ValueError(f"{column} contains {int(invalid.sum())} invalid values")

    if model_df["floor_area_sqm"].le(0).any():
        raise ValueError("floor_area_sqm must be greater than zero")
    if model_df["resale_price"].le(0).any():
        raise ValueError("resale_price must be greater than zero")

    for column in QUALITY_CATEGORICAL_FEATURES:
        model_df[column] = model_df[column].astype("string").str.strip()
        invalid = model_df[column].isna() | model_df[column].eq("")
        if invalid.any():
            raise ValueError(f"{column} contains {int(invalid.sum())} blank values")

    if "remaining_lease_months" in model_df.columns:
        lease_months = pd.to_numeric(
            model_df["remaining_lease_months"], errors="coerce"
        )
        invalid = ~np.isfinite(lease_months) | lease_months.le(0)
        if invalid.any():
            raise ValueError(
                f"remaining_lease_months contains {int(invalid.sum())} invalid values"
            )
        remaining_lease_years = lease_months.astype("float64") / 12.0
    else:
        remaining_lease_years = parse_remaining_lease(model_df["remaining_lease"])

    log_area = np.log(model_df["floor_area_sqm"].astype("float64"))
    model_df["log_floor_area_sqm"] = log_area
    model_df["log_floor_area_sqm_squared"] = log_area.pow(2)
    model_df["remaining_lease_years"] = remaining_lease_years
    model_df["remaining_lease_years_squared"] = remaining_lease_years.pow(2)
    model_df["storey_mid_squared"] = model_df["storey_mid"].pow(2)
    model_df["transaction_month"] = model_df["month"].dt.strftime("%Y-%m")

    return model_df


def add_trend_features(df: pd.DataFrame, origin_month: pd.Timestamp) -> pd.DataFrame:
    """Add extrapolatable trend and calendar-seasonality features."""
    trend_df = df.copy()
    origin = pd.Timestamp(origin_month).to_period("M")
    month_period = trend_df["month"].dt.to_period("M")
    month_number = (
        (month_period.dt.year - origin.year) * 12
        + (month_period.dt.month - origin.month)
    ).astype("float64")
    calendar_month = month_period.dt.month.astype("float64")

    trend_df["time_index"] = month_number
    trend_df["season_sin"] = np.sin(2.0 * np.pi * calendar_month / 12.0)
    trend_df["season_cos"] = np.cos(2.0 * np.pi * calendar_month / 12.0)
    return trend_df


def chronological_holdout(
    df: pd.DataFrame, holdout_months: int = DEFAULT_HOLDOUT_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Reserve the latest complete set of distinct months as the holdout."""
    if holdout_months < 1:
        raise ValueError("holdout_months must be at least 1")
    if "month" not in df.columns:
        raise ValueError("Model data is missing required column: month")

    months = pd.DatetimeIndex(pd.to_datetime(df["month"]).unique()).sort_values()
    if len(months) <= holdout_months:
        raise ValueError(
            "Not enough distinct months for the requested chronological holdout: "
            f"found {len(months)}, requested {holdout_months}"
        )

    cutoff = pd.Timestamp(months[-holdout_months]).to_period("M").to_timestamp()
    transaction_month = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    train = df.loc[transaction_month.lt(cutoff)].copy()
    holdout = df.loc[transaction_month.ge(cutoff)].copy()
    if train.empty or holdout.empty:
        raise ValueError("Chronological holdout produced an empty partition")
    if pd.Timestamp(train["month"].max()) >= pd.Timestamp(holdout["month"].min()):
        raise RuntimeError("Chronological holdout contains overlapping months")
    return train, holdout, cutoff


def expanding_window_splits(
    df: pd.DataFrame,
    *,
    test_months: int = DEFAULT_HOLDOUT_MONTHS,
    max_splits: int = DEFAULT_BACKTEST_SPLITS,
    min_train_months: int = DEFAULT_BACKTEST_MIN_TRAIN_MONTHS,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, pd.Timestamp]]:
    """Build non-overlapping, expanding-window splits ending at the latest month."""
    if test_months < 1:
        raise ValueError("test_months must be at least 1")
    if max_splits < 1:
        raise ValueError("max_splits must be at least 1")
    if min_train_months < 1:
        raise ValueError("min_train_months must be at least 1")
    if "month" not in df.columns:
        raise ValueError("Model data is missing required column: month")

    transaction_month = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    months = pd.DatetimeIndex(transaction_month.unique()).sort_values()
    possible_splits = (len(months) - min_train_months) // test_months
    split_count = min(max_splits, possible_splits)
    if split_count < 1:
        raise ValueError(
            "Not enough distinct months for an expanding-window backtest: "
            f"found {len(months)}, need at least "
            f"{min_train_months + test_months}"
        )

    first_test_position = len(months) - split_count * test_months
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.Timestamp]] = []
    for fold_number in range(1, split_count + 1):
        test_start_position = first_test_position + (fold_number - 1) * test_months
        test_month_index = months[
            test_start_position : test_start_position + test_months
        ]
        cutoff = pd.Timestamp(test_month_index[0])
        train = df.loc[transaction_month.lt(cutoff)].copy()
        test = df.loc[transaction_month.isin(test_month_index)].copy()
        if train["month"].nunique() < min_train_months:
            raise RuntimeError(
                "Expanding-window split violated minimum training history"
            )
        if test["month"].nunique() != test_months:
            raise RuntimeError(
                "Expanding-window split produced an incomplete test window"
            )
        splits.append((fold_number, train, test, cutoff))
    return splits


def _make_estimator(
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    ridge_alpha: float,
) -> Pipeline:
    """Construct the reproducible preprocessing and regression pipeline."""
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")

    preprocessor = ColumnTransformer(
        transformers=(
            ("numeric", StandardScaler(), list(numeric_features)),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                list(categorical_features),
            ),
        ),
        remainder="drop",
        sparse_threshold=1.0,
    )
    regressor = (
        LinearRegression()
        if ridge_alpha == 0
        else Ridge(
            alpha=ridge_alpha,
            solver="lsqr",
            tol=1e-8,
            max_iter=10_000,
        )
    )
    return Pipeline(
        steps=(
            ("preprocess", preprocessor),
            ("regressor", regressor),
        )
    )


def fit_hedonic_model(
    df: pd.DataFrame,
    *,
    time_mode: TimeMode,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> HedonicPriceModel:
    """Fit a hedonic log-price regression, optionally with Ridge regularisation."""
    if time_mode == "trend":
        numeric_features = QUALITY_NUMERIC_FEATURES + TREND_FEATURES
        categorical_features = QUALITY_CATEGORICAL_FEATURES
    elif time_mode == "monthly_fixed_effects":
        numeric_features = QUALITY_NUMERIC_FEATURES
        categorical_features = QUALITY_CATEGORICAL_FEATURES + ("transaction_month",)
    else:
        raise ValueError(f"Unsupported time_mode: {time_mode!r}")

    feature_columns = numeric_features + categorical_features
    missing = sorted(set(feature_columns).difference(df.columns))
    if missing:
        raise ValueError("Model features are missing: " + ", ".join(missing))

    target = np.log(df["resale_price"].to_numpy(dtype="float64"))
    estimator = _make_estimator(numeric_features, categorical_features, ridge_alpha)
    estimator.fit(df.loc[:, feature_columns], target)

    fitted_log_price = np.asarray(
        estimator.predict(df.loc[:, feature_columns]), dtype="float64"
    )
    # Duan's smearing factor reduces retransformation bias when reporting
    # predictions on the original dollar scale.
    smearing_factor = float(np.mean(np.exp(target - fitted_log_price)))
    return HedonicPriceModel(
        estimator=estimator,
        smearing_factor=smearing_factor,
        feature_columns=feature_columns,
        time_mode=time_mode,
    )


def regression_metrics(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> dict[str, float]:
    """Return MAE, RMSE and R-squared on the original price scale."""
    actual_values = np.asarray(actual, dtype="float64")
    predicted_values = np.asarray(predicted, dtype="float64")
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal shape")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted must contain only finite values")

    error = actual_values - predicted_values
    absolute_error = np.abs(error)
    squared_error = float(np.dot(error, error))
    centered_actual = actual_values - actual_values.mean()
    total_variation = float(np.dot(centered_actual, centered_actual))
    if total_variation == 0:
        r_squared = 1.0 if squared_error == 0 else 0.0
    else:
        r_squared = 1.0 - squared_error / total_variation
    return {
        "mae": float(np.mean(absolute_error)),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r_squared),
        "mape_pct": float(np.mean(absolute_error / actual_values) * 100.0),
    }


def segment_median_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    lookback_months: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Predict segment medians with progressively broader, safe fallbacks."""
    required = {"month", "town", "flat_type", "resale_price"}
    missing = sorted(
        required.difference(train.columns) | required.difference(test.columns)
    )
    if missing:
        raise ValueError(
            "Baseline data is missing required columns: " + ", ".join(missing)
        )
    if train.empty or test.empty:
        raise ValueError("Baseline train and test partitions must be non-empty")
    if lookback_months is not None and lookback_months < 1:
        raise ValueError("lookback_months must be at least 1")

    reference = train
    if lookback_months is not None:
        test_start = pd.Timestamp(test["month"].min()).to_period("M").to_timestamp()
        reference_start = test_start - pd.DateOffset(months=lookback_months)
        reference_month = (
            pd.to_datetime(train["month"]).dt.to_period("M").dt.to_timestamp()
        )
        reference = train.loc[
            reference_month.ge(reference_start) & reference_month.lt(test_start)
        ]
        if reference.empty:
            reference = train

    prediction = np.full(len(test), np.nan, dtype="float64")
    fallback_level = np.full(len(test), "unassigned", dtype=object)

    def fill_from_group(columns: tuple[str, ...], level_name: str) -> None:
        lookup = (
            reference.groupby(list(columns), as_index=False, observed=True)[
                "resale_price"
            ]
            .median()
            .rename(columns={"resale_price": "_baseline_price"})
        )
        candidates = (
            test.loc[:, list(columns)]
            .reset_index(drop=True)
            .merge(lookup, how="left", on=list(columns), sort=False)["_baseline_price"]
            .to_numpy(dtype="float64")
        )
        available = np.isnan(prediction) & np.isfinite(candidates)
        prediction[available] = candidates[available]
        fallback_level[available] = level_name

    fill_from_group(SEGMENT_COLUMNS, "town_x_flat_type")
    fill_from_group(("town",), "town")
    fill_from_group(("flat_type",), "flat_type")

    global_median = float(reference["resale_price"].median())
    if not np.isfinite(global_median):
        global_median = float(train["resale_price"].median())
    missing_prediction = np.isnan(prediction)
    prediction[missing_prediction] = global_median
    fallback_level[missing_prediction] = "global"

    counts = pd.Series(fallback_level, dtype="string").value_counts(sort=False)
    metadata: dict[str, object] = {
        "lookback_months": lookback_months,
        "reference_start_month": pd.Timestamp(reference["month"].min()).strftime(
            "%Y-%m"
        ),
        "reference_end_month": pd.Timestamp(reference["month"].max()).strftime("%Y-%m"),
        "reference_rows": len(reference),
        "global_fallback_median_price": global_median,
        "fallback_counts": {
            str(level): int(count) for level, count in counts.sort_index().items()
        },
    }
    return prediction, metadata


def fit_calibrated_interval_model(
    train: pd.DataFrame,
    *,
    point_model: HedonicPriceModel | None = None,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    calibration_months: int = DEFAULT_INTERVAL_CALIBRATION_MONTHS,
    quantiles: tuple[float, float, float] = DEFAULT_INTERVAL_QUANTILES,
) -> CalibratedIntervalModel:
    """Calibrate centered log-residual price quantiles on recent training months.

    Residual tail spreads come from a strictly later calibration partition, while
    the median remains anchored to the full-training point model after refitting.
    """
    lower_quantile, median_quantile, upper_quantile = quantiles
    if not 0 < lower_quantile < median_quantile < upper_quantile < 1:
        raise ValueError("quantiles must be strictly increasing and inside (0, 1)")
    if calibration_months < 1:
        raise ValueError("calibration_months must be at least 1")

    months = pd.DatetimeIndex(pd.to_datetime(train["month"]).unique()).sort_values()
    if len(months) < 2:
        raise ValueError("Interval calibration requires at least two training months")
    effective_calibration_months = min(calibration_months, len(months) - 1)
    calibration_cutoff = pd.Timestamp(months[-effective_calibration_months])
    transaction_month = (
        pd.to_datetime(train["month"]).dt.to_period("M").dt.to_timestamp()
    )
    development = train.loc[transaction_month.lt(calibration_cutoff)].copy()
    calibration = train.loc[transaction_month.ge(calibration_cutoff)].copy()
    if development.empty or calibration.empty:
        raise RuntimeError("Interval calibration produced an empty partition")

    calibration_model = fit_hedonic_model(
        development,
        time_mode="trend",
        ridge_alpha=ridge_alpha,
    )
    calibration_residual = np.log(
        calibration["resale_price"].to_numpy(dtype="float64")
    ) - calibration_model.predict_log_price(calibration)
    offsets = np.quantile(calibration_residual, quantiles)
    if point_model is None:
        point_model = fit_hedonic_model(
            train,
            time_mode="trend",
            ridge_alpha=ridge_alpha,
        )

    center_log_offset = float(np.log(point_model.smearing_factor))
    centered_offsets = offsets - offsets[1] + center_log_offset

    return CalibratedIntervalModel(
        model=point_model,
        lower_quantile=lower_quantile,
        median_quantile=median_quantile,
        upper_quantile=upper_quantile,
        lower_log_offset=float(centered_offsets[0]),
        median_log_offset=float(centered_offsets[1]),
        upper_log_offset=float(centered_offsets[2]),
        calibration={
            "calibration_start_month": pd.Timestamp(
                calibration["month"].min()
            ).strftime("%Y-%m"),
            "calibration_end_month": pd.Timestamp(calibration["month"].max()).strftime(
                "%Y-%m"
            ),
            "calibration_rows": len(calibration),
            "calibration_months": effective_calibration_months,
            "development_rows": len(development),
            "calibration_median_log_residual": float(offsets[1]),
            "interval_center_log_offset": center_log_offset,
        },
    )


def prediction_interval_metrics(
    actual: np.ndarray | pd.Series,
    predictions: pd.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, float]:
    """Measure empirical coverage, width and median-prediction accuracy."""
    required = {"lower_price", "median_price", "upper_price"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            "Interval predictions are missing required columns: " + ", ".join(missing)
        )

    actual_values = np.asarray(actual, dtype="float64")
    lower = predictions["lower_price"].to_numpy(dtype="float64")
    median = predictions["median_price"].to_numpy(dtype="float64")
    upper = predictions["upper_price"].to_numpy(dtype="float64")
    if not (
        len(actual_values) == len(lower) == len(median) == len(upper)
        and len(actual_values) > 0
    ):
        raise ValueError("Interval arrays must be non-empty and have equal lengths")
    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(lower).all()
        and np.isfinite(median).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError("Interval arrays must contain only finite values")
    if np.any(lower > median) or np.any(median > upper):
        raise ValueError("Interval predictions must satisfy lower <= median <= upper")

    covered = (actual_values >= lower) & (actual_values <= upper)
    width = upper - lower
    return {
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "target_coverage": float(upper_quantile - lower_quantile),
        "empirical_coverage": float(np.mean(covered)),
        "lower_violation_rate": float(np.mean(actual_values < lower)),
        "upper_violation_rate": float(np.mean(actual_values > upper)),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "median_prediction_mae": float(np.mean(np.abs(actual_values - median))),
    }


def build_error_slices(
    holdout: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Summarise out-of-time errors by key market and property segments."""
    if holdout.empty:
        raise ValueError("Cannot calculate error slices for an empty holdout")
    actual = holdout["resale_price"].to_numpy(dtype="float64")
    slice_data = pd.DataFrame(
        {
            "town": holdout["town"].astype("string").to_numpy(),
            "flat_type": holdout["flat_type"].astype("string").to_numpy(),
            "price_band": pd.cut(
                actual,
                bins=PRICE_BAND_EDGES,
                labels=PRICE_BAND_LABELS,
                right=False,
            ),
            "remaining_lease_band": pd.cut(
                holdout["remaining_lease_years"].to_numpy(dtype="float64"),
                bins=LEASE_BAND_EDGES,
                labels=LEASE_BAND_LABELS,
                right=False,
            ),
        }
    )
    records: list[dict[str, object]] = []
    for model_name, predicted in predictions.items():
        predicted_values = np.asarray(predicted, dtype="float64")
        if len(predicted_values) != len(actual):
            raise ValueError(
                f"Prediction length for {model_name!r} does not match holdout"
            )
        for dimension in (
            "town",
            "flat_type",
            "price_band",
            "remaining_lease_band",
        ):
            grouped_indices = slice_data.groupby(
                dimension,
                observed=True,
                sort=True,
                dropna=False,
            ).indices
            for value, positions in grouped_indices.items():
                position_array = np.asarray(positions, dtype="int64")
                actual_slice = actual[position_array]
                predicted_slice = predicted_values[position_array]
                metrics = regression_metrics(actual_slice, predicted_slice)
                records.append(
                    {
                        "model": model_name,
                        "slice_dimension": dimension,
                        "slice_value": str(value),
                        "observations": len(position_array),
                        **metrics,
                        "mean_error": float(np.mean(predicted_slice - actual_slice)),
                    }
                )

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["model", "slice_dimension", "slice_value"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def grouped_permutation_importance(
    model: HedonicPriceModel,
    holdout: pd.DataFrame,
    *,
    repeats: int = DEFAULT_PERMUTATION_REPEATS,
    sample_size: int = DEFAULT_IMPORTANCE_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Measure holdout degradation after jointly shuffling feature groups."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")

    if len(holdout) > sample_size:
        sample = holdout.sample(n=sample_size, random_state=random_seed)
    else:
        sample = holdout.copy()
    sample = sample.reset_index(drop=True)

    actual = sample["resale_price"].to_numpy(dtype="float64")
    baseline_metrics = regression_metrics(actual, model.predict_price(sample))
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []

    for group_name, group_columns in PERMUTATION_GROUPS:
        active_columns = tuple(
            column for column in group_columns if column in model.feature_columns
        )
        if not active_columns:
            continue
        mae_increases: list[float] = []
        r2_decreases: list[float] = []
        for _ in range(repeats):
            order = rng.permutation(len(sample))
            shuffled = sample.copy()
            for column in active_columns:
                shuffled[column] = sample[column].to_numpy()[order]
            shuffled_metrics = regression_metrics(actual, model.predict_price(shuffled))
            mae_increases.append(shuffled_metrics["mae"] - baseline_metrics["mae"])
            r2_decreases.append(baseline_metrics["r2"] - shuffled_metrics["r2"])

        records.append(
            {
                "feature_group": group_name,
                "mae_increase": float(np.mean(mae_increases)),
                "mae_increase_std": float(np.std(mae_increases, ddof=0)),
                "r2_decrease": float(np.mean(r2_decreases)),
                "r2_decrease_std": float(np.std(r2_decreases, ddof=0)),
                "repeats": repeats,
                "sample_rows": len(sample),
            }
        )

    importance = pd.DataFrame.from_records(records)
    return importance.sort_values(
        ["mae_increase", "feature_group"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def evaluate_chronological_holdout(
    df: pd.DataFrame,
    *,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    importance_sample_size: int = DEFAULT_IMPORTANCE_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
    interval_calibration_months: int = DEFAULT_INTERVAL_CALIBRATION_MONTHS,
) -> ModelEvaluation:
    """Compare the hedonic model with global and segment-median baselines."""
    prepared = prepare_model_data(df)
    train, holdout, cutoff = chronological_holdout(prepared, holdout_months)
    origin = pd.Timestamp(train["month"].min())
    train = add_trend_features(train, origin)
    holdout = add_trend_features(holdout, origin)

    model = fit_hedonic_model(train, time_mode="trend", ridge_alpha=ridge_alpha)
    actual = holdout["resale_price"].to_numpy(dtype="float64")
    baseline_price = float(train["resale_price"].median())
    baseline_prediction = np.full(len(holdout), baseline_price, dtype="float64")
    segment_prediction, segment_metadata = segment_median_baseline(train, holdout)
    recent_segment_prediction, recent_segment_metadata = segment_median_baseline(
        train,
        holdout,
        lookback_months=12,
    )
    model_prediction = model.predict_price(holdout)

    interval_model = fit_calibrated_interval_model(
        train,
        point_model=model,
        ridge_alpha=ridge_alpha,
        calibration_months=interval_calibration_months,
    )
    quantile_predictions = interval_model.predict_quantiles(holdout)
    interval_metrics: dict[str, object] = {
        **prediction_interval_metrics(
            actual,
            quantile_predictions,
            lower_quantile=interval_model.lower_quantile,
            upper_quantile=interval_model.upper_quantile,
        ),
        "median_quantile": interval_model.median_quantile,
        "median_model": "hedonic_ridge",
        **interval_model.calibration,
        "test_rows": len(holdout),
    }

    prediction_by_model = {
        "training_median_baseline": baseline_prediction,
        "town_flat_type_median_baseline": segment_prediction,
        "prior_12m_town_flat_type_median_baseline": recent_segment_prediction,
        "hedonic_ridge": model_prediction,
    }
    metrics = {
        name: regression_metrics(actual, prediction)
        for name, prediction in prediction_by_model.items()
    }
    error_slices = build_error_slices(holdout, prediction_by_model)

    importance = grouped_permutation_importance(
        model,
        holdout,
        repeats=permutation_repeats,
        sample_size=importance_sample_size,
        random_seed=random_seed,
    )
    split: dict[str, object] = {
        "holdout_months": holdout_months,
        "cutoff_month": cutoff.strftime("%Y-%m"),
        "training_start_month": pd.Timestamp(train["month"].min()).strftime("%Y-%m"),
        "training_end_month": pd.Timestamp(train["month"].max()).strftime("%Y-%m"),
        "holdout_start_month": pd.Timestamp(holdout["month"].min()).strftime("%Y-%m"),
        "holdout_end_month": pd.Timestamp(holdout["month"].max()).strftime("%Y-%m"),
        "training_rows": len(train),
        "holdout_rows": len(holdout),
        "baseline_training_median_price": baseline_price,
        "town_flat_type_baseline": segment_metadata,
        "prior_12m_town_flat_type_baseline": recent_segment_metadata,
    }
    return ModelEvaluation(
        model=model,
        metrics=metrics,
        permutation_importance=importance,
        split=split,
        error_slices=error_slices,
        interval_metrics=interval_metrics,
    )


def evaluate_rolling_backtests(
    df: pd.DataFrame,
    *,
    test_months: int = DEFAULT_HOLDOUT_MONTHS,
    max_splits: int = DEFAULT_BACKTEST_SPLITS,
    min_train_months: int = DEFAULT_BACKTEST_MIN_TRAIN_MONTHS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    interval_calibration_months: int = DEFAULT_INTERVAL_CALIBRATION_MONTHS,
) -> BacktestEvaluation:
    """Evaluate models on several historical, expanding-window test periods."""
    prepared = prepare_model_data(df)
    splits = expanding_window_splits(
        prepared,
        test_months=test_months,
        max_splits=max_splits,
        min_train_months=min_train_months,
    )
    fold_records: list[dict[str, object]] = []
    interval_records: list[dict[str, object]] = []

    for fold_number, train, test, _ in splits:
        origin = pd.Timestamp(train["month"].min())
        train = add_trend_features(train, origin)
        test = add_trend_features(test, origin)
        actual = test["resale_price"].to_numpy(dtype="float64")

        model = fit_hedonic_model(
            train,
            time_mode="trend",
            ridge_alpha=ridge_alpha,
        )
        global_median = float(train["resale_price"].median())
        global_prediction = np.full(len(test), global_median, dtype="float64")
        segment_prediction, segment_metadata = segment_median_baseline(train, test)
        recent_prediction, recent_metadata = segment_median_baseline(
            train,
            test,
            lookback_months=12,
        )
        model_prediction = model.predict_price(test)

        interval_model = fit_calibrated_interval_model(
            train,
            point_model=model,
            ridge_alpha=ridge_alpha,
            calibration_months=interval_calibration_months,
        )
        quantile_predictions = interval_model.predict_quantiles(test)
        prediction_by_model = {
            "training_median_baseline": global_prediction,
            "town_flat_type_median_baseline": segment_prediction,
            "prior_12m_town_flat_type_median_baseline": recent_prediction,
            "hedonic_ridge": model_prediction,
        }

        fold_context: dict[str, object] = {
            "fold": fold_number,
            "training_start_month": pd.Timestamp(train["month"].min()).strftime(
                "%Y-%m"
            ),
            "training_end_month": pd.Timestamp(train["month"].max()).strftime("%Y-%m"),
            "test_start_month": pd.Timestamp(test["month"].min()).strftime("%Y-%m"),
            "test_end_month": pd.Timestamp(test["month"].max()).strftime("%Y-%m"),
            "training_rows": len(train),
            "test_rows": len(test),
        }
        fallback_by_model = {
            "town_flat_type_median_baseline": segment_metadata["fallback_counts"],
            "prior_12m_town_flat_type_median_baseline": recent_metadata[
                "fallback_counts"
            ],
        }
        for model_name, prediction in prediction_by_model.items():
            fallback_counts = fallback_by_model.get(model_name, {})
            fold_records.append(
                {
                    **fold_context,
                    "model": model_name,
                    **regression_metrics(actual, prediction),
                    "exact_segment_rows": int(
                        fallback_counts.get("town_x_flat_type", 0)
                    ),
                    "town_fallback_rows": int(fallback_counts.get("town", 0)),
                    "flat_type_fallback_rows": int(fallback_counts.get("flat_type", 0)),
                    "global_fallback_rows": int(fallback_counts.get("global", 0)),
                }
            )

        interval_records.append(
            {
                **fold_context,
                "median_quantile": interval_model.median_quantile,
                "median_model": "hedonic_ridge",
                **interval_model.calibration,
                **prediction_interval_metrics(
                    actual,
                    quantile_predictions,
                    lower_quantile=interval_model.lower_quantile,
                    upper_quantile=interval_model.upper_quantile,
                ),
            }
        )

    return BacktestEvaluation(
        fold_metrics=pd.DataFrame.from_records(fold_records),
        interval_metrics=pd.DataFrame.from_records(interval_records),
    )


def derive_quality_adjusted_index(
    df: pd.DataFrame,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> tuple[pd.DataFrame, HedonicPriceModel]:
    """Return raw and model-based monthly indices with the first month at 100."""
    prepared = prepare_model_data(df)
    model = fit_hedonic_model(
        prepared,
        time_mode="monthly_fixed_effects",
        ridge_alpha=ridge_alpha,
    )

    monthly = (
        prepared.groupby("month", as_index=False, sort=True)
        .agg(
            transactions=("resale_price", "size"),
            raw_median_price=("resale_price", "median"),
        )
        .sort_values("month", kind="mergesort")
        .reset_index(drop=True)
    )
    if monthly.empty:
        raise ValueError("Cannot derive an index from an empty dataset")

    # Holding every property attribute fixed while changing only the month
    # isolates the model's monthly fixed effect. With this additive log model,
    # the chosen reference observation does not affect the index ratio.
    reference_rows = pd.concat([prepared.iloc[[0]]] * len(monthly), ignore_index=True)
    reference_rows["transaction_month"] = monthly["month"].dt.strftime("%Y-%m")
    reference_log_prices = model.predict_log_price(reference_rows)

    monthly["raw_price_index"] = (
        monthly["raw_median_price"] / monthly["raw_median_price"].iloc[0] * 100.0
    )
    monthly["quality_adjusted_price_index"] = (
        np.exp(reference_log_prices - reference_log_prices[0]) * 100.0
    )
    monthly["raw_monthly_change_pct"] = monthly["raw_price_index"].pct_change() * 100
    monthly["quality_adjusted_monthly_change_pct"] = (
        monthly["quality_adjusted_price_index"].pct_change() * 100
    )
    monthly["raw_change_from_base_pct"] = monthly["raw_price_index"] - 100.0
    monthly["quality_adjusted_change_from_base_pct"] = (
        monthly["quality_adjusted_price_index"] - 100.0
    )
    return monthly, model


def _write_index_plot(
    index: pd.DataFrame,
    output_path: Path,
    completeness: dict[str, object],
) -> None:
    """Save the raw-versus-adjusted index comparison chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(
        index["month"],
        index["raw_price_index"],
        color="#8C8C8C",
        linewidth=2.0,
        label="Raw monthly median",
    )
    axis.plot(
        index["month"],
        index["quality_adjusted_price_index"],
        color="#C83E4D",
        linewidth=2.4,
        label="Quality-adjusted model index",
    )
    axis.axhline(100.0, color="#333333", linewidth=0.8, alpha=0.55)
    axis.set_title("HDB resale price trends: raw vs quality-adjusted")
    axis.set_xlabel("")
    axis.set_ylabel(f"Index ({index['month'].iloc[0]:%b %Y} = 100)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)

    last_month = pd.Timestamp(index["month"].iloc[-1])
    latest_note = f"Index through {last_month:%b %Y}. "
    if completeness["provisional_latest_month_excluded"]:
        excluded_month = pd.Timestamp(str(completeness["excluded_month"]))
        latest_note += (
            f"Incomplete {excluded_month:%b %Y} source month excluded "
            f"({int(completeness['excluded_rows']):,} records). "
        )
    if last_month.month != 12:
        latest_note += f"{last_month.year} is a partial calendar year. "
    figure.text(
        0.01,
        0.01,
        latest_note + "Descriptive model-based index; not a causal estimate.",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def read_model_input(input_path: Path) -> pd.DataFrame:
    """Read a cleaned CSV or Parquet transaction snapshot."""
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError(
        f"Unsupported model input format {input_path.suffix!r}; use CSV or Parquet"
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for model-input provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_output_path(path: Path) -> Path:
    """Return a unique sibling path while preserving the output extension."""
    return path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")


def _write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
    **to_csv_kwargs: object,
) -> None:
    """Publish a CSV only after the complete temporary file is written."""
    temporary_path = _temporary_output_path(path)
    to_csv_kwargs.setdefault("lineterminator", "\n")
    try:
        frame.to_csv(temporary_path, **to_csv_kwargs)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    """Publish text only after the complete temporary file is written."""
    temporary_path = _temporary_output_path(path)
    try:
        temporary_path.write_text(text, encoding="utf-8", newline="\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalise_as_of(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive Singapore timestamp for market-month comparisons."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("as_of must be a valid date or timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(SINGAPORE_TIME_ZONE).tz_localize(None)
    return timestamp


def resolve_snapshot_as_of(
    input_path: Path,
    input_sha256: str,
    explicit_as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, str]:
    """Resolve an as-of timestamp from an override, matching manifest, or clock."""
    if explicit_as_of is not None:
        return _normalise_as_of(explicit_as_of), "explicit --as-of"

    metadata_path = input_path.parent / "snapshot_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            hash_key = (
                "processed_parquet_sha256"
                if input_path.suffix.lower() in {".parquet", ".pq"}
                else "processed_sha256"
            )
            if metadata.get(hash_key) == input_sha256:
                for timestamp_key in (
                    "source_modified_at_utc",
                    "generated_at_utc",
                ):
                    if metadata.get(timestamp_key):
                        return (
                            _normalise_as_of(str(metadata[timestamp_key])),
                            f"snapshot_metadata.json:{timestamp_key}",
                        )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    return (
        pd.Timestamp.now(tz=SINGAPORE_TIME_ZONE).tz_localize(None),
        "runtime clock (Asia/Singapore)",
    )


def exclude_provisional_latest_month(
    df: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Exclude the as-of month because it cannot yet be a complete month."""
    if df.empty:
        raise ValueError("Cannot model an empty transaction dataset")

    parsed_months = pd.to_datetime(df["month"], errors="coerce")
    if parsed_months.isna().any():
        raise ValueError(
            f"month contains {int(parsed_months.isna().sum())} invalid values"
        )
    month_periods = parsed_months.dt.to_period("M")
    latest_period = month_periods.max()
    as_of_timestamp = _normalise_as_of(as_of)
    as_of_period = as_of_timestamp.to_period("M")
    if latest_period > as_of_period:
        raise ValueError(
            f"Latest transaction month {latest_period} is after as-of month {as_of_period}"
        )

    exclusion_mask = month_periods.eq(as_of_period)
    filtered = df.loc[~exclusion_mask].copy()
    if filtered.empty:
        raise ValueError("Partial-month exclusion removed every transaction row")

    excluded_rows = int(exclusion_mask.sum())
    completeness: dict[str, object] = {
        "snapshot_as_of_singapore": as_of_timestamp.isoformat(),
        "source_latest_month": str(latest_period),
        "source_latest_month_rows": int(month_periods.eq(latest_period).sum()),
        "provisional_latest_month_excluded": bool(excluded_rows),
        "excluded_month": str(as_of_period) if excluded_rows else None,
        "excluded_rows": excluded_rows,
        "analysis_latest_month": str(month_periods.loc[~exclusion_mask].max()),
    }
    return filtered, completeness


def _portable_output_path(path: Path) -> str:
    """Prefer repository-relative paths in persisted, shareable reports."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_analysis(
    input_path: Path = DEFAULT_INPUT_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    *,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    importance_sample_size: int = DEFAULT_IMPORTANCE_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
    as_of: str | pd.Timestamp | None = None,
    index_ridge_alpha: float = DEFAULT_INDEX_RIDGE_ALPHA,
    backtest_splits: int = DEFAULT_BACKTEST_SPLITS,
    backtest_min_train_months: int = DEFAULT_BACKTEST_MIN_TRAIN_MONTHS,
    backtest_test_months: int | None = None,
    interval_calibration_months: int = DEFAULT_INTERVAL_CALIBRATION_MONTHS,
) -> dict[str, object]:
    """Run modelling, save reproducible reports, and return the summary."""
    source = read_model_input(input_path)
    input_digest = file_sha256(input_path)
    as_of_timestamp, as_of_source = resolve_snapshot_as_of(
        input_path,
        input_digest,
        explicit_as_of=as_of,
    )
    analysis_source, completeness = exclude_provisional_latest_month(
        source,
        as_of_timestamp,
    )
    evaluation = evaluate_chronological_holdout(
        analysis_source,
        holdout_months=holdout_months,
        ridge_alpha=ridge_alpha,
        permutation_repeats=permutation_repeats,
        importance_sample_size=importance_sample_size,
        random_seed=random_seed,
        interval_calibration_months=interval_calibration_months,
    )
    resolved_backtest_months = (
        holdout_months if backtest_test_months is None else backtest_test_months
    )
    analysis_month_count = (
        pd.to_datetime(analysis_source["month"]).dt.to_period("M").nunique()
    )
    effective_min_train_months = min(
        backtest_min_train_months,
        analysis_month_count - resolved_backtest_months,
    )
    backtests = evaluate_rolling_backtests(
        analysis_source,
        test_months=resolved_backtest_months,
        max_splits=backtest_splits,
        min_train_months=effective_min_train_months,
        ridge_alpha=ridge_alpha,
        interval_calibration_months=interval_calibration_months,
    )

    index, _ = derive_quality_adjusted_index(
        analysis_source,
        ridge_alpha=index_ridge_alpha,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "price_model_metrics.json"
    importance_path = reports_dir / "price_model_permutation_importance.csv"
    index_path = reports_dir / "quality_adjusted_price_index.csv"
    backtest_path = reports_dir / "price_model_backtests.csv"
    error_slices_path = reports_dir / "price_model_error_slices.csv"
    interval_metrics_path = reports_dir / "price_model_interval_metrics.csv"
    figure_path = images_dir / "raw_vs_quality_adjusted_price_index.png"

    _write_csv_atomic(
        index,
        index_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )
    _write_csv_atomic(
        evaluation.permutation_importance,
        importance_path,
        index=False,
        float_format="%.6f",
    )
    _write_csv_atomic(
        backtests.fold_metrics,
        backtest_path,
        index=False,
        float_format="%.6f",
    )
    _write_csv_atomic(
        evaluation.error_slices,
        error_slices_path,
        index=False,
        float_format="%.6f",
    )
    _write_csv_atomic(
        backtests.interval_metrics,
        interval_metrics_path,
        index=False,
        float_format="%.6f",
    )

    temporary_figure = _temporary_output_path(figure_path)
    try:
        _write_index_plot(index, temporary_figure, completeness)
        temporary_figure.replace(figure_path)
    finally:
        temporary_figure.unlink(missing_ok=True)

    baseline_mae = evaluation.metrics["training_median_baseline"]["mae"]
    recent_segment_baseline_mae = evaluation.metrics[
        "prior_12m_town_flat_type_median_baseline"
    ]["mae"]
    model_mae = evaluation.metrics["hedonic_ridge"]["mae"]
    mae_improvement_pct = (
        (baseline_mae - model_mae) / baseline_mae * 100.0 if baseline_mae else 0.0
    )
    segment_mae_improvement_pct = (
        (recent_segment_baseline_mae - model_mae) / recent_segment_baseline_mae * 100.0
        if recent_segment_baseline_mae
        else 0.0
    )

    rolling_metric_summary: dict[str, dict[str, float | int]] = {}
    for model_name, model_results in backtests.fold_metrics.groupby(
        "model",
        sort=True,
    ):
        rolling_metric_summary[str(model_name)] = {
            "folds": int(len(model_results)),
            "mean_mae": float(model_results["mae"].mean()),
            "mean_median_absolute_error": float(
                model_results["median_absolute_error"].mean()
            ),
            "mean_p90_absolute_error": float(
                model_results["p90_absolute_error"].mean()
            ),
            "mean_rmse": float(model_results["rmse"].mean()),
            "mean_r2": float(model_results["r2"].mean()),
            "mean_mape_pct": float(model_results["mape_pct"].mean()),
        }

    summary: dict[str, object] = {
        "input_provenance": {
            "path": _portable_output_path(input_path),
            "sha256": input_digest,
            "source_rows": len(source),
            "analysis_rows": len(analysis_source),
            "source_month_min": pd.to_datetime(source["month"]).min().strftime("%Y-%m"),
            "source_month_max": pd.to_datetime(source["month"]).max().strftime("%Y-%m"),
            "as_of_source": as_of_source,
            **completeness,
        },
        "methodology": {
            "target": "natural logarithm of resale_price",
            "estimator": "ridge-regularized hedonic regression for holdout prediction",
            "index_estimator": (
                "unpenalized hedonic linear regression with transaction-month "
                "fixed effects"
                if index_ridge_alpha == 0
                else "ridge-regularized hedonic monthly fixed-effects regression"
            ),
            "quality_controls": [
                "floor area (log quadratic)",
                "remaining lease (quadratic)",
                "storey midpoint (quadratic)",
                "town",
                "flat type",
                "flat model",
            ],
            "evaluation_time_controls": "linear month trend plus annual seasonality",
            "index_time_controls": "transaction-month fixed effects",
            "retransformation": "Duan smearing factor estimated on training residuals",
            "baselines": (
                "Global training median, all-history town x flat-type medians, "
                "and prior-12-month town x flat-type medians with town, flat-type, "
                "then global fallback levels."
            ),
            "rolling_backtests": (
                "Non-overlapping expanding-window folds ending at the latest "
                "complete analysis month."
            ),
            "prediction_intervals": (
                "Centered 10th and 90th percentile log-residual tail spreads "
                "calibrated on the latest pre-test training months, with the "
                "50th percentile anchored to the full-training Ridge estimate. "
                "These are empirical uncertainty ranges, not formal valuations."
            ),
            "error_slices": (
                "Out-of-time MAE, median and 90th-percentile absolute error, "
                "RMSE, R-squared, MAPE and mean error by town, flat type, "
                "observed price band and remaining-lease band."
            ),
            "permutation_importance": (
                "Mean holdout metric degradation after grouped shuffling. "
                "A negative value means shuffling did not worsen that metric, "
                "often because predictors overlap."
            ),
            "interpretation": (
                "Descriptive associations and a transaction-mix-adjusted trend; "
                "not causal effects or formal property valuations."
            ),
            "ridge_alpha": ridge_alpha,
            "index_ridge_alpha": index_ridge_alpha,
            "random_seed": random_seed,
        },
        "split": evaluation.split,
        "holdout_metrics": evaluation.metrics,
        "holdout_mae_improvement_vs_baseline_pct": mae_improvement_pct,
        "holdout_mae_improvement_vs_prior_12m_segment_baseline_pct": (
            segment_mae_improvement_pct
        ),
        "holdout_prediction_interval": evaluation.interval_metrics,
        "rolling_backtests": {
            "folds": int(backtests.fold_metrics["fold"].nunique()),
            "test_months_per_fold": resolved_backtest_months,
            "first_test_month": str(backtests.fold_metrics["test_start_month"].min()),
            "latest_test_month": str(backtests.fold_metrics["test_end_month"].max()),
            "metrics_by_model": rolling_metric_summary,
            "mean_empirical_interval_coverage": float(
                backtests.interval_metrics["empirical_coverage"].mean()
            ),
            "target_interval_coverage": float(
                backtests.interval_metrics["target_coverage"].iloc[0]
            ),
        },
        "index_summary": {
            "base_month": pd.Timestamp(index["month"].iloc[0]).strftime("%Y-%m"),
            "latest_month": pd.Timestamp(index["month"].iloc[-1]).strftime("%Y-%m"),
            "raw_change_from_base_pct": float(
                index["raw_change_from_base_pct"].iloc[-1]
            ),
            "quality_adjusted_change_from_base_pct": float(
                index["quality_adjusted_change_from_base_pct"].iloc[-1]
            ),
            "raw_minus_adjusted_change_percentage_points": float(
                index["raw_change_from_base_pct"].iloc[-1]
                - index["quality_adjusted_change_from_base_pct"].iloc[-1]
            ),
        },
        "outputs": {
            "metrics": _portable_output_path(metrics_path),
            "permutation_importance": _portable_output_path(importance_path),
            "quality_adjusted_index": _portable_output_path(index_path),
            "rolling_backtests": _portable_output_path(backtest_path),
            "error_slices": _portable_output_path(error_slices_path),
            "interval_metrics": _portable_output_path(interval_metrics_path),
            "comparison_chart": _portable_output_path(figure_path),
        },
        "output_sha256": {
            "permutation_importance": file_sha256(importance_path),
            "quality_adjusted_index": file_sha256(index_path),
            "rolling_backtests": file_sha256(backtest_path),
            "error_slices": file_sha256(error_slices_path),
            "interval_metrics": file_sha256(interval_metrics_path),
            "comparison_chart": file_sha256(figure_path),
        },
    }
    _write_text_atomic(
        metrics_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Clean CSV or Parquet path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--as-of",
        help=(
            "Snapshot observation date used to exclude an incomplete current month. "
            "Defaults to a matching snapshot manifest, then the runtime clock."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Report output directory (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Chart output directory (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=DEFAULT_HOLDOUT_MONTHS,
        help="Number of latest transaction months reserved for evaluation",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Non-negative ridge penalty",
    )
    parser.add_argument(
        "--index-ridge-alpha",
        type=float,
        default=DEFAULT_INDEX_RIDGE_ALPHA,
        help=(
            "Penalty for the fixed-effects index. The default 0 leaves month effects "
            "unpenalized; nonzero values are intended only for sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=DEFAULT_PERMUTATION_REPEATS,
        help="Number of deterministic shuffles per feature group",
    )
    parser.add_argument(
        "--importance-sample-size",
        type=int,
        default=DEFAULT_IMPORTANCE_SAMPLE_SIZE,
        help="Maximum holdout rows used for permutation importance",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=DEFAULT_BACKTEST_SPLITS,
        help="Maximum number of historical expanding-window folds",
    )
    parser.add_argument(
        "--backtest-min-train-months",
        type=int,
        default=DEFAULT_BACKTEST_MIN_TRAIN_MONTHS,
        help="Preferred minimum training history for rolling backtests",
    )
    parser.add_argument(
        "--backtest-test-months",
        type=int,
        help="Months per backtest fold; defaults to --holdout-months",
    )
    parser.add_argument(
        "--interval-calibration-months",
        type=int,
        default=DEFAULT_INTERVAL_CALIBRATION_MONTHS,
        help="Latest pre-test training months used for residual-quantile calibration",
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line modelling workflow."""
    args = parse_args()
    summary = run_analysis(
        input_path=args.input,
        reports_dir=args.reports_dir,
        images_dir=args.images_dir,
        holdout_months=args.holdout_months,
        ridge_alpha=args.ridge_alpha,
        permutation_repeats=args.permutation_repeats,
        importance_sample_size=args.importance_sample_size,
        as_of=args.as_of,
        index_ridge_alpha=args.index_ridge_alpha,
        backtest_splits=args.backtest_splits,
        backtest_min_train_months=args.backtest_min_train_months,
        backtest_test_months=args.backtest_test_months,
        interval_calibration_months=args.interval_calibration_months,
    )
    baseline = summary["holdout_metrics"]["training_median_baseline"]
    model = summary["holdout_metrics"]["hedonic_ridge"]
    print(
        "Chronological holdout: "
        f"{summary['split']['holdout_start_month']} to "
        f"{summary['split']['holdout_end_month']}"
    )
    provenance = summary["input_provenance"]
    if provenance["provisional_latest_month_excluded"]:
        print(
            "Excluded incomplete month: "
            f"{provenance['excluded_month']} ({provenance['excluded_rows']:,} rows)"
        )
    print(
        "Baseline  "
        f"MAE ${baseline['mae']:,.0f} | RMSE ${baseline['rmse']:,.0f} | "
        f"R2 {baseline['r2']:.3f}"
    )
    print(
        "Hedonic   "
        f"MAE ${model['mae']:,.0f} | RMSE ${model['rmse']:,.0f} | "
        f"R2 {model['r2']:.3f}"
    )
    print(f"Metrics: {summary['outputs']['metrics']}")
    print(f"Index: {summary['outputs']['quality_adjusted_index']}")
    print(f"Chart: {summary['outputs']['comparison_chart']}")


if __name__ == "__main__":
    main()
