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
    squared_error = float(np.dot(error, error))
    centered_actual = actual_values - actual_values.mean()
    total_variation = float(np.dot(centered_actual, centered_actual))
    if total_variation == 0:
        r_squared = 1.0 if squared_error == 0 else 0.0
    else:
        r_squared = 1.0 - squared_error / total_variation
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r_squared),
    }


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
) -> ModelEvaluation:
    """Compare the hedonic model with a training-median holdout baseline."""
    prepared = prepare_model_data(df)
    train, holdout, cutoff = chronological_holdout(prepared, holdout_months)
    origin = pd.Timestamp(train["month"].min())
    train = add_trend_features(train, origin)
    holdout = add_trend_features(holdout, origin)

    model = fit_hedonic_model(train, time_mode="trend", ridge_alpha=ridge_alpha)
    actual = holdout["resale_price"].to_numpy(dtype="float64")
    baseline_price = float(train["resale_price"].median())
    baseline_prediction = np.full(len(holdout), baseline_price, dtype="float64")
    model_prediction = model.predict_price(holdout)
    metrics = {
        "training_median_baseline": regression_metrics(actual, baseline_prediction),
        "hedonic_ridge": regression_metrics(actual, model_prediction),
    }

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
    }
    return ModelEvaluation(
        model=model,
        metrics=metrics,
        permutation_importance=importance,
        split=split,
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
    temporary_figure = _temporary_output_path(figure_path)
    try:
        _write_index_plot(index, temporary_figure, completeness)
        temporary_figure.replace(figure_path)
    finally:
        temporary_figure.unlink(missing_ok=True)

    baseline_mae = evaluation.metrics["training_median_baseline"]["mae"]
    model_mae = evaluation.metrics["hedonic_ridge"]["mae"]
    mae_improvement_pct = (
        (baseline_mae - model_mae) / baseline_mae * 100.0 if baseline_mae else 0.0
    )
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
            "comparison_chart": _portable_output_path(figure_path),
        },
        "output_sha256": {
            "permutation_importance": file_sha256(importance_path),
            "quality_adjusted_index": file_sha256(index_path),
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
