"""Data preparation and aggregations for the Streamlit dashboard.

The functions in this module intentionally do not import Streamlit or Plotly.
Keeping the calculations separate makes the dashboard fast to rerun and lets
the analytical definitions be tested without starting a web server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "hdb_resale_clean.csv"
)
DEFAULT_PARQUET_PATH = DEFAULT_DATA_PATH.with_suffix(".parquet")

BASE_COLUMNS = {
    "month",
    "town",
    "flat_type",
    "floor_area_sqm",
    "resale_price",
}
SNAPSHOT_STALE_AFTER_DAYS = 45


class DashboardDataError(ValueError):
    """Raised when a CSV cannot support the dashboard calculations."""


@dataclass(frozen=True)
class SnapshotFreshness:
    """Freshness details for a checksum-verified authoritative snapshot."""

    observed_at_sgt: pd.Timestamp
    age_days: int
    is_stale: bool


@dataclass(frozen=True)
class ComparableSalesResult:
    """Comparable transactions plus an explanation of the matching tier used."""

    transactions: pd.DataFrame
    match_level: str
    explanation: str
    requested_months: int
    effective_months: int
    floor_area_tolerance: float
    storey_tolerance: float | None
    lease_tolerance: float | None


def assess_snapshot_freshness(
    metadata: Mapping[str, object],
    source_sha256: str,
    *,
    equivalent_hashes: Sequence[str] = (),
    now: object | None = None,
) -> SnapshotFreshness | None:
    """Return source freshness only when provenance and timestamp are trustworthy.

    The clean CSV and Parquet hashes in the tracked manifest are authoritative.
    Callers may supply other hashes only after independently verifying that they
    are equivalent outputs, such as the enriched Parquet snapshot.
    """
    authoritative_hashes = {
        value
        for value in (
            metadata.get("processed_sha256"),
            metadata.get("processed_parquet_sha256"),
            *equivalent_hashes,
        )
        if isinstance(value, str) and value
    }
    if source_sha256 not in authoritative_hashes:
        return None

    timestamp_value = metadata.get("source_modified_at_utc")
    if timestamp_value is None:
        return None
    try:
        observed_at = pd.Timestamp(timestamp_value)
        current_time = (
            pd.Timestamp.now(tz="Asia/Singapore")
            if now is None
            else pd.Timestamp(now)
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(observed_at) or pd.isna(current_time):
        return None

    if observed_at.tzinfo is None:
        observed_at = observed_at.tz_localize("UTC")
    observed_at_sgt = observed_at.tz_convert("Asia/Singapore")
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("Asia/Singapore")
    else:
        current_time = current_time.tz_convert("Asia/Singapore")

    age_days = max(0, (current_time.date() - observed_at_sgt.date()).days)
    return SnapshotFreshness(
        observed_at_sgt=observed_at_sgt,
        age_days=age_days,
        is_stale=age_days > SNAPSHOT_STALE_AFTER_DAYS,
    )


def parse_remaining_lease_months(values: pd.Series) -> pd.Series:
    """Convert HDB lease descriptions into a numeric number of months.

    Examples accepted by the source include ``74 years`` and
    ``74 years 03 months``. Plain numeric values are treated as years so that
    already simplified extracts remain usable.
    """
    text = values.astype("string").str.strip().str.lower()
    years = pd.to_numeric(
        text.str.extract(r"(\d+)\s*years?", expand=False), errors="coerce"
    )
    months = pd.to_numeric(
        text.str.extract(r"(\d+)\s*months?", expand=False), errors="coerce"
    ).fillna(0)
    plain_years = pd.to_numeric(
        text.where(text.str.fullmatch(r"\d+(?:\.\d+)?")), errors="coerce"
    )
    years = years.fillna(plain_years)
    result = years * 12 + months
    return result.astype("float64")


def _require_numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column or raise a concise data-quality error."""
    converted = pd.to_numeric(data[column], errors="coerce")
    invalid_count = int(converted.isna().sum())
    if invalid_count:
        raise DashboardDataError(
            f"Column '{column}' contains {invalid_count:,} missing or non-numeric values. "
            "Re-run the cleaning pipeline before opening the dashboard."
        )
    return converted


def prepare_dashboard_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich a clean HDB transaction DataFrame.

    The normal cleaning pipeline already creates most derived columns. This
    function derives them again when reasonable so the dashboard also works
    with compatible extracts and uploaded files.
    """
    missing = sorted(BASE_COLUMNS.difference(data.columns))
    if missing:
        raise DashboardDataError(
            "The selected CSV is missing required columns: " + ", ".join(missing)
        )
    if data.empty:
        raise DashboardDataError("The selected CSV has no transaction rows.")

    prepared = data.copy()
    prepared["month"] = pd.to_datetime(prepared["month"], errors="coerce")
    invalid_months = int(prepared["month"].isna().sum())
    if invalid_months:
        raise DashboardDataError(
            f"Column 'month' contains {invalid_months:,} invalid dates. "
            "Re-run the cleaning pipeline before opening the dashboard."
        )
    prepared["month"] = prepared["month"].dt.to_period("M").dt.to_timestamp()

    for column in ("town", "flat_type"):
        prepared[column] = prepared[column].astype("string").str.strip().str.upper()
        invalid_text = int(
            prepared[column].isna().sum() + prepared[column].eq("").sum()
        )
        if invalid_text:
            raise DashboardDataError(
                f"Column '{column}' contains {invalid_text:,} missing or blank values."
            )

    prepared["resale_price"] = _require_numeric(prepared, "resale_price")
    prepared["floor_area_sqm"] = _require_numeric(prepared, "floor_area_sqm")
    if (prepared["resale_price"] <= 0).any():
        raise DashboardDataError("Column 'resale_price' must contain positive values.")
    if (prepared["floor_area_sqm"] <= 0).any():
        raise DashboardDataError(
            "Column 'floor_area_sqm' must contain positive values."
        )

    prepared["year"] = prepared["month"].dt.year.astype("int64")
    prepared["price_per_sqm"] = prepared["resale_price"] / prepared["floor_area_sqm"]

    if "storey_mid" in prepared.columns:
        prepared["storey_mid"] = pd.to_numeric(prepared["storey_mid"], errors="coerce")
    elif "storey_range" in prepared.columns:
        storeys = (
            prepared["storey_range"]
            .astype("string")
            .str.extract(r"^\s*(\d+)\s+TO\s+(\d+)\s*$", expand=True)
        )
        prepared["storey_mid"] = (
            pd.to_numeric(storeys[0], errors="coerce")
            + pd.to_numeric(storeys[1], errors="coerce")
        ) / 2
    else:
        prepared["storey_mid"] = np.nan

    if "remaining_lease_months" in prepared.columns:
        prepared["remaining_lease_months"] = pd.to_numeric(
            prepared["remaining_lease_months"], errors="coerce"
        )
    elif "remaining_lease" in prepared.columns:
        prepared["remaining_lease_months"] = parse_remaining_lease_months(
            prepared["remaining_lease"]
        )
    elif "flat_age" in prepared.columns:
        flat_age = pd.to_numeric(prepared["flat_age"], errors="coerce")
        prepared["remaining_lease_months"] = (99 - flat_age) * 12
    else:
        prepared["remaining_lease_months"] = np.nan

    prepared["remaining_lease_years"] = prepared["remaining_lease_months"] / 12
    prepared["is_million_dollar"] = prepared["resale_price"] >= 1_000_000

    if "flat_age" not in prepared.columns and "lease_commence_date" in prepared.columns:
        lease_start = pd.to_numeric(prepared["lease_commence_date"], errors="coerce")
        prepared["flat_age"] = prepared["year"] - lease_start

    optional_aliases = {
        "resale_price_real_sgd": (
            "resale_price_real_sgd",
            "real_resale_price",
            "cpi_adjusted_resale_price",
            "resale_price_real",
        ),
        "price_per_sqm_real_sgd": (
            "price_per_sqm_real_sgd",
            "real_price_per_sqm",
            "cpi_adjusted_price_per_sqm",
        ),
        "cpi_all_items": ("cpi_all_items", "cpi_index"),
        "latitude": ("latitude", "lat"),
        "longitude": ("longitude", "longitude_dd", "lon", "lng"),
        "nearest_mrt_distance_m": (
            "nearest_mrt_distance_m",
            "mrt_distance_m",
        ),
        "lower_price": (
            "lower_price",
            "prediction_lower",
            "predicted_price_lower",
        ),
        "median_price": (
            "median_price",
            "prediction_median",
            "predicted_price",
        ),
        "upper_price": (
            "upper_price",
            "prediction_upper",
            "predicted_price_upper",
        ),
    }
    for canonical, aliases in optional_aliases.items():
        source = next((alias for alias in aliases if alias in prepared.columns), None)
        if source is not None:
            prepared[canonical] = pd.to_numeric(prepared[source], errors="coerce")

    if "nearest_mrt_distance_m" not in prepared.columns:
        distance_km_source = next(
            (
                column
                for column in ("nearest_mrt_distance_km", "mrt_distance_km")
                if column in prepared.columns
            ),
            None,
        )
        if distance_km_source is not None:
            prepared["nearest_mrt_distance_m"] = (
                pd.to_numeric(prepared[distance_km_source], errors="coerce") * 1_000
            )
    if "nearest_mrt_distance_m" in prepared.columns:
        prepared["nearest_mrt_distance_km"] = (
            prepared["nearest_mrt_distance_m"] / 1_000
        )

    if (
        "resale_price_real_sgd" in prepared.columns
        and "price_per_sqm_real_sgd" not in prepared.columns
    ):
        prepared["price_per_sqm_real_sgd"] = (
            prepared["resale_price_real_sgd"] / prepared["floor_area_sqm"]
        )

    if "nearest_mrt_station" not in prepared.columns:
        station_source = next(
            (
                column
                for column in ("nearest_mrt_name", "mrt_station")
                if column in prepared.columns
            ),
            None,
        )
        if station_source is not None:
            prepared["nearest_mrt_station"] = prepared[station_source]
    for column in ("nearest_mrt_station", "mrt_distance_band"):
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string").str.strip()

    return prepared.sort_values("month", kind="stable").reset_index(drop=True)


def find_comparable_sales(
    data: pd.DataFrame,
    *,
    town: str,
    flat_type: str,
    floor_area_sqm: float,
    storey_mid: float | None,
    remaining_lease_years: float | None,
    recent_months: int = 24,
    minimum_transactions: int = 20,
) -> ComparableSalesResult:
    """Find recent comparable transactions with explicit, staged fallbacks.

    Town and flat type always match exactly. The first tier uses narrow property
    bands; later tiers widen those bands and, only when needed, extend history.
    The returned explanation is suitable for showing directly in the dashboard.
    """
    if recent_months < 1:
        raise ValueError("recent_months must be positive")
    if minimum_transactions < 1:
        raise ValueError("minimum_transactions must be positive")
    if floor_area_sqm <= 0:
        raise ValueError("floor_area_sqm must be positive")
    if data.empty:
        return ComparableSalesResult(
            transactions=data.copy(),
            match_level="No matches",
            explanation="The selected snapshot contains no transactions.",
            requested_months=recent_months,
            effective_months=recent_months,
            floor_area_tolerance=5,
            storey_tolerance=3 if storey_mid is not None else None,
            lease_tolerance=5 if remaining_lease_years is not None else None,
        )

    latest_month = pd.Timestamp(data["month"].max()).to_period("M").to_timestamp()
    base = data.loc[
        data["town"].eq(str(town).strip().upper())
        & data["flat_type"].eq(str(flat_type).strip().upper())
    ]

    tiers = [
        ("Close match", recent_months, 5.0, 3.0, 5.0),
        ("Wider property bands", recent_months, 10.0, 6.0, 10.0),
        ("Longer history", max(recent_months, 60), 10.0, 6.0, 10.0),
        ("Broad town-and-type match", max(recent_months, 60), 15.0, None, 15.0),
    ]

    last_result = base.iloc[0:0].copy()
    selected_tier = tiers[-1]
    for tier in tiers:
        label, months, area_tolerance, storey_tolerance, lease_tolerance = tier
        cutoff = latest_month - pd.DateOffset(months=months - 1)
        mask = base["month"].ge(cutoff)
        mask &= base["floor_area_sqm"].between(
            floor_area_sqm - area_tolerance,
            floor_area_sqm + area_tolerance,
            inclusive="both",
        )
        if storey_mid is not None and storey_tolerance is not None:
            mask &= base["storey_mid"].between(
                storey_mid - storey_tolerance,
                storey_mid + storey_tolerance,
                inclusive="both",
            )
        if remaining_lease_years is not None and lease_tolerance is not None:
            mask &= base["remaining_lease_years"].between(
                remaining_lease_years - lease_tolerance,
                remaining_lease_years + lease_tolerance,
                inclusive="both",
            )
        last_result = base.loc[mask].copy()
        selected_tier = tier
        if len(last_result) >= minimum_transactions:
            break

    label, months, area_tolerance, storey_tolerance, lease_tolerance = selected_tier
    if last_result.empty:
        explanation = (
            "No transactions matched even after widening the property bands. "
            "Try another town, flat type or floor area."
        )
    else:
        band_details = [f"floor area ±{area_tolerance:g} sqm"]
        if storey_mid is not None and storey_tolerance is not None:
            band_details.append(f"storey midpoint ±{storey_tolerance:g}")
        if remaining_lease_years is not None and lease_tolerance is not None:
            band_details.append(f"remaining lease ±{lease_tolerance:g} years")
        explanation = (
            f"{label}: exact town and flat type, "
            + ", ".join(band_details)
            + f", using the latest {months} months in the snapshot."
        )
        if len(last_result) < minimum_transactions:
            explanation += (
                f" Only {len(last_result):,} matches were available, below the "
                f"{minimum_transactions:,}-transaction stability target."
            )
        elif label != "Close match":
            explanation += (
                f" The criteria were widened because the closer tiers contained "
                f"fewer than {minimum_transactions:,} transactions."
            )

    return ComparableSalesResult(
        transactions=last_result.sort_values("month", ascending=False, kind="stable")
        .reset_index(drop=True),
        match_level=label,
        explanation=explanation,
        requested_months=recent_months,
        effective_months=months,
        floor_area_tolerance=area_tolerance,
        storey_tolerance=storey_tolerance if storey_mid is not None else None,
        lease_tolerance=(
            lease_tolerance if remaining_lease_years is not None else None
        ),
    )


def load_processed_csv(path_or_buffer: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read and prepare a processed CSV path or uploaded binary buffer."""
    try:
        data = pd.read_csv(path_or_buffer, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DashboardDataError(f"Could not read the selected CSV: {exc}") from exc
    return prepare_dashboard_data(data)


def load_processed_data(
    path_or_buffer: str | Path | IO[bytes],
    *,
    file_format: str | None = None,
) -> pd.DataFrame:
    """Read a processed Parquet/CSV source and prepare it for the dashboard."""
    inferred_format = file_format
    if inferred_format is None and isinstance(path_or_buffer, (str, Path)):
        inferred_format = Path(path_or_buffer).suffix
    normalised_format = str(inferred_format or ".csv").lower().lstrip(".")

    if normalised_format in {"parquet", "pq"}:
        try:
            data = pd.read_parquet(path_or_buffer)
        except (OSError, ImportError, ValueError) as exc:
            raise DashboardDataError(
                f"Could not read the selected Parquet file: {exc}"
            ) from exc
        return prepare_dashboard_data(data)
    if normalised_format == "csv":
        return load_processed_csv(path_or_buffer)
    raise DashboardDataError(
        f"Unsupported data format '.{normalised_format}'. Upload a CSV or Parquet file."
    )


def filter_transactions(
    data: pd.DataFrame,
    *,
    years: Sequence[int] | None = None,
    towns: Sequence[str] | None = None,
    flat_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return rows matching the selected dashboard filters."""
    mask = pd.Series(True, index=data.index)
    if years is not None:
        mask &= data["year"].isin(years)
    if towns is not None:
        mask &= data["town"].isin(towns)
    if flat_types is not None:
        mask &= data["flat_type"].isin(flat_types)
    return data.loc[mask].copy()


def coverage_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed months and mark years with fewer than 12 months."""
    coverage = (
        data.groupby("year", as_index=False)
        .agg(
            months_observed=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            transactions=("month", "size"),
        )
        .sort_values("year")
    )
    coverage["is_partial"] = coverage["months_observed"] < 12
    return coverage.reset_index(drop=True)


def monthly_trend(data: pd.DataFrame) -> pd.DataFrame:
    """Return monthly price, price-per-area and volume statistics."""
    trend = (
        data.groupby("month", as_index=False)
        .agg(
            median_price=("resale_price", "median"),
            price_q25=("resale_price", lambda values: values.quantile(0.25)),
            price_q75=("resale_price", lambda values: values.quantile(0.75)),
            median_price_per_sqm=("price_per_sqm", "median"),
            transactions=("resale_price", "size"),
            million_dollar_share=("is_million_dollar", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    if "resale_price_real_sgd" in data.columns:
        real_trend = (
            data.groupby("month", as_index=False)
            .agg(
                median_real_price=("resale_price_real_sgd", "median"),
                real_price_q25=(
                    "resale_price_real_sgd",
                    lambda values: values.quantile(0.25),
                ),
                real_price_q75=(
                    "resale_price_real_sgd",
                    lambda values: values.quantile(0.75),
                ),
            )
            .sort_values("month")
        )
        trend = trend.merge(real_trend, on="month", how="left", validate="one_to_one")
    return trend


def annual_trend(data: pd.DataFrame) -> pd.DataFrame:
    """Return annual metrics with coverage information."""
    annual = (
        data.groupby("year", as_index=False)
        .agg(
            median_price=("resale_price", "median"),
            price_q25=("resale_price", lambda values: values.quantile(0.25)),
            price_q75=("resale_price", lambda values: values.quantile(0.75)),
            median_price_per_sqm=("price_per_sqm", "median"),
            transactions=("resale_price", "size"),
            million_dollar_share=("is_million_dollar", "mean"),
            months_observed=("month", "nunique"),
        )
        .sort_values("year")
    )
    annual["is_partial"] = annual["months_observed"] < 12
    return annual.reset_index(drop=True)


def town_summary(data: pd.DataFrame, *, minimum_transactions: int = 1) -> pd.DataFrame:
    """Rank towns using robust price and sample-size measures."""
    summary = data.groupby("town", as_index=False).agg(
        median_price=("resale_price", "median"),
        median_price_per_sqm=("price_per_sqm", "median"),
        transactions=("resale_price", "size"),
        million_dollar_share=("is_million_dollar", "mean"),
    )
    return summary.loc[summary["transactions"] >= minimum_transactions].reset_index(
        drop=True
    )


def flat_type_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact five-number price summary for each flat type."""
    return (
        data.groupby("flat_type", as_index=False)
        .agg(
            price_q10=("resale_price", lambda values: values.quantile(0.10)),
            price_q25=("resale_price", lambda values: values.quantile(0.25)),
            median_price=("resale_price", "median"),
            price_q75=("resale_price", lambda values: values.quantile(0.75)),
            price_q90=("resale_price", lambda values: values.quantile(0.90)),
            transactions=("resale_price", "size"),
        )
        .sort_values("median_price")
        .reset_index(drop=True)
    )


def price_histogram(data: pd.DataFrame, *, bins: int = 36) -> pd.DataFrame:
    """Aggregate resale prices into bins without sending every row to Plotly."""
    prices = data["resale_price"].dropna().to_numpy(dtype="float64")
    if not len(prices):
        return pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_mid", "transactions"]
        )
    counts, edges = np.histogram(prices, bins=bins)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_mid": (edges[:-1] + edges[1:]) / 2,
            "transactions": counts,
        }
    )


def binned_price_profile(
    data: pd.DataFrame,
    column: str,
    *,
    bin_width: float,
) -> pd.DataFrame:
    """Return median prices and sample sizes for equal-width numeric bins."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    values = pd.to_numeric(data[column], errors="coerce")
    valid = data.loc[values.notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "bin_left",
                "bin_mid",
                "median_price",
                "median_price_per_sqm",
                "transactions",
            ]
        )
    valid["bin_left"] = np.floor(values.loc[valid.index] / bin_width) * bin_width
    profile = (
        valid.groupby("bin_left", as_index=False)
        .agg(
            median_price=("resale_price", "median"),
            median_price_per_sqm=("price_per_sqm", "median"),
            transactions=("resale_price", "size"),
        )
        .sort_values("bin_left")
    )
    profile["bin_mid"] = profile["bin_left"] + bin_width / 2
    return profile.reset_index(drop=True)
