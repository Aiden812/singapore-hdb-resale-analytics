"""Create advanced descriptive and robustness reports for HDB resale data.

The module deliberately keeps raw transaction medians separate from the
official HDB Resale Price Index (RPI).  A raw median can move when the mix of
flats sold changes; it is not a like-for-like property price index and neither
series should be interpreted as identifying a causal effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import requests
from matplotlib.ticker import FuncFormatter, PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.csv"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "hdb_resale.db"
DEFAULT_RPI_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "hdb_resale_rpi.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"

RPI_DATASET_ID = "d_14f63e595975691e7c24a27ae4c07c79"
RPI_SOURCE_URL = f"https://data.gov.sg/datasets/{RPI_DATASET_ID}/view"
RPI_API_URL = "https://data.gov.sg/api/action/datastore_search"
API_KEY_ENV_VAR = "DATA_GOV_SG_API_KEY"

MILLION_DOLLAR_THRESHOLD = 1_000_000.0
DEFAULT_MIN_TOWN_SAMPLE = 100
DEFAULT_LEASE_BIN_YEARS = 5
MATCH_TOLERANCE = 0.01
SINGAPORE_TIME_ZONE = ZoneInfo("Asia/Singapore")

TRANSACTION_REQUIRED_COLUMNS = {
    "month",
    "town",
    "remaining_lease",
    "resale_price",
}
RPI_REQUIRED_COLUMNS = {"quarter", "index"}
LEASE_PATTERN = re.compile(
    r"^\s*(?P<years>\d+)\s+years?"
    r"(?:\s+(?P<months>\d+)\s+months?)?\s*$",
    flags=re.IGNORECASE,
)
SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PALETTE = {
    "blue": "#176B87",
    "orange": "#E07A5F",
    "green": "#2A9D8F",
    "navy": "#264653",
    "gold": "#E9C46A",
    "grey": "#6B7280",
    "light_blue": "#B7DDE8",
}

DESCRIPTIVE_CAVEAT = (
    "Results are descriptive and unadjusted. Changes can reflect the mix of "
    "towns, flat types, sizes, leases and storeys sold; they do not identify "
    "causal effects."
)
DUPLICATE_CAVEAT = (
    "Fully identical rows cannot be confirmed as errors because the source has "
    "no transaction identifier or unit number. Removal is shown only as a "
    "sensitivity check."
)
RPI_CAVEAT = (
    "The raw-median series is composition-sensitive and is not equivalent to "
    "the official HDB Resale Price Index. Rebasing permits a visual comparison "
    "of movement, not a claim that the measures are interchangeable."
)


def _display_path(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_month_disclosure(
    input_path: Path,
    input_sha256: str,
    prepared: pd.DataFrame,
) -> dict[str, object]:
    """Describe whether the latest source month is incomplete at snapshot time."""
    as_of = pd.Timestamp(input_path.stat().st_mtime, unit="s", tz="UTC")
    as_of_source = "input file modified time"
    metadata_path = input_path.parent / "snapshot_metadata.json"
    if metadata_path.is_file():
        try:
            snapshot = json.loads(metadata_path.read_text(encoding="utf-8"))
            if snapshot.get("processed_sha256") == input_sha256:
                timestamp_key = next(
                    (
                        key
                        for key in ("source_modified_at_utc", "generated_at_utc")
                        if snapshot.get(key)
                    ),
                    None,
                )
                if timestamp_key is not None:
                    as_of = pd.Timestamp(snapshot[timestamp_key])
                    as_of_source = f"snapshot_metadata.json:{timestamp_key}"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(SINGAPORE_TIME_ZONE)
    else:
        as_of = as_of.tz_convert(SINGAPORE_TIME_ZONE)
    latest_period = prepared["month"].max().to_period("M")
    as_of_period = as_of.tz_localize(None).to_period("M")
    provisional = latest_period == as_of_period
    provisional_rows = int(prepared["month"].dt.to_period("M").eq(latest_period).sum())
    return {
        "snapshot_as_of_singapore": as_of.isoformat(),
        "as_of_source": as_of_source,
        "latest_month": str(latest_period),
        "latest_month_is_provisional": provisional,
        "provisional_rows": provisional_rows if provisional else 0,
        "definition": (
            "A transaction month matching the snapshot as-of month is provisional "
            "because that calendar month was not complete when the file was observed."
        ),
    }


def parse_remaining_lease_months(value: object) -> int:
    """Parse HDB values such as ``61 years 04 months`` into total months."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid remaining_lease value: {value!r}")
    match = LEASE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid remaining_lease value: {value!r}")

    years = int(match.group("years"))
    months = int(match.group("months") or 0)
    if months > 11:
        raise ValueError(
            f"Invalid remaining_lease month component in {value!r}: {months}"
        )
    return years * 12 + months


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a deterministic analysis-ready transaction frame."""
    missing = sorted(TRANSACTION_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Transaction data is missing required columns: " + ", ".join(missing)
        )

    prepared = df.copy()
    prepared["month"] = pd.to_datetime(prepared["month"], errors="coerce")
    prepared["resale_price"] = pd.to_numeric(prepared["resale_price"], errors="coerce")
    prepared["town"] = prepared["town"].astype("string").str.strip()

    invalid_counts = {
        "month": int(prepared["month"].isna().sum()),
        "resale_price": int(prepared["resale_price"].isna().sum()),
        "town": int(prepared["town"].isna().sum() + prepared["town"].eq("").sum()),
    }
    invalid_counts = {key: value for key, value in invalid_counts.items() if value}
    if invalid_counts:
        details = ", ".join(
            f"{column}={count}" for column, count in invalid_counts.items()
        )
        raise ValueError(f"Invalid transaction values: {details}")
    if (prepared["resale_price"] <= 0).any():
        raise ValueError("resale_price must be greater than zero")

    if "remaining_lease_months" in prepared.columns:
        lease_months = pd.to_numeric(
            prepared["remaining_lease_months"], errors="coerce"
        )
        invalid_lease_months = (
            lease_months.isna() | lease_months.lt(0) | lease_months.mod(1).ne(0)
        )
        if invalid_lease_months.any():
            examples = (
                prepared.loc[invalid_lease_months, "remaining_lease_months"]
                .head(5)
                .tolist()
            )
            raise ValueError(f"Invalid remaining_lease_months values: {examples}")
        prepared["remaining_lease_months"] = lease_months.astype("int64")
    else:
        parsed_leases: list[int] = []
        invalid_leases: list[str] = []
        for row_number, value in enumerate(prepared["remaining_lease"], start=1):
            try:
                parsed_leases.append(parse_remaining_lease_months(value))
            except ValueError:
                if len(invalid_leases) < 5:
                    invalid_leases.append(f"row {row_number}: {value!r}")
        if invalid_leases:
            raise ValueError(
                "Invalid remaining_lease values (first examples): "
                + "; ".join(invalid_leases)
            )

        prepared["remaining_lease_months"] = pd.Series(
            parsed_leases, index=prepared.index, dtype="int64"
        )
    prepared["year"] = prepared["month"].dt.year.astype("int64")
    prepared["month_number"] = prepared["month"].dt.month.astype("int64")
    prepared["quarter"] = (
        prepared["year"].astype(str) + "-Q" + prepared["month"].dt.quarter.astype(str)
    )
    return prepared


def annual_price_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return annual P25/P50/P75/P90 values and calendar coverage."""
    prepared = prepare_transactions(df)
    grouped = prepared.groupby("year", sort=True, observed=True)
    coverage = grouped.agg(
        transaction_count=("resale_price", "size"),
        months_observed=("month_number", "nunique"),
        first_month=("month", "min"),
        latest_month=("month", "max"),
    )
    quantiles = (
        grouped["resale_price"]
        .quantile([0.25, 0.50, 0.75, 0.90])
        .unstack()
        .rename(columns={0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"})
    )
    result = coverage.join(quantiles).reset_index()
    result["first_month"] = result["first_month"].dt.strftime("%Y-%m")
    result["latest_month"] = result["latest_month"].dt.strftime("%Y-%m")
    result["is_complete_year"] = result["months_observed"].eq(12)
    return result[
        [
            "year",
            "transaction_count",
            "months_observed",
            "is_complete_year",
            "first_month",
            "latest_month",
            "p25",
            "p50",
            "p75",
            "p90",
        ]
    ]


def million_dollar_share(
    df: pd.DataFrame,
    group_columns: str | Sequence[str],
    *,
    threshold: float = MILLION_DOLLAR_THRESHOLD,
) -> pd.DataFrame:
    """Return transaction counts and million-dollar share by selected groups."""
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    prepared = prepare_transactions(df)
    columns = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    if not columns:
        raise ValueError("At least one group column is required")
    missing = sorted(set(columns).difference(prepared.columns))
    if missing:
        raise ValueError("Unknown group columns: " + ", ".join(missing))

    working = prepared.assign(
        is_million_dollar=prepared["resale_price"].ge(threshold).astype("int64")
    )
    result = (
        working.groupby(columns, sort=True, observed=True)
        .agg(
            transaction_count=("resale_price", "size"),
            million_dollar_transactions=("is_million_dollar", "sum"),
        )
        .reset_index()
    )
    result["million_dollar_share_pct"] = (
        100.0 * result["million_dollar_transactions"] / result["transaction_count"]
    )
    result["threshold_sgd"] = float(threshold)
    return result.sort_values(columns, kind="stable").reset_index(drop=True)


def remaining_lease_bands(
    df: pd.DataFrame,
    *,
    bin_years: int = DEFAULT_LEASE_BIN_YEARS,
) -> pd.DataFrame:
    """Summarise price distributions by fixed-width remaining-lease bands."""
    if bin_years <= 0:
        raise ValueError("bin_years must be a positive integer")
    prepared = prepare_transactions(df)
    width_months = bin_years * 12
    prepared["lease_bin_start_years"] = (
        prepared["remaining_lease_months"] // width_months * bin_years
    ).astype("int64")

    grouped = prepared.groupby("lease_bin_start_years", sort=True, observed=True)
    summary = (
        grouped["resale_price"]
        .agg(
            transaction_count="size",
            p25=lambda values: values.quantile(0.25),
            median_resale_price="median",
            p75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary["lease_bin_end_years"] = summary["lease_bin_start_years"] + bin_years - 1
    summary["remaining_lease_band"] = (
        summary["lease_bin_start_years"].astype(str)
        + "-"
        + summary["lease_bin_end_years"].astype(str)
        + " years"
    )
    return summary[
        [
            "remaining_lease_band",
            "lease_bin_start_years",
            "lease_bin_end_years",
            "transaction_count",
            "p25",
            "median_resale_price",
            "p75",
        ]
    ]


def complete_calendar_years(df: pd.DataFrame) -> list[int]:
    """Return years containing at least one record in every calendar month."""
    prepared = prepare_transactions(df)
    coverage = prepared.groupby("year", sort=True)["month_number"].nunique()
    return [int(year) for year in coverage[coverage.eq(12)].index]


def town_endpoint_growth(
    df: pd.DataFrame,
    *,
    min_sample: int = DEFAULT_MIN_TOWN_SAMPLE,
    first_year: int | None = None,
    last_year: int | None = None,
) -> pd.DataFrame:
    """Compare town medians at two complete-year endpoints with sample guards."""
    if min_sample <= 0:
        raise ValueError("min_sample must be a positive integer")
    prepared = prepare_transactions(df)
    complete_years = complete_calendar_years(prepared)
    if len(complete_years) < 2:
        raise ValueError("At least two complete calendar years are required")

    selected_first = int(first_year) if first_year is not None else complete_years[0]
    selected_last = int(last_year) if last_year is not None else complete_years[-1]
    if selected_first >= selected_last:
        raise ValueError("first_year must be earlier than last_year")
    incomplete = [
        year for year in (selected_first, selected_last) if year not in complete_years
    ]
    if incomplete:
        raise ValueError(
            "Town-growth endpoints must be complete calendar years: "
            + ", ".join(map(str, incomplete))
        )

    endpoint_data = prepared.loc[prepared["year"].isin([selected_first, selected_last])]
    stats = (
        endpoint_data.groupby(["town", "year"], sort=True, observed=True)[
            "resale_price"
        ]
        .agg([("median", "median"), ("count", "size")])
        .reset_index()
    )
    first_stats = stats.loc[
        stats["year"].eq(selected_first), ["town", "median", "count"]
    ]
    first_stats = first_stats.rename(
        columns={"median": "first_year_median", "count": "first_year_count"}
    )
    last_stats = stats.loc[stats["year"].eq(selected_last), ["town", "median", "count"]]
    last_stats = last_stats.rename(
        columns={"median": "last_year_median", "count": "last_year_count"}
    )
    result = first_stats.merge(last_stats, on="town", how="outer", sort=True)
    result.insert(1, "first_year", selected_first)
    result.insert(2, "last_year", selected_last)
    result["first_year_count"] = result["first_year_count"].fillna(0).astype("int64")
    result["last_year_count"] = result["last_year_count"].fillna(0).astype("int64")
    result["minimum_endpoint_sample"] = int(min_sample)
    result["has_both_endpoints"] = (
        result["first_year_median"].notna() & result["last_year_median"].notna()
    )
    result["meets_minimum_sample"] = (
        result["has_both_endpoints"]
        & result["first_year_count"].ge(min_sample)
        & result["last_year_count"].ge(min_sample)
    )
    growth = (
        100.0
        * (result["last_year_median"] - result["first_year_median"])
        / result["first_year_median"]
    )
    result["median_price_growth_pct"] = growth.where(result["meets_minimum_sample"])
    result["eligibility_status"] = np.select(
        [
            ~result["has_both_endpoints"],
            ~result["meets_minimum_sample"],
        ],
        ["missing_endpoint", "below_minimum_sample"],
        default="eligible",
    )
    return result.sort_values(
        ["meets_minimum_sample", "median_price_growth_pct", "town"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def duplicate_removal_sensitivity(
    df: pd.DataFrame,
    *,
    subset: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare annual and overall medians before/after exact-row removal."""
    prepared = prepare_transactions(df)
    duplicate_subset = list(subset) if subset is not None else list(df.columns)
    missing = sorted(set(duplicate_subset).difference(prepared.columns))
    if missing:
        raise ValueError("Unknown duplicate subset columns: " + ", ".join(missing))

    without_duplicates = prepared.drop_duplicates(subset=duplicate_subset, keep="first")
    with_stats = (
        prepared.groupby("year", sort=True)["resale_price"]
        .agg([("rows_with_duplicates", "size"), ("median_with_duplicates", "median")])
        .reset_index()
    )
    without_stats = (
        without_duplicates.groupby("year", sort=True)["resale_price"]
        .agg(
            [
                ("rows_without_duplicates", "size"),
                ("median_without_duplicates", "median"),
            ]
        )
        .reset_index()
    )
    annual = with_stats.merge(without_stats, on="year", how="outer", sort=True)
    annual["rows_removed"] = (
        annual["rows_with_duplicates"] - annual["rows_without_duplicates"]
    ).astype("int64")
    annual["median_difference_sgd"] = (
        annual["median_without_duplicates"] - annual["median_with_duplicates"]
    )
    annual["median_difference_pct"] = (
        100.0 * annual["median_difference_sgd"] / annual["median_with_duplicates"]
    )

    overall_with = float(prepared["resale_price"].median())
    overall_without = float(without_duplicates["resale_price"].median())
    overall = {
        "rows_with_duplicates": int(len(prepared)),
        "rows_without_duplicates": int(len(without_duplicates)),
        "rows_removed": int(len(prepared) - len(without_duplicates)),
        "median_with_duplicates": overall_with,
        "median_without_duplicates": overall_without,
        "median_difference_sgd": overall_without - overall_with,
        "median_difference_pct": 100.0
        * (overall_without - overall_with)
        / overall_with,
    }
    return annual, overall


def reconcile_python_sql(
    df: pd.DataFrame,
    database_path: Path,
    *,
    table_name: str = "resale_transactions",
    tolerance: float = MATCH_TOLERANCE,
) -> pd.DataFrame:
    """Reconcile annual Python medians with the SQLite analytics table."""
    if not SAFE_SQL_IDENTIFIER.fullmatch(table_name):
        raise ValueError(f"Unsafe SQLite table name: {table_name!r}")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")

    python_medians = annual_price_quantiles(df)[["year", "p50"]].rename(
        columns={"p50": "python_median"}
    )
    query = f"""
        WITH ranked AS (
            SELECT
                year,
                resale_price,
                ROW_NUMBER() OVER (
                    PARTITION BY year ORDER BY resale_price
                ) AS price_position,
                COUNT(*) OVER (PARTITION BY year) AS group_size
            FROM {table_name}
            WHERE resale_price IS NOT NULL
        )
        SELECT
            year,
            AVG(resale_price) AS sql_median
        FROM ranked
        WHERE price_position IN (
            (group_size + 1) / 2,
            (group_size + 2) / 2
        )
        GROUP BY year
        ORDER BY year
    """
    with closing(sqlite3.connect(database_path)) as connection:
        sql_medians = pd.read_sql_query(query, connection)

    result = python_medians.merge(sql_medians, on="year", how="outer", sort=True)
    result["absolute_difference_sgd"] = (
        result["python_median"] - result["sql_median"]
    ).abs()
    result["matches_within_tolerance"] = (
        result["python_median"].notna()
        & result["sql_median"].notna()
        & result["absolute_difference_sgd"].le(tolerance)
    )
    result["tolerance_sgd"] = float(tolerance)
    return result


def normalise_rpi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardise an official HDB quarterly RPI frame."""
    renamed = {
        column: str(column).strip().lower().replace(" ", "_") for column in df.columns
    }
    normalised = df.rename(columns=renamed).copy()
    missing = sorted(RPI_REQUIRED_COLUMNS.difference(normalised.columns))
    if missing:
        raise ValueError("RPI data is missing required columns: " + ", ".join(missing))

    normalised = normalised[["quarter", "index"]]
    normalised["quarter"] = normalised["quarter"].astype("string").str.strip()
    normalised["index"] = pd.to_numeric(normalised["index"], errors="coerce")
    valid_quarter = normalised["quarter"].str.fullmatch(r"\d{4}-Q[1-4]", na=False)
    if not valid_quarter.all():
        examples = normalised.loc[~valid_quarter, "quarter"].head(5).tolist()
        raise ValueError(f"Invalid RPI quarter values: {examples}")
    if normalised["index"].isna().any() or (normalised["index"] <= 0).any():
        raise ValueError("RPI index values must be positive numbers")
    if normalised["quarter"].duplicated().any():
        duplicates = normalised.loc[
            normalised["quarter"].duplicated(keep=False), "quarter"
        ].unique()
        raise ValueError(
            "RPI data contains duplicate quarters: " + ", ".join(duplicates)
        )

    normalised["_quarter_period"] = pd.PeriodIndex(
        normalised["quarter"].str.replace("-", "", regex=False), freq="Q"
    )
    normalised = normalised.sort_values("_quarter_period", kind="stable")
    return normalised.drop(columns="_quarter_period").reset_index(drop=True)


def download_rpi_dataset(
    output_path: Path,
    *,
    force: bool = False,
    page_size: int = 500,
    session: requests.Session | None = None,
) -> int:
    """Download the official quarterly RPI to a validated local CSV snapshot."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass force=True to replace it."
        )
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    headers = {}
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        headers["x-api-key"] = api_key

    client = session or requests.Session()
    owns_session = session is None
    records: list[dict[str, object]] = []
    offset = 0
    try:
        while True:
            response = client.get(
                RPI_API_URL,
                params={
                    "resource_id": RPI_DATASET_ID,
                    "limit": page_size,
                    "offset": offset,
                },
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                raise RuntimeError(f"data.gov.sg RPI request failed: {payload}")
            result = payload.get("result") or {}
            page = result.get("records") or []
            records.extend(page)
            total = int(result.get("total", len(records)))
            if not page or len(records) >= total:
                break
            offset += len(page)

        if not records:
            raise ValueError("Official RPI API returned no records")
        rpi = normalise_rpi_data(pd.DataFrame.from_records(records))
        rpi.to_csv(temporary_path, index=False, lineterminator="\n")
        normalise_rpi_data(pd.read_csv(temporary_path))
        temporary_path.replace(output_path)
        return len(rpi)
    finally:
        temporary_path.unlink(missing_ok=True)
        if owns_session:
            client.close()


def quarterly_rpi_benchmark(
    df: pd.DataFrame,
    rpi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare rebased quarterly transaction medians with the official HDB RPI."""
    prepared = prepare_transactions(df)
    rpi = normalise_rpi_data(rpi_df).rename(columns={"index": "official_rpi"})
    transaction_quarters = (
        prepared.groupby("quarter", sort=True, observed=True)
        .agg(
            transaction_count=("resale_price", "size"),
            months_observed=("month_number", "nunique"),
            raw_median_resale_price=("resale_price", "median"),
        )
        .reset_index()
    )
    transaction_quarters["has_all_three_month_labels"] = transaction_quarters[
        "months_observed"
    ].eq(3)

    benchmark = transaction_quarters.merge(rpi, on="quarter", how="inner")
    if benchmark.empty:
        raise ValueError(
            "Transaction and official RPI data have no overlapping quarters"
        )
    benchmark["_quarter_period"] = pd.PeriodIndex(
        benchmark["quarter"].str.replace("-", "", regex=False), freq="Q"
    )
    benchmark = benchmark.sort_values("_quarter_period", kind="stable").reset_index(
        drop=True
    )
    base_raw = float(benchmark.loc[0, "raw_median_resale_price"])
    base_rpi = float(benchmark.loc[0, "official_rpi"])
    benchmark["raw_median_rebased"] = (
        100.0 * benchmark["raw_median_resale_price"] / base_raw
    )
    benchmark["official_rpi_rebased"] = 100.0 * benchmark["official_rpi"] / base_rpi
    benchmark["rebased_gap_points"] = (
        benchmark["raw_median_rebased"] - benchmark["official_rpi_rebased"]
    )
    benchmark["raw_median_qoq_pct"] = (
        benchmark["raw_median_resale_price"].pct_change() * 100.0
    )
    benchmark["official_rpi_qoq_pct"] = benchmark["official_rpi"].pct_change() * 100.0
    benchmark["base_quarter"] = benchmark.loc[0, "quarter"]
    return benchmark.drop(columns="_quarter_period")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )


def _currency_axis(value: float, _: int) -> str:
    return f"S${value / 1_000:,.0f}k"


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)
    axis.grid(axis="x", visible=False)


def plot_annual_quantiles(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot annual price percentiles while marking incomplete years."""
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "bold"}):
        figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        x = summary["year"].to_numpy(dtype=float)
        p25 = summary["p25"].to_numpy(dtype=float)
        p50 = summary["p50"].to_numpy(dtype=float)
        p75 = summary["p75"].to_numpy(dtype=float)
        p90 = summary["p90"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            p25,
            p75,
            color=PALETTE["light_blue"],
            alpha=0.65,
            label="P25-P75",
        )
        axis.plot(
            x,
            p50,
            color=PALETTE["blue"],
            marker="o",
            linewidth=2.4,
            label="Median (P50)",
        )
        axis.plot(
            x, p90, color=PALETTE["orange"], linestyle="--", linewidth=1.9, label="P90"
        )
        partial = summary.loc[~summary["is_complete_year"]]
        if not partial.empty:
            axis.scatter(
                partial["year"],
                partial["p50"],
                s=90,
                facecolors="white",
                edgecolors=PALETTE["orange"],
                linewidths=2,
                zorder=5,
                label="Partial year",
            )
        axis.set_title("HDB resale price distribution by year")
        axis.set_xlabel("Transaction year")
        axis.set_ylabel("Resale price")
        axis.yaxis.set_major_formatter(FuncFormatter(_currency_axis))
        axis.set_xticks(summary["year"])
        axis.legend(frameon=False, ncol=2, loc="upper left")
        axis.text(
            0,
            -0.18,
            "Raw transaction percentiles; open marker denotes a year without all 12 month labels. "
            "The latest source month may be provisional; see report metadata. "
            + DESCRIPTIVE_CAVEAT,
            transform=axis.transAxes,
            fontsize=8.5,
            color=PALETTE["grey"],
            wrap=True,
        )
        _style_axes(axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def plot_million_dollar_share(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the annual share of transactions at or above S$1 million."""
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "bold"}):
        figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
        axis.plot(
            summary["year"],
            summary["million_dollar_share_pct"],
            color=PALETTE["green"],
            marker="o",
            linewidth=2.4,
        )
        axis.set_title("Share of HDB resale transactions at or above S$1 million")
        axis.set_xlabel("Transaction year")
        axis.set_ylabel("Share of transactions")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
        axis.set_xticks(summary["year"])
        axis.text(
            0,
            -0.18,
            "The newest year may be partial and its latest source month may be provisional. "
            + DESCRIPTIVE_CAVEAT,
            transform=axis.transAxes,
            fontsize=8.5,
            color=PALETTE["grey"],
            wrap=True,
        )
        _style_axes(axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def plot_remaining_lease(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot median resale price by remaining-lease band with sample sizes."""
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "bold"}):
        figure, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
        bars = axis.bar(
            summary["remaining_lease_band"],
            summary["median_resale_price"],
            color=PALETTE["blue"],
            alpha=0.9,
        )
        for bar, count in zip(bars, summary["transaction_count"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(count):,}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
                color=PALETTE["grey"],
            )
        axis.set_title("Median HDB resale price by remaining-lease band")
        axis.set_xlabel("Remaining lease (parsed from source text)")
        axis.set_ylabel("Median resale price")
        axis.yaxis.set_major_formatter(FuncFormatter(_currency_axis))
        axis.tick_params(axis="x", rotation=35)
        axis.text(
            0,
            -0.25,
            DESCRIPTIVE_CAVEAT,
            transform=axis.transAxes,
            fontsize=8.5,
            color=PALETTE["grey"],
            wrap=True,
        )
        _style_axes(axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def plot_town_growth(
    summary: pd.DataFrame, output_path: Path, *, limit: int = 12
) -> None:
    """Plot eligible endpoint growth values with endpoint sample sizes."""
    eligible = summary.loc[summary["meets_minimum_sample"]].head(limit).copy()
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "bold"}):
        figure, axis = plt.subplots(figsize=(10.5, 7), constrained_layout=True)
        if eligible.empty:
            axis.text(
                0.5,
                0.5,
                "No towns meet the endpoint sample guard",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        else:
            eligible = eligible.sort_values("median_price_growth_pct")
            bars = axis.barh(
                eligible["town"],
                eligible["median_price_growth_pct"],
                color=PALETTE["orange"],
            )
            for bar, first_count, last_count in zip(
                bars,
                eligible["first_year_count"],
                eligible["last_year_count"],
                strict=True,
            ):
                axis.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"  n={int(first_count):,}/{int(last_count):,}",
                    va="center",
                    fontsize=8,
                    color=PALETTE["navy"],
                )
            first_year = int(eligible["first_year"].iloc[0])
            last_year = int(eligible["last_year"].iloc[0])
            minimum = int(eligible["minimum_endpoint_sample"].iloc[0])
            axis.set_title(
                f"Largest changes in town median transaction price, {first_year}-{last_year}"
            )
            axis.set_xlabel("Change in raw median resale price")
            axis.set_ylabel("")
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            axis.text(
                0,
                -0.14,
                f"Only towns with at least {minimum:,} records at both endpoints are ranked; labels show first/last n. "
                + DESCRIPTIVE_CAVEAT,
                transform=axis.transAxes,
                fontsize=8.5,
                color=PALETTE["grey"],
                wrap=True,
            )
            _style_axes(axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def plot_rpi_benchmark(benchmark: pd.DataFrame, output_path: Path) -> None:
    """Plot transaction medians and official RPI rebased to the overlap start."""
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titleweight": "bold"}):
        figure, axis = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
        x = np.arange(len(benchmark))
        axis.plot(
            x,
            benchmark["raw_median_rebased"],
            color=PALETTE["orange"],
            linewidth=2.2,
            label="Raw quarterly transaction median",
        )
        axis.plot(
            x,
            benchmark["official_rpi_rebased"],
            color=PALETTE["blue"],
            linewidth=2.5,
            label="Official HDB RPI",
        )
        tick_positions = np.arange(0, len(benchmark), 4)
        if len(benchmark) - 1 not in tick_positions:
            tick_positions = np.append(tick_positions, len(benchmark) - 1)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            benchmark.loc[tick_positions, "quarter"], rotation=35, ha="right"
        )
        base_quarter = benchmark["base_quarter"].iloc[0]
        axis.set_title(
            f"Raw resale median versus official HDB RPI (rebased: {base_quarter}=100)"
        )
        axis.set_xlabel("Quarter")
        axis.set_ylabel("Rebased index")
        axis.legend(frameon=False, loc="upper left")
        axis.text(
            0,
            -0.23,
            RPI_CAVEAT + f" Source: HDB via data.gov.sg ({RPI_DATASET_ID}).",
            transform=axis.transAxes,
            fontsize=8.5,
            color=PALETTE["grey"],
            wrap=True,
        )
        _style_axes(axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def _write_markdown_report(
    path: Path,
    *,
    annual: pd.DataFrame,
    million_year: pd.DataFrame,
    town_growth: pd.DataFrame,
    duplicate_overall: dict[str, float | int],
    rpi_benchmark: pd.DataFrame | None,
    sql_reconciliation: pd.DataFrame | None,
    partial_month: dict[str, object],
) -> None:
    latest = annual.iloc[-1]
    incomplete = annual.loc[~annual["is_complete_year"], "year"].astype(str).tolist()
    eligible = town_growth.loc[town_growth["meets_minimum_sample"]]
    lines = [
        "# Advanced descriptive and robustness analysis",
        "",
        "## Scope and coverage",
        "",
        (
            f"The transaction snapshot spans {annual['first_month'].min()} to "
            f"{annual['latest_month'].max()} and contains "
            f"{int(annual['transaction_count'].sum()):,} rows."
        ),
        "",
    ]
    if incomplete:
        lines.extend(
            [
                (
                    "Years without all 12 month labels are disclosed as partial: "
                    + ", ".join(incomplete)
                    + ". They are retained in annual descriptive tables but are not "
                    "used as town-growth endpoints."
                ),
                "",
            ]
        )
    if partial_month["latest_month_is_provisional"]:
        as_of = pd.Timestamp(str(partial_month["snapshot_as_of_singapore"]))
        lines.extend(
            [
                (
                    f"The latest transaction month, {partial_month['latest_month']}, "
                    f"is provisional: the snapshot was observed as of "
                    f"{as_of.day} {as_of:%B %Y} Singapore time and contains "
                    f"{int(partial_month['provisional_rows']):,} rows for that month. "
                    "Those rows remain in year-to-date descriptive totals."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Price distribution",
            "",
            (
                f"In {int(latest['year'])}, the observed P25, median, P75 and P90 were "
                f"S${latest['p25']:,.0f}, S${latest['p50']:,.0f}, "
                f"S${latest['p75']:,.0f} and S${latest['p90']:,.0f}. "
                f"This period contains {int(latest['months_observed'])} month labels."
            ),
            "",
            "## Million-dollar transactions",
            "",
            (
                f"The observed share in {int(million_year.iloc[-1]['year'])} was "
                f"{million_year.iloc[-1]['million_dollar_share_pct']:.2f}% "
                f"({int(million_year.iloc[-1]['million_dollar_transactions']):,} of "
                f"{int(million_year.iloc[-1]['transaction_count']):,} records)."
            ),
            "",
            "## Town endpoint comparison",
            "",
        ]
    )
    if eligible.empty:
        lines.extend(["No town met the configured endpoint sample guard.", ""])
    else:
        top = eligible.iloc[0]
        lines.extend(
            [
                (
                    f"Among eligible towns, {top['town']} had the largest change in raw "
                    f"median price between {int(top['first_year'])} and "
                    f"{int(top['last_year'])}: {top['median_price_growth_pct']:.2f}%. "
                    f"Endpoint sample sizes were {int(top['first_year_count']):,} and "
                    f"{int(top['last_year_count']):,}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Duplicate sensitivity",
            "",
            (
                f"The exact-row sensitivity removes {int(duplicate_overall['rows_removed']):,} "
                f"rows. The overall median changes by "
                f"S${duplicate_overall['median_difference_sgd']:,.2f} "
                f"({duplicate_overall['median_difference_pct']:.4f}%)."
            ),
            "",
            DUPLICATE_CAVEAT,
            "",
            "## Official HDB RPI benchmark",
            "",
        ]
    )
    if rpi_benchmark is None:
        lines.extend(
            [
                (
                    "No local RPI snapshot was supplied. Run with `--download-rpi` once "
                    "or pass `--rpi-input` to generate the benchmark."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"The benchmark covers {rpi_benchmark.iloc[0]['quarter']} through "
                    f"{rpi_benchmark.iloc[-1]['quarter']}. Both series are rebased to 100 "
                    "at the first common quarter."
                ),
                "",
                f"Official source: [{RPI_DATASET_ID}]({RPI_SOURCE_URL}).",
                "",
                RPI_CAVEAT,
                "",
            ]
        )
    lines.extend(["## Python/SQL reconciliation", ""])
    if sql_reconciliation is None:
        lines.extend(
            ["SQLite reconciliation was skipped because no database was available.", ""]
        )
    else:
        matches = int(sql_reconciliation["matches_within_tolerance"].sum())
        lines.extend(
            [
                (
                    f"Python and SQLite annual medians matched within S${MATCH_TOLERANCE:.2f} "
                    f"for {matches} of {len(sql_reconciliation)} compared years."
                ),
                "",
            ]
        )
    lines.extend(["## Interpretation", "", DESCRIPTIVE_CAVEAT, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_analysis(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    database_path: Path | None = DEFAULT_DATABASE_PATH,
    rpi_input_path: Path | None = DEFAULT_RPI_INPUT_PATH,
    min_town_sample: int = DEFAULT_MIN_TOWN_SAMPLE,
    lease_bin_years: int = DEFAULT_LEASE_BIN_YEARS,
) -> dict[str, object]:
    """Generate deterministic report tables, figures and metadata."""
    source = pd.read_csv(input_path)
    prepared = prepare_transactions(source)
    input_digest = sha256_file(input_path)
    partial_month = partial_month_disclosure(
        input_path,
        input_digest,
        prepared,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    annual = annual_price_quantiles(prepared)
    million_year = million_dollar_share(prepared, "year")
    million_town = million_dollar_share(prepared, "town").sort_values(
        ["million_dollar_share_pct", "transaction_count", "town"],
        ascending=[False, False, True],
        kind="stable",
    )
    million_town_year = million_dollar_share(prepared, ["year", "town"])
    lease = remaining_lease_bands(prepared, bin_years=lease_bin_years)
    growth = town_endpoint_growth(prepared, min_sample=min_town_sample)
    duplicate_annual, duplicate_overall = duplicate_removal_sensitivity(source)

    table_paths = {
        "annual_price_quantiles": reports_dir / "annual_price_quantiles.csv",
        "million_dollar_share_by_year": reports_dir
        / "million_dollar_share_by_year.csv",
        "million_dollar_share_by_town": reports_dir
        / "million_dollar_share_by_town.csv",
        "million_dollar_share_by_town_year": reports_dir
        / "million_dollar_share_by_town_year.csv",
        "remaining_lease_bands": reports_dir / "remaining_lease_bands.csv",
        "town_endpoint_growth": reports_dir / "town_endpoint_growth.csv",
        "duplicate_sensitivity": reports_dir / "duplicate_sensitivity.csv",
    }
    for frame, path in [
        (annual, table_paths["annual_price_quantiles"]),
        (million_year, table_paths["million_dollar_share_by_year"]),
        (million_town, table_paths["million_dollar_share_by_town"]),
        (million_town_year, table_paths["million_dollar_share_by_town_year"]),
        (lease, table_paths["remaining_lease_bands"]),
        (growth, table_paths["town_endpoint_growth"]),
        (duplicate_annual, table_paths["duplicate_sensitivity"]),
    ]:
        _save_csv(frame, path)

    sql_reconciliation: pd.DataFrame | None = None
    if database_path is not None and database_path.exists():
        sql_reconciliation = reconcile_python_sql(prepared, database_path)
        table_paths["python_sql_reconciliation"] = (
            reports_dir / "python_sql_reconciliation.csv"
        )
        _save_csv(sql_reconciliation, table_paths["python_sql_reconciliation"])

    rpi_benchmark: pd.DataFrame | None = None
    rpi_metadata: dict[str, object] = {
        "dataset_id": RPI_DATASET_ID,
        "source_url": RPI_SOURCE_URL,
        "api_url": RPI_API_URL,
        "local_snapshot_available": False,
    }
    if rpi_input_path is not None and rpi_input_path.exists():
        rpi = normalise_rpi_data(pd.read_csv(rpi_input_path))
        rpi_benchmark = quarterly_rpi_benchmark(prepared, rpi)
        table_paths["quarterly_rpi_benchmark"] = (
            reports_dir / "quarterly_rpi_benchmark.csv"
        )
        _save_csv(rpi_benchmark, table_paths["quarterly_rpi_benchmark"])
        rpi_metadata.update(
            {
                "local_snapshot_available": True,
                "local_snapshot": _display_path(rpi_input_path),
                "sha256": sha256_file(rpi_input_path),
                "first_quarter": str(rpi.iloc[0]["quarter"]),
                "latest_quarter": str(rpi.iloc[-1]["quarter"]),
                "benchmark_first_quarter": str(rpi_benchmark.iloc[0]["quarter"]),
                "benchmark_latest_quarter": str(rpi_benchmark.iloc[-1]["quarter"]),
                "base_quarter": str(rpi_benchmark.iloc[0]["base_quarter"]),
            }
        )

    plot_paths = {
        "annual_price_distribution": images_dir / "annual_price_distribution.png",
        "million_dollar_share": images_dir / "million_dollar_share_by_year.png",
        "remaining_lease": images_dir / "remaining_lease_price_bands.png",
        "town_growth": images_dir / "town_growth_with_sample_guard.png",
    }
    plot_annual_quantiles(annual, plot_paths["annual_price_distribution"])
    plot_million_dollar_share(million_year, plot_paths["million_dollar_share"])
    plot_remaining_lease(lease, plot_paths["remaining_lease"])
    plot_town_growth(growth, plot_paths["town_growth"])
    if rpi_benchmark is not None:
        plot_paths["rpi_benchmark"] = images_dir / "raw_median_vs_official_rpi.png"
        plot_rpi_benchmark(rpi_benchmark, plot_paths["rpi_benchmark"])

    incomplete_years = [
        {
            "year": int(row.year),
            "months_observed": int(row.months_observed),
            "latest_month": str(row.latest_month),
        }
        for row in annual.loc[~annual["is_complete_year"]].itertuples()
    ]
    metadata: dict[str, object] = {
        "analysis_type": "descriptive_and_robustness",
        "transaction_source": {
            "path": _display_path(input_path),
            "sha256": input_digest,
            "row_count": int(len(prepared)),
            "first_month": prepared["month"].min().strftime("%Y-%m"),
            "latest_month": prepared["month"].max().strftime("%Y-%m"),
        },
        "partial_year_disclosure": {
            "incomplete_years": incomplete_years,
            "definition": "A complete year contains at least one record in all 12 month labels.",
        },
        "partial_month_disclosure": partial_month,
        "parameters": {
            "million_dollar_threshold_sgd": MILLION_DOLLAR_THRESHOLD,
            "minimum_town_endpoint_sample": int(min_town_sample),
            "remaining_lease_bin_years": int(lease_bin_years),
            "sql_match_tolerance_sgd": MATCH_TOLERANCE,
        },
        "duplicate_sensitivity_overall": duplicate_overall,
        "official_rpi": rpi_metadata,
        "sql_reconciliation": {
            "performed": sql_reconciliation is not None,
            "database": _display_path(database_path)
            if database_path is not None and database_path.exists()
            else None,
            "all_years_match": bool(
                sql_reconciliation["matches_within_tolerance"].all()
            )
            if sql_reconciliation is not None
            else None,
        },
        "outputs": {
            "tables": {
                name: _display_path(path) for name, path in sorted(table_paths.items())
            },
            "plots": {
                name: _display_path(path) for name, path in sorted(plot_paths.items())
            },
        },
        "caveats": [DESCRIPTIVE_CAVEAT, DUPLICATE_CAVEAT, RPI_CAVEAT],
    }
    metadata_path = reports_dir / "advanced_analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown_report(
        reports_dir / "advanced_analysis_report.md",
        annual=annual,
        million_year=million_year,
        town_growth=growth,
        duplicate_overall=duplicate_overall,
        rpi_benchmark=rpi_benchmark,
        sql_reconciliation=sql_reconciliation,
        partial_month=partial_month,
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse deterministic CLI inputs and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument("--rpi-input", type=Path, default=DEFAULT_RPI_INPUT_PATH)
    parser.add_argument(
        "--download-rpi",
        action="store_true",
        help="Download the official RPI only when the local snapshot is absent.",
    )
    parser.add_argument(
        "--refresh-rpi",
        action="store_true",
        help="Replace the local RPI snapshot from the official API.",
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--min-town-sample", type=int, default=DEFAULT_MIN_TOWN_SAMPLE)
    parser.add_argument("--lease-bin-years", type=int, default=DEFAULT_LEASE_BIN_YEARS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the advanced analysis command-line workflow."""
    args = parse_args(argv)
    if args.refresh_rpi:
        rows = download_rpi_dataset(args.rpi_input, force=True)
        print(f"Refreshed official RPI snapshot: {rows:,} quarters")
    elif args.download_rpi and not args.rpi_input.exists():
        rows = download_rpi_dataset(args.rpi_input)
        print(f"Downloaded official RPI snapshot: {rows:,} quarters")

    metadata = run_analysis(
        args.input,
        reports_dir=args.reports_dir,
        images_dir=args.images_dir,
        database_path=None if args.skip_sql else args.database,
        rpi_input_path=args.rpi_input,
        min_town_sample=args.min_town_sample,
        lease_bin_years=args.lease_bin_years,
    )
    print(
        "Advanced analysis complete: "
        f"{len(metadata['outputs']['tables']):,} tables and "
        f"{len(metadata['outputs']['plots']):,} plots"
    )
    print(f"Reports: {args.reports_dir}")
    print(f"Images: {args.images_dir}")


if __name__ == "__main__":
    main()
