"""Clean and validate Singapore HDB resale transaction data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "hdb_resale_raw.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.csv"
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "snapshot_metadata.json"
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
DATASET_URL = f"https://data.gov.sg/datasets/{DATASET_ID}/view"

REQUIRED_COLUMNS = {
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
}

TEXT_COLUMNS = (
    "town",
    "flat_type",
    "street_name",
    "flat_model",
    "storey_range",
)

ENGINEERED_COLUMNS = (
    "year",
    "month_number",
    "price_per_sqm",
    "flat_age",
    "storey_mid",
    "remaining_lease_months",
)

REMAINING_LEASE_PATTERN = re.compile(
    r"^\s*(?P<years>\d+)\s+years?"
    r"(?:\s+(?P<months>\d+)\s+months?)?\s*$",
    flags=re.IGNORECASE,
)


def get_storey_midpoint(storey_range: str) -> float:
    """Return the midpoint of a value such as ``04 TO 06``."""
    try:
        lower, upper = storey_range.split(" TO ")
        return (int(lower) + int(upper)) / 2
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid storey range: {storey_range!r}") from exc


def parse_remaining_lease_months(remaining_lease: str) -> int:
    """Convert values such as ``74 years 03 months`` to total months."""
    match = REMAINING_LEASE_PATTERN.fullmatch(str(remaining_lease))
    if match is None:
        raise ValueError(f"Invalid remaining lease: {remaining_lease!r}")

    years = int(match.group("years"))
    months = int(match.group("months") or 0)
    if months >= 12:
        raise ValueError(f"Invalid remaining lease: {remaining_lease!r}")

    total_months = years * 12 + months
    if not 0 < total_months <= 99 * 12:
        raise ValueError(f"Invalid remaining lease: {remaining_lease!r}")
    return total_months


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_snapshot_metadata(
    *,
    raw_path: Path,
    clean_path: Path,
    metadata_path: Path,
    clean_df: pd.DataFrame,
    parquet_path: Path | None = None,
) -> dict[str, object]:
    """Write a reproducibility manifest for the raw and clean snapshots."""
    metadata: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_file": raw_path.name,
        "source_modified_at_utc": datetime.fromtimestamp(
            raw_path.stat().st_mtime,
            UTC,
        ).isoformat(),
        "source_sha256": file_sha256(raw_path),
        "processed_file": clean_path.name,
        "processed_sha256": file_sha256(clean_path),
        "row_count": int(len(clean_df)),
        "column_count": int(len(clean_df.columns)),
        "columns": list(clean_df.columns),
        "month_min": clean_df["month"].min().strftime("%Y-%m"),
        "month_max": clean_df["month"].max().strftime("%Y-%m"),
        "fully_identical_rows_retained": int(clean_df.duplicated().sum()),
    }
    if parquet_path is not None:
        metadata.update(
            {
                "processed_parquet_file": parquet_path.name,
                "processed_parquet_sha256": file_sha256(parquet_path),
                "processed_parquet_bytes": parquet_path.stat().st_size,
            }
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return metadata


def write_clean_outputs(
    *,
    clean_df: pd.DataFrame,
    raw_path: Path,
    clean_path: Path,
    parquet_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Atomically publish cleaned CSV, Parquet, and snapshot metadata files."""
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = clean_path.with_name(f".{clean_path.name}.{uuid4().hex}.tmp")
    temporary_parquet = parquet_path.with_name(
        f".{parquet_path.name}.{uuid4().hex}.tmp"
    )

    try:
        clean_df.to_csv(
            temporary_csv,
            index=False,
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
        clean_df.to_parquet(
            temporary_parquet,
            index=False,
            compression="zstd",
        )
        temporary_csv.replace(clean_path)
        temporary_parquet.replace(parquet_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
        temporary_parquet.unlink(missing_ok=True)

    return write_snapshot_metadata(
        raw_path=raw_path,
        clean_path=clean_path,
        metadata_path=metadata_path,
        clean_df=clean_df,
        parquet_path=parquet_path,
    )


def validate_source_columns(df: pd.DataFrame) -> None:
    """Raise an informative error when required source columns are absent."""
    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing_columns)
        )


def validate_clean_data(df: pd.DataFrame) -> None:
    """Validate fields required by the downstream analysis."""
    checked_columns = [*sorted(REQUIRED_COLUMNS), *ENGINEERED_COLUMNS]
    missing_counts = df[checked_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        raise ValueError(f"Cleaning produced missing values: {details}")

    blank_text_counts = {
        column: int(df[column].eq("").sum()) for column in TEXT_COLUMNS
    }
    blank_text_counts = {
        column: count for column, count in blank_text_counts.items() if count
    }
    if blank_text_counts:
        details = ", ".join(
            f"{column}={count}" for column, count in blank_text_counts.items()
        )
        raise ValueError(f"Cleaning produced blank text values: {details}")

    if (df["floor_area_sqm"] <= 0).any():
        raise ValueError("floor_area_sqm must be greater than zero")
    if (df["resale_price"] <= 0).any():
        raise ValueError("resale_price must be greater than zero")
    if not np.isfinite(df["price_per_sqm"]).all():
        raise ValueError("price_per_sqm contains a non-finite value")
    if not df["lease_commence_date"].mod(1).eq(0).all():
        raise ValueError("lease_commence_date must contain whole years")
    if ((df["flat_age"] < 0) | (df["flat_age"] > 99)).any():
        raise ValueError("flat_age must be between 0 and 99 years")
    if (
        (df["remaining_lease_months"] <= 0) | (df["remaining_lease_months"] > 99 * 12)
    ).any():
        raise ValueError("remaining_lease_months must be between 1 and 1188")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with analysis-ready engineered features.

    Fully identical rows are retained because the source does not include a
    transaction identifier that would establish whether they are erroneous.
    """
    validate_source_columns(df)
    clean_df = df.copy()

    for column in TEXT_COLUMNS:
        clean_df[column] = clean_df[column].astype("string").str.strip()

    clean_df["month"] = pd.to_datetime(
        clean_df["month"], format="%Y-%m", errors="coerce"
    )
    clean_df["resale_price"] = pd.to_numeric(clean_df["resale_price"], errors="coerce")
    clean_df["floor_area_sqm"] = pd.to_numeric(
        clean_df["floor_area_sqm"], errors="coerce"
    )
    clean_df["lease_commence_date"] = pd.to_numeric(
        clean_df["lease_commence_date"], errors="coerce"
    )

    clean_df["year"] = clean_df["month"].dt.year
    clean_df["month_number"] = clean_df["month"].dt.month
    clean_df["price_per_sqm"] = clean_df["resale_price"] / clean_df["floor_area_sqm"]
    clean_df["flat_age"] = clean_df["year"] - clean_df["lease_commence_date"]
    clean_df["storey_mid"] = clean_df["storey_range"].apply(get_storey_midpoint)
    clean_df["remaining_lease_months"] = clean_df["remaining_lease"].apply(
        parse_remaining_lease_months
    )

    validate_clean_data(clean_df)
    clean_df["year"] = clean_df["year"].astype("int64")
    clean_df["month_number"] = clean_df["month_number"].astype("int64")
    clean_df["lease_commence_date"] = clean_df["lease_commence_date"].astype("int64")
    clean_df["flat_age"] = clean_df["flat_age"].astype("int64")
    clean_df["remaining_lease_months"] = clean_df["remaining_lease_months"].astype(
        "int64"
    )
    return clean_df


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Clean CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Snapshot metadata path (default: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help=f"Compressed Parquet path (default: {DEFAULT_PARQUET_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CSV cleaning pipeline."""
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    clean_df = clean_data(raw_df)

    metadata = write_clean_outputs(
        clean_df=clean_df,
        raw_path=args.input,
        clean_path=args.output,
        parquet_path=args.parquet,
        metadata_path=args.metadata,
    )

    print(f"Saved {len(clean_df):,} rows and {len(clean_df.columns)} columns")
    print(f"Output: {args.output}")
    print(f"Parquet: {args.parquet}")
    print(f"Metadata: {args.metadata}")
    print(
        f"Fully identical rows retained: {metadata['fully_identical_rows_retained']:,}"
    )


if __name__ == "__main__":
    main()
