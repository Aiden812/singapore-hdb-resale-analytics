"""Enrich HDB resale transactions with official CPI and MRT datasets.

The workflow downloads and caches token-free data.gov.sg sources, deflates
transaction prices to a documented CPI reference month, and optionally joins
nearest-MRT distances when ``src/geocode_blocks.py`` has produced a block
coordinate cache. Missing CPI months and block coordinates remain missing;
the pipeline never forward-fills economic data or fabricates locations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

if __package__:
    from .http_retry import get_with_retry
else:
    from http_retry import get_with_retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
DEFAULT_CPI_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "singapore_cpi_all_items.csv"
DEFAULT_MRT_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "mrt_station_exits.geojson"
DEFAULT_BLOCK_COORDINATES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "block_coordinates.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hdb_resale_enriched.parquet"
)
DEFAULT_MONTHLY_REPORT_PATH = PROJECT_ROOT / "reports" / "monthly_price_inflation.csv"
DEFAULT_MRT_STATIONS_REPORT_PATH = PROJECT_ROOT / "reports" / "mrt_station_exits.csv"
DEFAULT_MRT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "mrt_proximity_summary.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "reports" / "official_enrichment_metadata.json"

CPI_DATASET_ID = "d_bdaff844e3ef89d39fceb962ff8f0791"
CPI_DATASET_TITLE = "Consumer Price Index (CPI), 2024 As Base Year, Monthly"
CPI_DATASET_URL = f"https://data.gov.sg/datasets/{CPI_DATASET_ID}/view"
CPI_API_URL = "https://data.gov.sg/api/action/datastore_search"

MRT_DATASET_ID = "d_b39d3a0871985372d7e1637193335da5"
MRT_DATASET_TITLE = "LTA MRT Station Exit (GEOJSON)"
MRT_DATASET_URL = f"https://data.gov.sg/datasets/{MRT_DATASET_ID}/view"
MRT_DOWNLOAD_API_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    f"{MRT_DATASET_ID}/poll-download"
)

API_KEY_ENV_VAR = "DATA_GOV_SG_API_KEY"
OPEN_DATA_LICENCE_URL = "https://data.gov.sg/open-data-licence"
EARTH_RADIUS_M = 6_371_008.8
SINGAPORE_LATITUDE_RANGE = (1.10, 1.60)
SINGAPORE_LONGITUDE_RANGE = (103.50, 104.20)

# The official exit layer currently uses bare station codes for a small number
# of MRT features. Canonicalise those records instead of silently dropping
# valid exits during the MRT/LRT split.
MRT_STATION_CODE_NAMES = {
    "CC9": "PAYA LEBAR MRT STATION",
    "CC30": "KEPPEL MRT STATION",
    "CC31": "CANTONMENT MRT STATION",
    "CC32": "PRINCE EDWARD ROAD MRT STATION",
    "DT4": "HUME MRT STATION",
    "DT18": "TELOK AYER MRT STATION",
    "NE18": "PUNGGOL COAST MRT STATION",
}

MRT_DISTANCE_BINS = (-np.inf, 500.0, 1_000.0, 2_000.0, np.inf)
MRT_DISTANCE_LABELS = (
    "0-500 m",
    ">500-1,000 m",
    ">1,000-2,000 m",
    ">2,000 m",
)
MRT_REPORT_COLUMNS = (
    "mrt_distance_band",
    "transaction_count",
    "median_resale_price",
    "median_price_per_sqm",
)


def _display_path(path: Path) -> str:
    """Return a project-relative path when the file is inside the repository."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _api_headers() -> dict[str, str]:
    """Return the optional data.gov.sg API-key header."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    return {"x-api-key": api_key} if api_key else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically with deterministic row and line formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary_path, index=False, lineterminator="\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write compressed Parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary_path, index=False, compression="zstd")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(payload: object, path: Path) -> None:
    """Write JSON atomically with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalise_column_name(column: object) -> str:
    """Return a compact identifier for flexible official-source schemas."""
    return "".join(
        character for character in str(column).casefold() if character.isalnum()
    )


def normalise_cpi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a validated monthly ``All Items`` CPI series.

    The official dataset is wide (one ``DataSeries`` row and one column per
    month). The function also accepts the pipeline's cached long format.
    """
    column_lookup = {_normalise_column_name(column): column for column in df.columns}
    if {"month", "cpiallitems"}.issubset(column_lookup):
        normalised = df[[column_lookup["month"], column_lookup["cpiallitems"]]].rename(
            columns={
                column_lookup["month"]: "month",
                column_lookup["cpiallitems"]: "cpi_all_items",
            }
        )
    else:
        data_series_column = column_lookup.get("dataseries")
        if data_series_column is None:
            raise ValueError(
                "CPI data must contain DataSeries or month/cpi_all_items columns"
            )
        series = (
            df[data_series_column]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.casefold()
        )
        matches = df.loc[series.eq("all items")]
        if len(matches) != 1:
            raise ValueError(
                "CPI data must contain exactly one All Items series; "
                f"found {len(matches)}"
            )

        source_row = matches.iloc[0]
        records: list[dict[str, object]] = []
        for column in df.columns:
            compact = str(column).strip().replace(" ", "")
            parsed_month = pd.to_datetime(compact, format="%Y%b", errors="coerce")
            if pd.isna(parsed_month):
                continue
            records.append(
                {
                    "month": parsed_month,
                    "cpi_all_items": source_row[column],
                }
            )
        if not records:
            raise ValueError("CPI data contains no YYYYMon month columns")
        normalised = pd.DataFrame.from_records(records)

    normalised = normalised.copy()
    normalised["month"] = pd.to_datetime(normalised["month"], errors="coerce")
    normalised["month"] = normalised["month"].dt.to_period("M").dt.to_timestamp()
    normalised["cpi_all_items"] = pd.to_numeric(
        normalised["cpi_all_items"], errors="coerce"
    )
    invalid = (
        normalised["month"].isna()
        | normalised["cpi_all_items"].isna()
        | normalised["cpi_all_items"].le(0)
    )
    if invalid.any():
        examples = normalised.loc[invalid].head(5).to_dict("records")
        raise ValueError(f"CPI data contains invalid month/index values: {examples}")
    if normalised["month"].duplicated().any():
        duplicates = normalised.loc[
            normalised["month"].duplicated(keep=False), "month"
        ].dt.strftime("%Y-%m")
        raise ValueError(
            "CPI data contains duplicate months: "
            + ", ".join(sorted(duplicates.unique()))
        )
    return normalised.sort_values("month", kind="stable").reset_index(drop=True)


def download_cpi_dataset(
    output_path: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> int:
    """Cache the official All Items CPI series in normalized monthly form."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass force=True to replace it."
        )

    client = session or requests.Session()
    owns_session = session is None
    try:
        response = get_with_retry(
            client,
            CPI_API_URL,
            params={
                "resource_id": CPI_DATASET_ID,
                "limit": 5,
                "filters": json.dumps({"DataSeries": "All Items"}),
            },
            headers=_api_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"data.gov.sg CPI request failed: {payload}")
        records = (payload.get("result") or {}).get("records") or []
        cpi = normalise_cpi_data(pd.DataFrame.from_records(records))
        cached = cpi.assign(month=cpi["month"].dt.strftime("%Y-%m"))
        _write_csv(cached, output_path)
        normalise_cpi_data(pd.read_csv(output_path))
        return len(cached)
    finally:
        if owns_session:
            client.close()


def normalise_mrt_geojson(
    payload: dict[str, object],
    *,
    include_lrt: bool = False,
) -> pd.DataFrame:
    """Validate official station-exit GeoJSON and return a point table.

    Although the source is named the MRT Station Exit layer, it currently also
    contains LRT features. They are explicitly excluded unless requested.
    """
    if payload.get("type") != "FeatureCollection":
        raise ValueError("MRT GeoJSON must be a FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("MRT GeoJSON contains no features")

    records: list[dict[str, object]] = []
    for position, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"MRT feature {position} is not an object")
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            raise ValueError(f"MRT feature {position} is not a Point")
        if not isinstance(properties, dict):
            raise ValueError(f"MRT feature {position} has invalid properties")
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValueError(f"MRT feature {position} has invalid coordinates")
        try:
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MRT feature {position} has non-numeric coordinates"
            ) from exc

        station_name = str(properties.get("STATION_NA", "")).strip()
        exit_code = str(properties.get("EXIT_CODE", "")).strip()
        if not station_name:
            raise ValueError(f"MRT feature {position} has no station name")
        upper_name = station_name.upper()
        if "LRT STATION" in upper_name:
            rail_mode = "LRT"
        elif "MRT STATION" in upper_name:
            rail_mode = "MRT"
        elif upper_name in MRT_STATION_CODE_NAMES:
            rail_mode = "MRT"
            station_name = MRT_STATION_CODE_NAMES[upper_name]
        else:
            rail_mode = "OTHER"
        if rail_mode != "MRT" and not (include_lrt and rail_mode == "LRT"):
            continue

        records.append(
            {
                "station_name": station_name,
                "exit_code": exit_code,
                "latitude": latitude,
                "longitude": longitude,
                "object_id": properties.get("OBJECTID"),
                "rail_mode": rail_mode,
            }
        )

    if not records:
        raise ValueError("MRT GeoJSON contains no eligible MRT station exits")
    stations = pd.DataFrame.from_records(records)
    _validate_coordinate_columns(stations, source_name="MRT station exits")
    return (
        stations.drop_duplicates(["station_name", "exit_code", "latitude", "longitude"])
        .sort_values(
            ["station_name", "exit_code", "latitude", "longitude"], kind="stable"
        )
        .reset_index(drop=True)
    )


def download_mrt_dataset(
    output_path: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> int:
    """Download and validate the official LTA station-exit GeoJSON cache."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass force=True to replace it."
        )

    client = session or requests.Session()
    owns_session = session is None
    try:
        poll_response = get_with_retry(
            client,
            MRT_DOWNLOAD_API_URL,
            headers=_api_headers(),
            timeout=60,
        )
        poll_response.raise_for_status()
        poll_payload = poll_response.json()
        if poll_payload.get("code") not in (None, 0):
            raise RuntimeError(f"data.gov.sg MRT request failed: {poll_payload}")
        download_url = (poll_payload.get("data") or {}).get("url")
        if not download_url:
            raise RuntimeError(
                "data.gov.sg MRT response did not provide a download URL"
            )

        # The export URL is presigned; do not forward data.gov.sg API headers.
        data_response = get_with_retry(
            client,
            str(download_url),
            timeout=120,
        )
        data_response.raise_for_status()
        payload = data_response.json()
        stations = normalise_mrt_geojson(payload)
        _write_json(payload, output_path)
        normalise_mrt_geojson(json.loads(output_path.read_text(encoding="utf-8")))
        return len(stations)
    finally:
        if owns_session:
            client.close()


def _validate_coordinate_columns(frame: pd.DataFrame, *, source_name: str) -> None:
    """Require finite coordinates inside a broad Singapore bounding box."""
    missing = sorted({"latitude", "longitude"}.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: " + ", ".join(missing))
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    invalid = (
        latitude.isna()
        | longitude.isna()
        | ~latitude.between(*SINGAPORE_LATITUDE_RANGE)
        | ~longitude.between(*SINGAPORE_LONGITUDE_RANGE)
    )
    if invalid.any():
        raise ValueError(
            f"{source_name} contains {int(invalid.sum())} invalid Singapore coordinates"
        )


def normalise_block_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Return validated, unique matched rows from the OneMap block cache."""
    required = {"address_key", "latitude", "longitude"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Block coordinate cache is missing columns: " + ", ".join(missing)
        )
    coordinates = df.copy()
    if "match_status" in coordinates.columns:
        coordinates = coordinates.loc[
            coordinates["match_status"].astype("string").str.casefold().eq("matched")
        ]
    coordinates["address_key"] = coordinates["address_key"].astype("string").str.strip()
    coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce")
    coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce")
    coordinates = coordinates.dropna(subset=["address_key", "latitude", "longitude"])
    coordinates = coordinates.loc[coordinates["address_key"].ne("")]
    if coordinates.empty:
        return coordinates.loc[:, ["address_key", "latitude", "longitude"]]
    _validate_coordinate_columns(coordinates, source_name="Block coordinate cache")

    conflicts = coordinates.groupby("address_key", observed=True).agg(
        latitude_values=("latitude", "nunique"),
        longitude_values=("longitude", "nunique"),
    )
    conflicts = conflicts.loc[
        conflicts["latitude_values"].gt(1) | conflicts["longitude_values"].gt(1)
    ]
    if not conflicts.empty:
        examples = ", ".join(conflicts.index.astype(str).tolist()[:5])
        raise ValueError(f"Block coordinate cache has conflicting keys: {examples}")
    return (
        coordinates[["address_key", "latitude", "longitude"]]
        .drop_duplicates("address_key")
        .sort_values("address_key", kind="stable")
        .reset_index(drop=True)
    )


def haversine_distance_m(
    latitude_1: object,
    longitude_1: object,
    latitude_2: object,
    longitude_2: object,
) -> np.ndarray:
    """Return great-circle distance in metres for broadcast-compatible inputs."""
    lat_1 = np.asarray(latitude_1, dtype=float)
    lon_1 = np.asarray(longitude_1, dtype=float)
    lat_2 = np.asarray(latitude_2, dtype=float)
    lon_2 = np.asarray(longitude_2, dtype=float)
    if not all(np.isfinite(values).all() for values in (lat_1, lon_1, lat_2, lon_2)):
        raise ValueError("Haversine coordinates must be finite numbers")

    lat_1_rad = np.radians(lat_1)
    lat_2_rad = np.radians(lat_2)
    delta_latitude = lat_2_rad - lat_1_rad
    delta_longitude = np.radians(lon_2 - lon_1)
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(delta_longitude / 2.0) ** 2
    )
    angular_distance = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return EARTH_RADIUS_M * angular_distance


def assign_mrt_distance_band(distance_m: pd.Series) -> pd.Series:
    """Assign ordered, non-overlapping walk-distance bands."""
    numeric = pd.to_numeric(distance_m, errors="coerce")
    if numeric.dropna().lt(0).any():
        raise ValueError("MRT distances cannot be negative")
    return pd.cut(
        numeric,
        bins=MRT_DISTANCE_BINS,
        labels=MRT_DISTANCE_LABELS,
        right=True,
        ordered=True,
    )


def nearest_mrt_exits(
    block_coordinates: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    chunk_size: int = 2_000,
) -> pd.DataFrame:
    """Find the nearest official MRT exit for each unique block coordinate."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    blocks = normalise_block_coordinates(block_coordinates)
    if blocks.empty:
        return pd.DataFrame(
            columns=[
                "address_key",
                "latitude",
                "longitude",
                "nearest_mrt_station",
                "nearest_mrt_exit_code",
                "nearest_mrt_distance_m",
                "mrt_distance_band",
            ]
        )
    required_station_columns = {
        "station_name",
        "exit_code",
        "latitude",
        "longitude",
    }
    missing = sorted(required_station_columns.difference(stations.columns))
    if missing:
        raise ValueError("MRT station data is missing columns: " + ", ".join(missing))
    station_points = stations.copy()
    _validate_coordinate_columns(station_points, source_name="MRT station exits")
    if station_points.empty:
        raise ValueError("MRT station data contains no exits")
    station_points = station_points.sort_values(
        ["station_name", "exit_code", "latitude", "longitude"], kind="stable"
    ).reset_index(drop=True)

    station_latitude = station_points["latitude"].to_numpy(dtype=float)
    station_longitude = station_points["longitude"].to_numpy(dtype=float)
    nearest_indices: list[np.ndarray] = []
    nearest_distances: list[np.ndarray] = []
    for start in range(0, len(blocks), chunk_size):
        chunk = blocks.iloc[start : start + chunk_size]
        distance_matrix = haversine_distance_m(
            chunk["latitude"].to_numpy(dtype=float)[:, None],
            chunk["longitude"].to_numpy(dtype=float)[:, None],
            station_latitude[None, :],
            station_longitude[None, :],
        )
        indices = np.argmin(distance_matrix, axis=1)
        nearest_indices.append(indices)
        nearest_distances.append(distance_matrix[np.arange(len(chunk)), indices])

    selected_indices = np.concatenate(nearest_indices)
    proximity = blocks.copy()
    proximity["nearest_mrt_station"] = station_points.iloc[selected_indices][
        "station_name"
    ].to_numpy()
    proximity["nearest_mrt_exit_code"] = station_points.iloc[selected_indices][
        "exit_code"
    ].to_numpy()
    proximity["nearest_mrt_distance_m"] = np.round(np.concatenate(nearest_distances), 1)
    proximity["mrt_distance_band"] = assign_mrt_distance_band(
        proximity["nearest_mrt_distance_m"]
    )
    return proximity


def _address_keys(transactions: pd.DataFrame) -> pd.Series:
    """Build keys identical to ``geocode_blocks.normalise_address``."""
    missing = sorted({"block", "street_name"}.difference(transactions.columns))
    if missing:
        raise ValueError("Transaction data is missing columns: " + ", ".join(missing))
    block = (
        transactions["block"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    street = (
        transactions["street_name"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    invalid = block.isna() | street.isna() | block.eq("") | street.eq("")
    if invalid.any():
        raise ValueError(
            f"Transaction data contains {int(invalid.sum())} blank block/street values"
        )
    return block + "|" + street


def add_cpi_adjustment(
    transactions: pd.DataFrame,
    cpi_data: pd.DataFrame,
    *,
    reference_month: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    """Join monthly CPI and add prices in reference-month Singapore dollars."""
    missing = sorted({"month", "resale_price"}.difference(transactions.columns))
    if missing:
        raise ValueError("Transaction data is missing columns: " + ", ".join(missing))
    enriched = transactions.drop(
        columns=[
            "cpi_all_items",
            "cpi_reference_month",
            "resale_price_real_sgd",
            "price_per_sqm_real_sgd",
        ],
        errors="ignore",
    ).copy()
    transaction_month = pd.to_datetime(enriched["month"], errors="coerce")
    resale_price = pd.to_numeric(enriched["resale_price"], errors="coerce")
    if transaction_month.isna().any():
        raise ValueError("Transaction data contains invalid month values")
    if resale_price.isna().any() or resale_price.le(0).any():
        raise ValueError("Transaction data contains invalid resale_price values")
    enriched["resale_price"] = resale_price
    enriched["_month_key"] = transaction_month.dt.to_period("M").dt.to_timestamp()

    cpi = normalise_cpi_data(cpi_data)
    if reference_month is None:
        reference_timestamp = cpi["month"].max()
    else:
        try:
            reference_timestamp = pd.Period(reference_month, freq="M").to_timestamp()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid CPI reference month: {reference_month!r}"
            ) from exc
    reference_rows = cpi.loc[cpi["month"].eq(reference_timestamp), "cpi_all_items"]
    if reference_rows.empty:
        raise ValueError(
            "CPI reference month is not available: "
            + reference_timestamp.strftime("%Y-%m")
        )
    reference_cpi = float(reference_rows.iloc[0])

    enriched = enriched.merge(
        cpi.rename(columns={"month": "_month_key"}),
        on="_month_key",
        how="left",
        validate="many_to_one",
    )
    reference_label = reference_timestamp.strftime("%Y-%m")
    enriched["cpi_reference_month"] = reference_label
    enriched["resale_price_real_sgd"] = (
        enriched["resale_price"] * reference_cpi / enriched["cpi_all_items"]
    )
    if "price_per_sqm" in enriched.columns:
        price_per_sqm = pd.to_numeric(enriched["price_per_sqm"], errors="coerce")
    elif "floor_area_sqm" in enriched.columns:
        floor_area = pd.to_numeric(enriched["floor_area_sqm"], errors="coerce")
        if floor_area.isna().any() or floor_area.le(0).any():
            raise ValueError("Transaction data contains invalid floor_area_sqm values")
        price_per_sqm = enriched["resale_price"] / floor_area
    else:
        raise ValueError(
            "Transaction data needs price_per_sqm or floor_area_sqm for CPI enrichment"
        )
    enriched["price_per_sqm_real_sgd"] = (
        price_per_sqm * reference_cpi / enriched["cpi_all_items"]
    )
    return enriched.drop(columns="_month_key"), reference_label


def add_mrt_proximity(
    transactions: pd.DataFrame,
    proximity: pd.DataFrame,
) -> pd.DataFrame:
    """Join one nearest-exit record to each transaction address."""
    drop_columns = [
        "address_key",
        "latitude",
        "longitude",
        "nearest_mrt_station",
        "nearest_mrt_exit_code",
        "nearest_mrt_distance_m",
        "mrt_distance_band",
    ]
    enriched = transactions.drop(columns=drop_columns, errors="ignore").copy()
    enriched["address_key"] = _address_keys(enriched)
    return enriched.merge(
        proximity,
        on="address_key",
        how="left",
        validate="many_to_one",
    )


def monthly_price_inflation_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """Summarize nominal and CPI-adjusted monthly transaction medians."""
    required = {
        "month",
        "resale_price",
        "cpi_all_items",
        "cpi_reference_month",
        "resale_price_real_sgd",
    }
    missing = sorted(required.difference(enriched.columns))
    if missing:
        raise ValueError("Enriched data is missing columns: " + ", ".join(missing))
    prepared = enriched.copy()
    prepared["month"] = pd.to_datetime(prepared["month"], errors="coerce")
    if prepared["month"].isna().any():
        raise ValueError("Enriched data contains invalid month values")
    prepared["_month_key"] = prepared["month"].dt.to_period("M").dt.to_timestamp()
    summary = (
        prepared.groupby("_month_key", sort=True, observed=True)
        .agg(
            transaction_count=("resale_price", "size"),
            median_resale_price=("resale_price", "median"),
            cpi_all_items=("cpi_all_items", "first"),
            cpi_reference_month=("cpi_reference_month", "first"),
            median_resale_price_real_sgd=("resale_price_real_sgd", "median"),
        )
        .reset_index()
        .rename(columns={"_month_key": "month"})
    )
    nominal_base = summary["median_resale_price"].dropna().iloc[0]
    real_base = summary["median_resale_price_real_sgd"].dropna().iloc[0]
    summary["nominal_price_index"] = (
        100.0 * summary["median_resale_price"] / nominal_base
    )
    summary["real_price_index"] = (
        100.0 * summary["median_resale_price_real_sgd"] / real_base
    )
    summary["month"] = summary["month"].dt.strftime("%Y-%m")
    return summary


def mrt_proximity_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed resale outcomes by nearest-MRT distance band."""
    required = {
        "mrt_distance_band",
        "nearest_mrt_distance_m",
        "resale_price",
        "price_per_sqm",
    }
    if not required.issubset(enriched.columns):
        return pd.DataFrame(columns=MRT_REPORT_COLUMNS)
    matched = enriched.loc[enriched["nearest_mrt_distance_m"].notna()].copy()
    if matched.empty:
        return pd.DataFrame(columns=MRT_REPORT_COLUMNS)
    matched["mrt_distance_band"] = pd.Categorical(
        matched["mrt_distance_band"],
        categories=MRT_DISTANCE_LABELS,
        ordered=True,
    )
    summary = (
        matched.groupby("mrt_distance_band", sort=True, observed=True)
        .agg(
            transaction_count=("resale_price", "size"),
            median_resale_price=("resale_price", "median"),
            median_price_per_sqm=("price_per_sqm", "median"),
        )
        .reset_index()
    )
    summary["mrt_distance_band"] = summary["mrt_distance_band"].astype("string")
    return summary.loc[:, MRT_REPORT_COLUMNS]


def _load_transactions(path: Path) -> pd.DataFrame:
    """Load the clean snapshot from CSV or Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Clean transaction snapshot not found: {path}")
    if path.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _file_metadata(path: Path) -> dict[str, object]:
    """Return stable provenance fields for a local source cache."""
    return {
        "file": _display_path(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "modified_at_utc": datetime.fromtimestamp(
            path.stat().st_mtime, UTC
        ).isoformat(),
    }


def run_enrichment(
    input_path: Path,
    *,
    cpi_cache_path: Path = DEFAULT_CPI_CACHE_PATH,
    mrt_cache_path: Path = DEFAULT_MRT_CACHE_PATH,
    block_coordinates_path: Path = DEFAULT_BLOCK_COORDINATES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    monthly_report_path: Path = DEFAULT_MONTHLY_REPORT_PATH,
    mrt_stations_report_path: Path = DEFAULT_MRT_STATIONS_REPORT_PATH,
    mrt_summary_path: Path = DEFAULT_MRT_SUMMARY_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    download_missing: bool = True,
    refresh_cpi: bool = False,
    refresh_mrt: bool = False,
    cpi_reference_month: str | None = None,
) -> dict[str, object]:
    """Run official CPI enrichment and optional block-to-MRT proximity."""
    if refresh_cpi:
        download_cpi_dataset(cpi_cache_path, force=True)
    elif not cpi_cache_path.exists():
        if not download_missing:
            raise FileNotFoundError(
                f"CPI cache not found in offline mode: {cpi_cache_path}"
            )
        download_cpi_dataset(cpi_cache_path)

    if refresh_mrt:
        download_mrt_dataset(mrt_cache_path, force=True)
    elif not mrt_cache_path.exists() and download_missing:
        download_mrt_dataset(mrt_cache_path)

    transactions = _load_transactions(input_path)
    cpi = normalise_cpi_data(pd.read_csv(cpi_cache_path))
    enriched, reference_label = add_cpi_adjustment(
        transactions,
        cpi,
        reference_month=cpi_reference_month,
    )
    monthly_summary = monthly_price_inflation_summary(enriched)
    _write_csv(monthly_summary, monthly_report_path)

    limitations: list[str] = []
    station_exits = pd.DataFrame(
        columns=[
            "station_name",
            "exit_code",
            "latitude",
            "longitude",
            "object_id",
            "rail_mode",
        ]
    )
    mrt_source: dict[str, object] = {
        "dataset_id": MRT_DATASET_ID,
        "dataset_title": MRT_DATASET_TITLE,
        "dataset_url": MRT_DATASET_URL,
        "download_api_url": MRT_DOWNLOAD_API_URL,
        "licence_url": OPEN_DATA_LICENCE_URL,
        "cache_available": mrt_cache_path.exists(),
    }
    if mrt_cache_path.exists():
        station_exits = normalise_mrt_geojson(
            json.loads(mrt_cache_path.read_text(encoding="utf-8"))
        )
        mrt_source.update(_file_metadata(mrt_cache_path))
        mrt_source["mrt_exit_count"] = int(len(station_exits))
    else:
        limitations.append(
            "Official MRT station-exit cache is unavailable. Re-run online to download it."
        )
    _write_csv(station_exits, mrt_stations_report_path)

    mrt_status = "unavailable_missing_block_coordinates"
    proximity = pd.DataFrame()
    if not block_coordinates_path.exists():
        limitations.append(
            "Nearest-MRT distances are unavailable because block_coordinates.csv "
            "has not been generated. Set ONEMAP_API_TOKEN and run src/geocode_blocks.py; "
            "no block coordinates were fabricated."
        )
    else:
        block_coordinates = normalise_block_coordinates(
            pd.read_csv(block_coordinates_path)
        )
        if block_coordinates.empty:
            mrt_status = "unavailable_no_matched_block_coordinates"
            limitations.append(
                "Nearest-MRT distances are unavailable because the OneMap cache has "
                "no matched block coordinates."
            )
        elif station_exits.empty:
            mrt_status = "unavailable_missing_mrt_cache"
        else:
            proximity = nearest_mrt_exits(block_coordinates, station_exits)
            enriched = add_mrt_proximity(enriched, proximity)
            matched_rows = int(enriched["nearest_mrt_distance_m"].notna().sum())
            mrt_status = "complete" if matched_rows == len(enriched) else "partial"
            if mrt_status == "partial":
                limitations.append(
                    "MRT coverage is partial because the OneMap block cache does not "
                    "contain every transaction address."
                )

    mrt_summary = mrt_proximity_summary(enriched)
    _write_csv(mrt_summary, mrt_summary_path)
    _write_parquet(enriched, output_path)

    transaction_months = pd.to_datetime(enriched["month"]).dt.to_period("M")
    unmatched_cpi = sorted(
        transaction_months.loc[enriched["cpi_all_items"].isna()].astype(str).unique()
    )
    cpi_matched_rows = int(enriched["cpi_all_items"].notna().sum())
    if unmatched_cpi:
        limitations.append(
            "CPI-adjusted prices are unavailable for transaction months not yet "
            "published in the official CPI series; values were not forward-filled."
        )

    if "nearest_mrt_distance_m" in enriched.columns:
        mrt_matched_rows = int(enriched["nearest_mrt_distance_m"].notna().sum())
        matched_unique_addresses = int(
            enriched.loc[
                enriched["nearest_mrt_distance_m"].notna(), "address_key"
            ].nunique()
        )
        eligible_unique_addresses = int(enriched["address_key"].nunique())
    else:
        mrt_matched_rows = 0
        matched_unique_addresses = 0
        eligible_unique_addresses = int(_address_keys(enriched).nunique())

    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            **_file_metadata(input_path),
            "row_count": int(len(transactions)),
            "month_min": str(transaction_months.min()),
            "month_max": str(transaction_months.max()),
        },
        "output": {
            **_file_metadata(output_path),
            "row_count": int(len(enriched)),
            "column_count": int(len(enriched.columns)),
        },
        "cpi": {
            "status": "complete" if cpi_matched_rows == len(enriched) else "partial",
            "reference_month": reference_label,
            "matched_rows": cpi_matched_rows,
            "coverage_pct": round(100.0 * cpi_matched_rows / len(enriched), 4),
            "unmatched_months": unmatched_cpi,
            "definition": (
                "resale_price_real_sgd = resale_price * reference-month CPI / "
                "transaction-month CPI"
            ),
            "source": {
                "dataset_id": CPI_DATASET_ID,
                "dataset_title": CPI_DATASET_TITLE,
                "dataset_url": CPI_DATASET_URL,
                "api_url": CPI_API_URL,
                "licence_url": OPEN_DATA_LICENCE_URL,
                **_file_metadata(cpi_cache_path),
                "month_min": cpi["month"].min().strftime("%Y-%m"),
                "month_max": cpi["month"].max().strftime("%Y-%m"),
                "month_count": int(len(cpi)),
            },
        },
        "mrt": {
            "status": mrt_status,
            "matched_rows": mrt_matched_rows,
            "coverage_pct": round(100.0 * mrt_matched_rows / len(enriched), 4),
            "matched_unique_addresses": matched_unique_addresses,
            "eligible_unique_addresses": eligible_unique_addresses,
            "distance_definition": (
                "Straight-line great-circle distance from the OneMap block point to "
                "the nearest official MRT station exit (LRT exits excluded)."
            ),
            "block_coordinates": _file_metadata(block_coordinates_path)
            if block_coordinates_path.exists()
            else None,
            "source": mrt_source,
        },
        "outputs": {
            "monthly_price_inflation": _display_path(monthly_report_path),
            "mrt_station_exits": _display_path(mrt_stations_report_path),
            "mrt_proximity_summary": _display_path(mrt_summary_path),
        },
        "limitations": limitations,
    }
    _write_json(metadata, metadata_path)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line inputs and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--cpi-cache", type=Path, default=DEFAULT_CPI_CACHE_PATH)
    parser.add_argument("--mrt-cache", type=Path, default=DEFAULT_MRT_CACHE_PATH)
    parser.add_argument(
        "--block-coordinates", type=Path, default=DEFAULT_BLOCK_COORDINATES_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--monthly-report", type=Path, default=DEFAULT_MONTHLY_REPORT_PATH
    )
    parser.add_argument(
        "--mrt-stations-report",
        type=Path,
        default=DEFAULT_MRT_STATIONS_REPORT_PATH,
    )
    parser.add_argument("--mrt-summary", type=Path, default=DEFAULT_MRT_SUMMARY_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--cpi-reference-month",
        help="Reference month for real SGD (default: latest official CPI month)",
    )
    parser.add_argument("--refresh-cpi", action="store_true")
    parser.add_argument("--refresh-mrt", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use existing source caches and make no data.gov.sg requests",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the official-source enrichment workflow."""
    args = parse_args(argv)
    if args.offline and (args.refresh_cpi or args.refresh_mrt):
        raise ValueError("--offline cannot be combined with a refresh option")
    metadata = run_enrichment(
        args.input,
        cpi_cache_path=args.cpi_cache,
        mrt_cache_path=args.mrt_cache,
        block_coordinates_path=args.block_coordinates,
        output_path=args.output,
        monthly_report_path=args.monthly_report,
        mrt_stations_report_path=args.mrt_stations_report,
        mrt_summary_path=args.mrt_summary,
        metadata_path=args.metadata,
        download_missing=not args.offline,
        refresh_cpi=args.refresh_cpi,
        refresh_mrt=args.refresh_mrt,
        cpi_reference_month=args.cpi_reference_month,
    )
    print(
        "Official enrichment complete: "
        f"{metadata['output']['row_count']:,} transaction rows"
    )
    print(
        f"CPI coverage: {metadata['cpi']['coverage_pct']:.2f}% "
        f"({metadata['cpi']['reference_month']} SGD)"
    )
    print(
        f"MRT coverage: {metadata['mrt']['coverage_pct']:.2f}% "
        f"({metadata['mrt']['status']})"
    )
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
