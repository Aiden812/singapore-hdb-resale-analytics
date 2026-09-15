"""Geocode unique HDB blocks with the token-protected OneMap Search API.

The script is deliberately opt-in: it only sends requests when
``ONEMAP_API_TOKEN`` is configured, caches one row per unique block/street
combination, and resumes from that cache on later runs.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "block_coordinates.csv"
ONEMAP_SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
TOKEN_ENV_VAR = "ONEMAP_API_TOKEN"
OUTPUT_COLUMNS = (
    "address_key",
    "search_address",
    "matched_address",
    "postal_code",
    "latitude",
    "longitude",
    "match_status",
)


def normalise_address(block: object, street_name: object) -> tuple[str, str]:
    """Return a stable cache key and a OneMap search string."""
    block_text = " ".join(str(block).strip().upper().split())
    street_text = " ".join(str(street_name).strip().upper().split())
    if not block_text or not street_text:
        raise ValueError("block and street_name must both be populated")
    address_key = f"{block_text}|{street_text}"
    return address_key, f"{block_text} {street_text} SINGAPORE"


def parse_search_payload(
    payload: dict[str, object],
    *,
    address_key: str,
    search_address: str,
) -> dict[str, object]:
    """Convert a OneMap response into one deterministic cache record."""
    error = payload.get("error")
    if error:
        raise RuntimeError(f"OneMap request failed: {error}")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {
            "address_key": address_key,
            "search_address": search_address,
            "matched_address": "",
            "postal_code": "",
            "latitude": pd.NA,
            "longitude": pd.NA,
            "match_status": "not_found",
        }

    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("OneMap returned an unexpected result structure")
    try:
        latitude = float(result["LATITUDE"])
        longitude = float(result["LONGITUDE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OneMap result is missing valid coordinates") from exc

    return {
        "address_key": address_key,
        "search_address": search_address,
        "matched_address": str(result.get("ADDRESS", "")),
        "postal_code": str(result.get("POSTAL", "")),
        "latitude": latitude,
        "longitude": longitude,
        "match_status": "matched",
    }


def search_address(
    session: requests.Session,
    *,
    token: str,
    address_key: str,
    search_value: str,
) -> dict[str, object]:
    """Request the highest-ranked OneMap match for one address."""
    response = session.get(
        ONEMAP_SEARCH_URL,
        headers={"Authorization": token},
        params={
            "searchVal": search_value,
            "returnGeom": "Y",
            "getAddrDetails": "Y",
            "pageNum": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return parse_search_payload(
        response.json(),
        address_key=address_key,
        search_address=search_value,
    )


def load_cache(path: Path) -> pd.DataFrame:
    """Load a prior cache, or return an empty frame with the output schema."""
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    cache = pd.read_csv(path, dtype={"postal_code": "string"})
    missing = sorted(set(OUTPUT_COLUMNS).difference(cache.columns))
    if missing:
        raise ValueError("Coordinate cache is missing columns: " + ", ".join(missing))
    return cache.loc[:, OUTPUT_COLUMNS]


def write_cache(records: list[dict[str, object]], output_path: Path) -> None:
    """Write cache records atomically in stable address order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    cache = cache.drop_duplicates("address_key", keep="last").sort_values("address_key")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    cache.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def geocode_blocks(
    input_path: Path,
    output_path: Path,
    *,
    token: str,
    limit: int | None = None,
    delay_seconds: float = 0.25,
) -> tuple[int, int]:
    """Geocode uncached unique blocks and return new and total cache counts."""
    if not token.strip():
        raise ValueError(f"Set {TOKEN_ENV_VAR} before using the OneMap API")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    source = pd.read_csv(input_path, usecols=["block", "street_name"])
    addresses = {
        normalise_address(row.block, row.street_name)
        for row in source.itertuples(index=False)
    }
    cache = load_cache(output_path)
    cached_keys = set(cache["address_key"].astype(str))
    pending = sorted(
        (address_key, search_value)
        for address_key, search_value in addresses
        if address_key not in cached_keys
    )
    if limit is not None:
        pending = pending[:limit]

    records = cache.to_dict("records")
    with requests.Session() as session:
        for position, (address_key, search_value) in enumerate(pending, start=1):
            records.append(
                search_address(
                    session,
                    token=token,
                    address_key=address_key,
                    search_value=search_value,
                )
            )
            if position % 50 == 0:
                write_cache(records, output_path)
            if position < len(pending) and delay_seconds:
                time.sleep(delay_seconds)

    write_cache(records, output_path)
    total_records = len({str(record["address_key"]) for record in records})
    return len(pending), total_records


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum new addresses to request during this run",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.25,
        help="Pause between requests to reduce API pressure (default: 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    """Geocode the selected number of unique HDB block addresses."""
    args = parse_args()
    token = os.environ.get(TOKEN_ENV_VAR, "")
    new_count, total_count = geocode_blocks(
        args.input,
        args.output,
        token=token,
        limit=args.limit,
        delay_seconds=args.delay_seconds,
    )
    print(f"Requested {new_count:,} new addresses")
    print(f"Cache now contains {total_count:,} addresses")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
