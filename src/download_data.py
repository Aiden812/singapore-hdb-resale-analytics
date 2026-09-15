"""Download the current HDB resale dataset from data.gov.sg."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import requests

if __package__:
    from .http_retry import get_with_retry
else:
    from http_retry import get_with_retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "hdb_resale_raw.csv"
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
API_ROOT = "https://api-open.data.gov.sg/v1/public/api/datasets"
API_KEY_ENV_VAR = "DATA_GOV_SG_API_KEY"
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


def api_headers() -> dict[str, str]:
    """Return an optional data.gov.sg API-key header."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    return {"x-api-key": api_key} if api_key else {}


def request_download_url(
    session: requests.Session,
    *,
    poll_interval: float,
    max_attempts: int,
) -> str:
    """Initiate a dataset export and poll until its URL is available."""
    dataset_url = f"{API_ROOT}/{DATASET_ID}"
    initiate_response = get_with_retry(
        session,
        f"{dataset_url}/initiate-download",
        headers=api_headers(),
        timeout=60,
    )
    initiate_response.raise_for_status()

    for attempt in range(1, max_attempts + 1):
        response = get_with_retry(
            session,
            f"{dataset_url}/poll-download",
            headers=api_headers(),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        if data.get("url"):
            return str(data["url"])

        status = str(data.get("status", "pending")).lower()
        if status in {"error", "failed", "failure"}:
            raise RuntimeError(f"data.gov.sg download failed: {payload}")
        if attempt < max_attempts:
            time.sleep(poll_interval)

    raise TimeoutError("Timed out while waiting for the data.gov.sg download")


def validate_download(path: Path) -> None:
    """Confirm that a downloaded file is a non-empty CSV with the source schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                "Downloaded CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        if next(reader, None) is None:
            raise ValueError("Downloaded CSV contains no data rows")


def download_dataset(output_path: Path, *, force: bool = False) -> int:
    """Download the CSV atomically and return its size in bytes."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass --force to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with requests.Session() as session:
            download_url = request_download_url(
                session,
                poll_interval=15,
                max_attempts=12,
            )
            # The export URL is presigned; do not forward data.gov.sg API headers.
            with get_with_retry(
                session,
                download_url,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)

        validate_download(temporary_path)
        temporary_path.replace(output_path)
        return output_path.stat().st_size
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing raw CSV",
    )
    return parser.parse_args()


def main() -> None:
    """Download the latest available dataset snapshot."""
    args = parse_args()
    size = download_dataset(args.output, force=args.force)
    print(f"Downloaded {size / (1024 * 1024):,.1f} MB")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
