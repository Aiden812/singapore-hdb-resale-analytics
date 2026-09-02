"""Clean raw Singapore HDB resale transaction data."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Run the data-cleaning pipeline."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
