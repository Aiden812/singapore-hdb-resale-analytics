"""Integrity checks for portfolio artifacts committed with the project."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "snapshot_metadata.json"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hdb_analysis.ipynb"
PORTFOLIO_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "portfolio_analysis.ipynb"


def sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SnapshotArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cls.data = pd.read_parquet(
            PARQUET_PATH,
            columns=["month", "resale_price", "remaining_lease_months"],
        )

    def test_parquet_matches_the_metadata_manifest(self) -> None:
        self.assertEqual(
            sha256(PARQUET_PATH),
            self.metadata["processed_parquet_sha256"],
        )
        self.assertEqual(len(self.data), self.metadata["row_count"])
        self.assertEqual(
            PARQUET_PATH.stat().st_size, self.metadata["processed_parquet_bytes"]
        )

    def test_snapshot_coverage_and_numeric_quality(self) -> None:
        months = pd.to_datetime(self.data["month"])
        self.assertEqual(months.min().strftime("%Y-%m"), self.metadata["month_min"])
        self.assertEqual(months.max().strftime("%Y-%m"), self.metadata["month_max"])
        self.assertTrue(self.data["resale_price"].gt(0).all())
        self.assertTrue(self.data["remaining_lease_months"].between(1, 1188).all())


class NotebookArtifactTests(unittest.TestCase):
    def test_saved_notebooks_have_no_failed_or_unexecuted_code_cells(self) -> None:
        for path in (NOTEBOOK_PATH, PORTFOLIO_NOTEBOOK_PATH):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell.get("cell_type") == "code"
                ]
                errors = [
                    output
                    for cell in code_cells
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]

                self.assertTrue(code_cells)
                self.assertFalse(errors)
                self.assertFalse(
                    [cell for cell in code_cells if cell.get("execution_count") is None]
                )


class DocumentationArtifactTests(unittest.TestCase):
    def test_readme_relative_links_resolve(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        link_targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
        local_targets = [
            target.split("#", maxsplit=1)[0]
            for target in link_targets
            if not target.startswith(("http://", "https://", "#"))
        ]

        missing = [
            target
            for target in local_targets
            if target and not (PROJECT_ROOT / target).exists()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
