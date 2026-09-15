"""Apply the portfolio-quality notebook refinements in an idempotent way."""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hdb_analysis.ipynb"


def cell_source(cell: nbformat.NotebookNode) -> str:
    """Return one cell's source as text."""
    return str(cell.get("source", ""))


def find_cell(notebook: nbformat.NotebookNode, prefix: str) -> int:
    """Find a cell by prefix, with a unique-contains fallback."""
    for index, cell in enumerate(notebook.cells):
        if cell_source(cell).strip().startswith(prefix):
            return index

    containing = [
        index
        for index, cell in enumerate(notebook.cells)
        if prefix in cell_source(cell)
    ]
    if len(containing) == 1:
        return containing[0]
    raise ValueError(f"Notebook cell not found: {prefix!r}")


def replace_cell(
    notebook: nbformat.NotebookNode,
    prefix: str,
    source: str,
) -> None:
    """Replace a cell and clear stale code output when applicable."""
    target_source = source.strip()
    cell = next(
        (
            candidate
            for candidate in notebook.cells
            if cell_source(candidate).strip() == target_source
        ),
        None,
    )
    if cell is None:
        cell = notebook.cells[find_cell(notebook, prefix)]
    cell.source = target_source
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def upsert_before(
    notebook: nbformat.NotebookNode,
    before_prefix: str,
    marker: str,
    new_cells: list[nbformat.NotebookNode],
) -> None:
    """Insert tagged cells once, replacing a prior tagged block if present."""
    existing = [
        index
        for index, cell in enumerate(notebook.cells)
        if marker in cell.get("metadata", {}).get("tags", [])
    ]
    for index in reversed(existing):
        del notebook.cells[index]
    insert_at = find_cell(notebook, before_prefix)
    for cell in new_cells:
        cell.metadata["tags"] = [marker]
    notebook.cells[insert_at:insert_at] = new_cells


def upgrade_notebook(path: Path = NOTEBOOK_PATH) -> None:
    """Upgrade notebook integration, wording, and exploratory plots."""
    notebook = nbformat.read(path, as_version=4)

    replace_cell(
        notebook,
        "# Singapore HDB Resale Market Analysis",
        """
# Singapore HDB Resale Market Analysis

## Objective

This notebook explores Singapore HDB resale transactions from 2017 onwards. It describes price distributions, transaction volumes, flat characteristics and geographical differences before the separate quality-adjusted model controls for transaction mix.

**Stakeholder:** a buyer or market analyst comparing observed market segments while avoiding causal interpretations of descriptive data.

Data source: Housing & Development Board via data.gov.sg.
        """,
    )
    replace_cell(
        notebook,
        "import pandas as pd",
        """
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean_data import (
    clean_data,
    parse_remaining_lease_months,
    write_clean_outputs,
)

PRIMARY = "#0B3C5D"
ACCENT = "#E07A5F"
NEUTRAL = "#8D99AE"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})
sgd_axis = FuncFormatter(lambda value, _: f"S${value / 1_000:,.0f}k")
        """,
    )
    replace_cell(
        notebook,
        'df = pd.read_csv("../data/raw/hdb_resale_raw.csv")',
        """
raw_path = PROJECT_ROOT / "data" / "raw" / "hdb_resale_raw.csv"
df = pd.read_csv(raw_path)
raw_df = df.copy()
        """,
    )

    upsert_before(
        notebook,
        "### Clean Text Columns",
        "remaining-lease-feature",
        [
            nbformat.v4.new_markdown_cell(
                """### Convert Remaining Lease

The source text is converted to total months so lease comparisons use a validated numeric measure."""
            ),
            nbformat.v4.new_code_cell(
                """df["remaining_lease_months"] = df["remaining_lease"].apply(
    parse_remaining_lease_months
)"""
            ),
        ],
    )

    upsert_before(
        notebook,
        "### Validate the Transformations",
        "shared-cleaning-reconciliation",
        [
            nbformat.v4.new_markdown_cell(
                """### Reconcile with the Shared Cleaning Pipeline

The step-by-step transformations above are retained for explanation. This check makes `src/clean_data.py` the authoritative implementation and fails if the notebook produces different values."""
            ),
            nbformat.v4.new_code_cell(
                """pipeline_df = clean_data(raw_df)
pd.testing.assert_frame_equal(
    df.loc[:, pipeline_df.columns].reset_index(drop=True),
    pipeline_df.reset_index(drop=True),
    check_dtype=False,
)
df = pipeline_df"""
            ),
        ],
    )

    replace_cell(
        notebook,
        "df.to_csv(",
        """
clean_csv_path = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.csv"
clean_parquet_path = PROJECT_ROOT / "data" / "processed" / "hdb_resale_clean.parquet"
metadata_path = PROJECT_ROOT / "data" / "processed" / "snapshot_metadata.json"

snapshot_metadata = write_clean_outputs(
    clean_df=df,
    raw_path=raw_path,
    clean_path=clean_csv_path,
    parquet_path=clean_parquet_path,
    metadata_path=metadata_path,
)
snapshot_metadata
        """,
    )
    replace_cell(
        notebook,
        "## Analytical Questions",
        """
## Analytical Questions

1. How have observed HDB resale prices changed over time?
2. Which towns have the highest observed median resale prices?
3. Which towns record the most transactions?
4. How do median resale prices differ by flat type?
5. How is floor area associated with resale price?
6. How is approximate flat age associated with resale price?
7. How do observed prices differ across storey ranges?
8. Which towns show the greatest change in median transaction price?

These are descriptive questions. The quality-adjusted model evaluates the same market while holding recorded property characteristics constant.
        """,
    )
    replace_cell(
        notebook,
        "transactions_by_year.plot(",
        """
year_colours = [
    ACCENT if year == df["year"].max() and df.loc[df["year"].eq(year), "month_number"].nunique() < 12 else PRIMARY
    for year in transactions_by_year.index
]
axis = transactions_by_year.plot(
    kind="bar",
    figsize=(10, 5),
    color=year_colours,
)
axis.set_title("HDB resale transaction records by year")
axis.set_xlabel("Year")
axis.set_ylabel("Transaction records")
axis.text(
    1,
    -0.20,
    "The highlighted latest year is incomplete.",
    transform=axis.transAxes,
    ha="right",
    color=NEUTRAL,
)
plt.tight_layout()
plt.savefig("../images/transactions_by_year.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "yearly_price = (",
        """
yearly_price = df.groupby("year")["resale_price"].median()

axis = yearly_price.plot(
    figsize=(10, 5),
    marker="o",
    linewidth=2.4,
    color=PRIMARY,
)
axis.scatter(yearly_price.index[-1], yearly_price.iloc[-1], color=ACCENT, zorder=3)
axis.set_title("Observed median HDB resale price by year")
axis.set_xlabel("Year")
axis.set_ylabel("Median resale price")
axis.yaxis.set_major_formatter(sgd_axis)
axis.text(
    1,
    -0.20,
    "Latest year is year-to-date; medians are not mix-adjusted.",
    transform=axis.transAxes,
    ha="right",
    color=NEUTRAL,
)
plt.tight_layout()
plt.savefig("../images/median_price_by_year.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "town_prices.plot(",
        """
axis = town_prices.sort_values().plot(
    kind="barh",
    figsize=(10, 8),
    color=PRIMARY,
)
axis.set_title("Observed median resale price by town")
axis.set_xlabel("Median resale price")
axis.set_ylabel("")
axis.xaxis.set_major_formatter(sgd_axis)
plt.tight_layout()
plt.savefig("../images/median_price_by_town.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "town_transactions.head(10).plot(",
        """
axis = town_transactions.head(10).sort_values().plot(
    kind="barh",
    figsize=(9, 5.5),
    color=PRIMARY,
)
axis.set_title("Towns with the most resale transaction records")
axis.set_xlabel("Transaction records")
axis.set_ylabel("")
plt.tight_layout()
plt.savefig("../images/top_towns_by_transactions.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "flat_type_prices = (",
        """
flat_type_summary = (
    df.groupby("flat_type")["resale_price"]
      .agg(median_price="median", transactions="size")
      .sort_values("median_price")
)
flat_type_prices = flat_type_summary["median_price"]

axis = flat_type_prices.plot(
    kind="barh",
    figsize=(8, 5),
    color=PRIMARY,
)
axis.set_title("Observed median resale price by flat type")
axis.set_xlabel("Median resale price")
axis.set_ylabel("")
axis.xaxis.set_major_formatter(sgd_axis)
for position, (_, row) in enumerate(flat_type_summary.iterrows()):
    axis.text(row["median_price"], position, f"  n={int(row['transactions']):,}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("../images/median_price_by_flat_type.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "sample_df = df.sample(",
        """
plot_df = df[[
    "floor_area_sqm",
    "flat_age",
    "resale_price",
]].dropna()
        """,
    )
    replace_cell(
        notebook,
        'plt.figure(figsize=(9, 5))\n\nplt.scatter(\n    sample_df["floor_area_sqm"]',
        """
figure, axis = plt.subplots(figsize=(9, 5))
hexbin = axis.hexbin(
    plot_df["floor_area_sqm"],
    plot_df["resale_price"],
    gridsize=45,
    mincnt=1,
    bins="log",
    cmap="Blues",
)
figure.colorbar(hexbin, ax=axis, label="log10 transaction count")
axis.set_title("Floor area and resale price density")
axis.set_xlabel("Floor area (sqm)")
axis.set_ylabel("Resale price")
axis.yaxis.set_major_formatter(sgd_axis)
plt.tight_layout()
plt.savefig("../images/floor_area_vs_price.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        'plt.figure(figsize=(9, 5))\n\nplt.scatter(\n    sample_df["flat_age"]',
        """
figure, axis = plt.subplots(figsize=(9, 5))
hexbin = axis.hexbin(
    plot_df["flat_age"],
    plot_df["resale_price"],
    gridsize=42,
    mincnt=1,
    bins="log",
    cmap="Blues",
)
figure.colorbar(hexbin, ax=axis, label="log10 transaction count")
axis.set_title("Approximate flat age and resale price density")
axis.set_xlabel("Approximate flat age (years)")
axis.set_ylabel("Resale price")
axis.yaxis.set_major_formatter(sgd_axis)
plt.tight_layout()
plt.savefig("../images/flat_age_vs_price.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "storey_prices = (",
        """
storey_prices = (
    df.groupby(["storey_mid", "storey_range"], as_index=False)
      .agg(median_price=("resale_price", "median"), transactions=("resale_price", "size"))
      .sort_values("storey_mid")
)

storey_prices[["storey_range", "median_price", "transactions"]]
        """,
    )
    replace_cell(
        notebook,
        "plt.figure(figsize=(10, 5))\nplt.plot(\n    storey_prices",
        """
figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(
    storey_prices["storey_mid"],
    storey_prices["median_price"],
    marker="o",
    linewidth=2.2,
    color=PRIMARY,
)
axis.set_xticks(storey_prices["storey_mid"])
axis.set_xticklabels(storey_prices["storey_range"], rotation=45, ha="right")
axis.set_title("Observed median resale price by storey range")
axis.set_xlabel("Storey range")
axis.set_ylabel("Median resale price")
axis.yaxis.set_major_formatter(sgd_axis)
axis.text(
    1,
    -0.28,
    f"Smallest band: n={storey_prices['transactions'].min():,}; comparisons are not adjusted.",
    transform=axis.transAxes,
    ha="right",
    color=NEUTRAL,
)
plt.tight_layout()
plt.savefig("../images/median_price_by_storey.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )
    replace_cell(
        notebook,
        "months_by_year = df.groupby",
        """
months_by_year = df.groupby("year")["month_number"].nunique()
current_year = pd.Timestamp.today().year
complete_years = (
    months_by_year[months_by_year.eq(12) & (months_by_year.index < current_year)]
      .index
      .sort_values()
)
if len(complete_years) < 2:
    raise ValueError("At least two complete calendar years are required.")

first_complete_year = int(complete_years[0])
last_complete_year = int(complete_years[-1])
minimum_endpoint_sample = 100

town_year_summary = (
    df[df["year"].isin([first_complete_year, last_complete_year])]
      .groupby(["town", "year"])["resale_price"]
      .agg(median_price="median", transactions="size")
      .reset_index()
)
price_endpoints = town_year_summary.pivot(
    index="town", columns="year", values="median_price"
)
count_endpoints = town_year_summary.pivot(
    index="town", columns="year", values="transactions"
)
town_growth = price_endpoints[[first_complete_year, last_complete_year]].dropna().copy()
town_growth["first_year_count"] = count_endpoints[first_complete_year]
town_growth["last_year_count"] = count_endpoints[last_complete_year]
town_growth = town_growth[
    town_growth["first_year_count"].ge(minimum_endpoint_sample)
    & town_growth["last_year_count"].ge(minimum_endpoint_sample)
].copy()
town_growth["growth_pct"] = (
    (town_growth[last_complete_year] - town_growth[first_complete_year])
    / town_growth[first_complete_year]
    * 100
)
town_growth = town_growth.sort_values("growth_pct", ascending=False)
town_growth.head(10)
        """,
    )
    replace_cell(
        notebook,
        "top_town_growth = town_growth.head(10)",
        """
top_town_growth = town_growth.head(10).sort_values("growth_pct")
axis = top_town_growth["growth_pct"].plot(
    kind="barh",
    figsize=(10, 6),
    color=ACCENT,
)
axis.set_title(
    f"Change in observed town median price ({first_complete_year}-{last_complete_year})"
)
axis.set_xlabel("Change in median transaction price (%)")
axis.set_ylabel("")
for position, (_, row) in enumerate(top_town_growth.iterrows()):
    axis.text(
        row["growth_pct"],
        position,
        f"  n={int(row['first_year_count']):,}/{int(row['last_year_count']):,}",
        va="center",
        fontsize=8,
    )
plt.tight_layout()
plt.savefig("../images/town_price_growth.png", dpi=170, bbox_inches="tight")
plt.show()
        """,
    )

    nbformat.write(notebook, path)


if __name__ == "__main__":
    upgrade_notebook()
