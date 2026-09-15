"""Build the executed portfolio notebook from reproducible report artifacts."""

from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "portfolio_analysis.ipynb"


def markdown(source: str) -> nbformat.NotebookNode:
    """Create a clean Markdown cell."""
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str) -> nbformat.NotebookNode:
    """Create a clean code cell."""
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook(output_path: Path = OUTPUT_PATH) -> None:
    """Write the advanced, decision-oriented analysis notebook."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    notebook.cells = [
        markdown(
            """
# Quality-Adjusted Singapore HDB Resale Market Analysis

## Decision question

How much of the observed resale-price movement remains after controlling for the recorded mix of flats sold?

This report is designed for buyers and market analysts. It combines a chronological model evaluation, a hedonic price index, the official HDB Resale Price Index benchmark, distribution analysis and robustness checks. Results describe recorded transactions; they are not causal estimates or property valuations.
            """
        ),
        code(
            """
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path("..").resolve()
REPORTS = PROJECT_ROOT / "reports"
IMAGES = PROJECT_ROOT / "images"
DATA = PROJECT_ROOT / "data" / "processed"

PRIMARY = "#0B3C5D"
ACCENT = "#E07A5F"
plt.style.use("seaborn-v0_8-whitegrid")
pd.options.display.float_format = "{:,.2f}".format
            """
        ),
        markdown("## Reproducible data snapshot"),
        code(
            """
snapshot = json.loads((DATA / "snapshot_metadata.json").read_text(encoding="utf-8"))
parquet_path = DATA / "hdb_resale_clean.parquet"
transactions = pd.read_parquet(parquet_path)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert len(transactions) == snapshot["row_count"]
assert len(transactions.columns) == snapshot["column_count"]
assert sha256(parquet_path) == snapshot["processed_parquet_sha256"]

advanced_metadata = json.loads(
    (REPORTS / "advanced_analysis_metadata.json").read_text(encoding="utf-8")
)
assert advanced_metadata["transaction_source"]["sha256"] == snapshot["processed_sha256"]
assert advanced_metadata["transaction_source"]["row_count"] == snapshot["row_count"]

pd.Series({
    "Coverage": f"{snapshot['month_min']} to {snapshot['month_max']}",
    "Rows": f"{snapshot['row_count']:,}",
    "Columns": snapshot["column_count"],
    "Exact duplicate rows retained": f"{snapshot['fully_identical_rows_retained']:,}",
    "Parquet SHA-256": snapshot["processed_parquet_sha256"],
}, name="Snapshot")
            """
        ),
        markdown(
            """
## 1. Raw versus quality-adjusted movement

The model controls for floor area, remaining lease, storey, town, flat type and flat model. Month fixed effects form the quality-adjusted index. This separates recorded transaction mix from the common time component in the model; it does not establish causal appreciation.
            """
        ),
        code(
            """
model_report = json.loads(
    (REPORTS / "price_model_metrics.json").read_text(encoding="utf-8")
)
assert model_report["input_provenance"]["sha256"] == snapshot["processed_parquet_sha256"]
assert sha256(REPORTS / "quality_adjusted_price_index.csv") == model_report["output_sha256"]["quality_adjusted_index"]
index_summary = model_report["index_summary"]
provenance = model_report["input_provenance"]
coverage_note = ""
if provenance["provisional_latest_month_excluded"]:
    coverage_note = (
        f" The incomplete **{provenance['excluded_month']}** source month "
        f"({provenance['excluded_rows']:,} records) is excluded from the model and index."
    )

display(Markdown(f'''
From **{index_summary['base_month']}** through **{index_summary['latest_month']}**, the raw monthly median index increased **{index_summary['raw_change_from_base_pct']:.1f}%**. The quality-adjusted model increased **{index_summary['quality_adjusted_change_from_base_pct']:.1f}%**. The raw-versus-adjusted difference is **{index_summary['raw_minus_adjusted_change_percentage_points']:.1f} percentage points** under this model; it is not a causal attribution.{coverage_note}
'''))
display(Image(filename=str(IMAGES / "raw_vs_quality_adjusted_price_index.png")))
            """
        ),
        markdown("## 2. Chronological holdout evaluation"),
        code(
            """
split = model_report["split"]
metrics = (
    pd.DataFrame(model_report["holdout_metrics"])
      .T.rename_axis("Model")
      .rename(columns={"mae": "MAE (S$)", "rmse": "RMSE (S$)", "r2": "R²"})
)
display(Markdown(
    f"Training ends **{split['training_end_month']}**. The untouched holdout covers "
    f"**{split['holdout_start_month']} to {split['holdout_end_month']}** "
    f"({split['holdout_rows']:,} records)."
))
display(metrics.style.format({"MAE (S$)": "S${:,.0f}", "RMSE (S$)": "S${:,.0f}", "R²": "{:.3f}"}))
            """
        ),
        code(
            """
importance = pd.read_csv(REPORTS / "price_model_permutation_importance.csv")
importance_display = importance[[
    "feature_group", "mae_increase", "mae_increase_std", "r2_decrease"
]].copy()
importance_display.columns = [
    "Feature group", "MAE increase (S$)", "MAE increase SD", "R² decrease"
]
importance_display.style.format({
    "MAE increase (S$)": "S${:,.0f}",
    "MAE increase SD": "S${:,.0f}",
    "R² decrease": "{:.3f}",
})
            """
        ),
        markdown(
            """
Permutation importance is computed only after evaluating the model on the holdout. Negative values mean shuffling that group did not worsen the selected metric, often because predictors overlap; they are not evidence of a protective or causal effect.
            """
        ),
        markdown("## 3. Benchmark against the official HDB Resale Price Index"),
        code(
            """
rpi_benchmark = pd.read_csv(REPORTS / "quarterly_rpi_benchmark.csv")
display(Image(filename=str(IMAGES / "raw_median_vs_official_rpi.png")))
rpi_benchmark.tail(8)
            """
        ),
        markdown(
            """
The official RPI and raw transaction median are rebased to the first overlapping quarter for visual comparison. They are not interchangeable: the raw median changes with the types and locations of flats transacted, while the official index uses its own quality-adjustment methodology.
            """
        ),
        markdown("## 4. Price distribution, not only the median"),
        code(
            """
annual_distribution = pd.read_csv(REPORTS / "annual_price_quantiles.csv")
display(Image(filename=str(IMAGES / "annual_price_distribution.png")))
annual_distribution.tail().style.format({
    "p25": "S${:,.0f}",
    "p50": "S${:,.0f}",
    "p75": "S${:,.0f}",
    "p90": "S${:,.0f}",
    "transaction_count": "{:,.0f}",
})
            """
        ),
        markdown("## 5. Million-dollar transaction share"),
        code(
            """
million_share = pd.read_csv(REPORTS / "million_dollar_share_by_year.csv")
display(Image(filename=str(IMAGES / "million_dollar_share_by_year.png")))
million_share.tail().style.format({
    "transaction_count": "{:,.0f}",
    "million_dollar_transactions": "{:,.0f}",
    "million_dollar_share_pct": "{:.2f}%",
})
            """
        ),
        markdown("## 6. Remaining lease and observed price"),
        code(
            """
lease_bands = pd.read_csv(REPORTS / "remaining_lease_bands.csv")
display(Image(filename=str(IMAGES / "remaining_lease_price_bands.png")))
lease_bands.style.format({
    "transaction_count": "{:,.0f}",
    "p25": "S${:,.0f}",
    "median_resale_price": "S${:,.0f}",
    "p75": "S${:,.0f}",
})
            """
        ),
        markdown(
            """
Lease-band medians remain descriptive. Location, construction cohort and flat characteristics differ across bands. The hedonic model includes a nonlinear remaining-lease term, but its association should still not be interpreted as a standalone causal discount.
            """
        ),
        markdown("## 7. Town comparison with endpoint sample guards"),
        code(
            """
town_growth = pd.read_csv(REPORTS / "town_endpoint_growth.csv")
eligible_growth = town_growth.loc[town_growth["meets_minimum_sample"]].head(10)
display(Image(filename=str(IMAGES / "town_growth_with_sample_guard.png")))
eligible_growth[[
    "town", "first_year", "last_year", "first_year_count",
    "last_year_count", "median_price_growth_pct"
]].style.format({
    "first_year_count": "{:,.0f}",
    "last_year_count": "{:,.0f}",
    "median_price_growth_pct": "{:.1f}%",
})
            """
        ),
        markdown(
            """
Only towns with at least 100 records in both endpoint years are ranked. This is a change in the observed median transaction price, not a repeat-sales appreciation estimate; the public dataset does not identify individual units.
            """
        ),
        markdown("## 8. Robustness and cross-tool reconciliation"),
        code(
            """
duplicate_check = pd.read_csv(REPORTS / "duplicate_sensitivity.csv")
sql_check = pd.read_csv(REPORTS / "python_sql_reconciliation.csv")

display(Markdown(
    f"Removing **{snapshot['fully_identical_rows_retained']:,}** fully identical rows "
    "changes the overall median by **S$0** in this snapshot. They remain in the main "
    "analysis because the source lacks a transaction ID or unit number."
))
display(sql_check.style.format({
    "python_median": "S${:,.2f}",
    "sql_median": "S${:,.2f}",
    "absolute_difference_sgd": "S${:,.2f}",
}))
assert sql_check["matches_within_tolerance"].all()
            """
        ),
        markdown("## Conclusions for a buyer or market analyst"),
        code(
            """
latest_distribution = annual_distribution.iloc[-1]
latest_million = million_share.iloc[-1]
top_town = eligible_growth.iloc[0]

display(Markdown(f'''
1. **Separate market movement from transaction mix.** The raw index increased {index_summary['raw_change_from_base_pct']:.1f}% from the base month, compared with {index_summary['quality_adjusted_change_from_base_pct']:.1f}% after recorded characteristics were held constant in the model.
2. **Compare distributions and price per square metre, not only headline medians.** In {int(latest_distribution['year'])}, the observed P25-to-P90 range was S${latest_distribution['p25']:,.0f} to S${latest_distribution['p90']:,.0f}; that year currently contains {int(latest_distribution['months_observed'])} month labels.
3. **Treat premium-segment growth as a separate market signal.** Million-dollar transactions represent {latest_million['million_dollar_share_pct']:.2f}% of records in the latest partial year.
4. **Require adequate samples for town rankings.** {top_town['town'].title()} leads the guarded endpoint comparison, with {int(top_town['first_year_count']):,} and {int(top_town['last_year_count']):,} records at the two endpoints.

These findings support market comparison and monitoring. They do not identify the price of a specific flat, buyer demand, or causal premiums.
'''))
            """
        ),
        markdown(
            """
## Reproduce the report

From the project root:

```powershell
python src/clean_data.py
python src/build_database.py
python src/advanced_analysis.py --download-rpi
python src/model_price.py --input data/processed/hdb_resale_clean.parquet
python scripts/build_portfolio_notebook.py
```

The downloadable transaction data, official RPI snapshot, generated table metadata and model methodology are documented in the repository README and report files.
            """
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


if __name__ == "__main__":
    build_notebook()
