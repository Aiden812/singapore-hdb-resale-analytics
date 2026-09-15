"""Interactive explorer for Singapore HDB resale transactions."""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_PARQUET_PATH,
    SNAPSHOT_STALE_AFTER_DAYS,
    DashboardDataError,
    annual_trend,
    assess_snapshot_freshness,
    binned_price_profile,
    coverage_by_year,
    filter_transactions,
    find_comparable_sales,
    flat_type_distribution,
    load_processed_data,
    monthly_trend,
    price_histogram,
    town_summary,
)

st.set_page_config(
    page_title="HDB Resale Market Explorer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


TEAL = "#0F766E"
TEAL_LIGHT = "#5EEAD4"
AMBER = "#D97706"
NAVY = "#172033"
SLATE = "#64748B"
GRID = "#E2E8F0"
SOFT_BLUE = "#E0F2FE"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_INDEX_PATH = PROJECT_ROOT / "reports" / "quality_adjusted_price_index.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "reports" / "price_model_metrics.json"
SNAPSHOT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "snapshot_metadata.json"
ENRICHED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hdb_resale_enriched.parquet"
ENRICHMENT_METADATA_PATH = PROJECT_ROOT / "reports" / "official_enrichment_metadata.json"
MODEL_INTERVAL_METRICS_PATH = PROJECT_ROOT / "reports" / "price_model_interval_metrics.csv"
INFLATION_REPORT_PATH = PROJECT_ROOT / "reports" / "monthly_price_inflation.csv"
MRT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "mrt_proximity_summary.csv"
MRT_EXITS_PATH = PROJECT_ROOT / "reports" / "mrt_station_exits.csv"

st.markdown(
    """
    <style>
      .stApp { background: #f8fafc; }
      .block-container { max-width: 1480px; padding-top: 1.8rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.045);
      }
      [data-testid="stMetricLabel"] { color: #64748b; }
      [data-testid="stMetricValue"] { color: #172033; }
      .hero {
        padding: 1.55rem 1.75rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(120deg, #0f766e 0%, #115e59 56%, #172033 100%);
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.16);
        margin-bottom: 1.15rem;
      }
      .hero h1 { margin: 0.15rem 0 0.35rem; font-size: 2.15rem; line-height: 1.16; color: white; }
      .hero p { margin: 0; max-width: 850px; color: #ccfbf1; font-size: 1.02rem; }
      .eyebrow { font-size: 0.76rem; letter-spacing: 0.13em; text-transform: uppercase; font-weight: 700; color: #99f6e4; }
      .source-strip {
        display: flex; gap: 0.65rem; align-items: center; flex-wrap: wrap;
        margin: 0 0 1rem; color: #475569; font-size: 0.87rem;
      }
      .source-pill { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 999px; padding: 0.22rem 0.65rem; font-weight: 650; }
      .coverage-notice {
        margin: -0.25rem 0 1rem;
        padding: 0.58rem 0.78rem;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        border-radius: 10px;
        background: #fffbeb;
        color: #78350f;
        font-size: 0.86rem;
        line-height: 1.4;
      }
      .coverage-notice strong { color: #92400e; }
      .section-kicker { color: #0f766e; font-size: 0.78rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 750; margin-bottom: -0.45rem; }
      .small-note { color: #64748b; font-size: 0.86rem; line-height: 1.45; }
      div[data-testid="stPlotlyChart"] { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 0.3rem; }
      button[data-baseweb="tab"] { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def currency(value: float, *, compact: bool = False) -> str:
    """Format a Singapore-dollar value for labels and KPI cards."""
    if pd.isna(value):
        return "—"
    if compact and abs(value) >= 1_000_000:
        return f"S${value / 1_000_000:.2f}m"
    if compact and abs(value) >= 1_000:
        return f"S${value / 1_000:.0f}k"
    return f"S${value:,.0f}"


def percent(value: float) -> str:
    """Format a fractional value as a percentage."""
    return "—" if pd.isna(value) else f"{value:.1%}"


def readable_name(value: str) -> str:
    """Display uppercase source categories in title case."""
    return str(value).title().replace("Dbss", "DBSS")


@st.cache_data(show_spinner=False)
def load_disk_data(path_string: str, modified_ns: int, file_size: int) -> pd.DataFrame:
    """Cache a disk snapshot while invalidating when the file changes."""
    del modified_ns, file_size
    return load_processed_data(Path(path_string))


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for data and report provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def cached_file_sha256(path_string: str, modified_ns: int, file_size: int) -> str:
    """Cache a file digest while invalidating when the file changes."""
    del modified_ns, file_size
    return file_sha256(Path(path_string))


@st.cache_data(show_spinner=False)
def load_optional_report(path_string: str, modified_ns: int) -> pd.DataFrame:
    """Load a small optional CSV report and invalidate when it changes."""
    del modified_ns
    return pd.read_csv(path_string)


def enrichment_metadata() -> dict[str, object] | None:
    """Return verified enrichment metadata when it is readable."""
    if not ENRICHMENT_METADATA_PATH.is_file():
        return None
    try:
        metadata = json.loads(ENRICHMENT_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def tracked_snapshot_metadata() -> dict[str, object] | None:
    """Return the repository snapshot manifest when it is readable."""
    if not SNAPSHOT_METADATA_PATH.is_file():
        return None
    try:
        metadata = json.loads(SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def default_enriched_snapshot_is_verified() -> bool:
    """Require matching file hash and row count before preferring enrichment."""
    metadata = enrichment_metadata()
    if metadata is None or not ENRICHED_DATA_PATH.is_file():
        return False
    enrichment_input = metadata.get("input", {})
    output = metadata.get("output", {})
    if not isinstance(enrichment_input, dict) or not isinstance(output, dict):
        return False
    snapshot = tracked_snapshot_metadata()
    if snapshot is None:
        return False
    clean_hashes = {
        value
        for value in (
            snapshot.get("processed_parquet_sha256"),
            snapshot.get("processed_sha256"),
        )
        if isinstance(value, str)
    }
    if enrichment_input.get("sha256") not in clean_hashes:
        return False
    stat = ENRICHED_DATA_PATH.stat()
    actual_hash = cached_file_sha256(
        str(ENRICHED_DATA_PATH),
        stat.st_mtime_ns,
        stat.st_size,
    )
    if output.get("sha256") != actual_hash:
        return False
    input_rows = enrichment_input.get("row_count")
    expected_rows = output.get("row_count")
    snapshot_rows = snapshot.get("row_count")
    return (
        isinstance(expected_rows, int)
        and expected_rows > 0
        and input_rows == expected_rows
        and snapshot_rows == expected_rows
    )


def optional_report(path: Path) -> pd.DataFrame | None:
    """Load an optional report without making it a dashboard dependency."""
    if not path.is_file():
        return None
    try:
        return load_optional_report(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def provisional_source_month(
    transactions: pd.DataFrame,
    source_sha256: str,
    *,
    equivalent_hashes: tuple[str, ...] = (),
) -> dict[str, object] | None:
    """Return a provisional latest-month disclosure with verified source provenance."""
    as_of: pd.Timestamp | None = None
    if SNAPSHOT_METADATA_PATH.is_file():
        try:
            snapshot = json.loads(SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))
            matching_hashes = {
                snapshot.get("processed_sha256"),
                snapshot.get("processed_parquet_sha256"),
                *equivalent_hashes,
            }
            if source_sha256 in matching_hashes:
                timestamp_value = snapshot.get("source_modified_at_utc")
                if timestamp_value:
                    as_of = pd.Timestamp(timestamp_value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    if as_of is None:
        return None

    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("Asia/Singapore")
    else:
        as_of = as_of.tz_convert("Asia/Singapore")
    latest_month = transactions["month"].max().to_period("M")
    if latest_month != as_of.tz_localize(None).to_period("M"):
        return None

    month_rows = int(transactions["month"].dt.to_period("M").eq(latest_month).sum())
    return {
        "month": str(latest_month),
        "rows": month_rows,
        "as_of": as_of,
    }


def source_matches_model_snapshot(
    source_sha256: str,
    model_sha256: object,
) -> bool:
    """Accept identical files or the manifest's equivalent CSV/Parquet pair."""
    if source_sha256 == model_sha256:
        return True
    if not SNAPSHOT_METADATA_PATH.is_file() or not isinstance(model_sha256, str):
        return False
    try:
        snapshot = json.loads(SNAPSHOT_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    canonical_hashes = {
        value
        for value in (
            snapshot.get("processed_sha256"),
            snapshot.get("processed_parquet_sha256"),
        )
        if isinstance(value, str)
    }
    if {source_sha256, model_sha256}.issubset(canonical_hashes):
        return True

    metadata = enrichment_metadata()
    if metadata is None:
        return False
    enrichment_input = metadata.get("input", {})
    enrichment_output = metadata.get("output", {})
    if not isinstance(enrichment_input, dict) or not isinstance(
        enrichment_output, dict
    ):
        return False
    return (
        source_sha256 == enrichment_output.get("sha256")
        and model_sha256 == enrichment_input.get("sha256")
    )


@st.cache_data(show_spinner=False)
def load_uploaded_data(contents: bytes, file_format: str) -> pd.DataFrame:
    """Cache an uploaded snapshot by its bytes and file type."""
    return load_processed_data(io.BytesIO(contents), file_format=file_format)


@st.cache_data(show_spinner=False)
def load_model_artifacts(
    index_path: str,
    index_modified_ns: int,
    metrics_path: str,
    metrics_modified_ns: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the precomputed quality-adjusted index and holdout metrics."""
    del index_modified_ns, metrics_modified_ns
    with Path(metrics_path).open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    expected_index_hash = metrics.get("output_sha256", {}).get("quality_adjusted_index")
    if not expected_index_hash:
        raise ValueError("Model metrics do not include an index checksum")
    if file_sha256(Path(index_path)) != expected_index_hash:
        raise ValueError("Model index checksum does not match the metrics bundle")

    index = pd.read_csv(index_path, parse_dates=["month"])
    required = {
        "month",
        "raw_price_index",
        "quality_adjusted_price_index",
        "transactions",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError("Price-index report is missing: " + ", ".join(missing))
    return index.sort_values("month"), metrics


def apply_chart_style(
    figure: go.Figure,
    *,
    height: int = 390,
    legend: bool = True,
) -> go.Figure:
    """Apply a consistent, accessible house style to Plotly charts."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=28, r=24, t=48, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=13),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=NAVY, bordercolor=GRID),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
        modebar=dict(remove=["lasso2d", "select2d"]),
    )
    figure.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return figure


def show_data_error(details: str) -> None:
    """Render actionable setup guidance and stop the Streamlit run."""
    st.error(
        "The dashboard could not load a processed data snapshot.",
        icon="📂",
    )
    st.markdown(details)
    st.markdown("**Create the local data files with:**")
    st.code("python -m src.download_data\npython -m src.clean_data", language="bash")
    st.caption(
        "The dashboard prefers a checksum-verified enriched snapshot, then falls "
        "back to data/processed/hdb_resale_clean.csv. You can also upload either "
        "format from the sidebar."
    )
    st.stop()


def selected_data_source(
    uploaded_file: object | None,
) -> tuple[pd.DataFrame, str, str]:
    """Load the uploaded, configured, Parquet, or CSV data source in that order."""
    if uploaded_file is not None:
        uploaded_name = str(uploaded_file.name)
        uploaded_suffix = Path(uploaded_name).suffix.lower().lstrip(".")
        contents = uploaded_file.getvalue()
        return (
            load_uploaded_data(contents, uploaded_suffix),
            f"Uploaded snapshot · {uploaded_name}",
            hashlib.sha256(contents).hexdigest(),
        )

    configured_path = os.environ.get("HDB_DATA_PATH")
    candidates = [Path(configured_path)] if configured_path else []
    if default_enriched_snapshot_is_verified():
        candidates.append(ENRICHED_DATA_PATH)
    candidates.extend([DEFAULT_PARQUET_PATH, DEFAULT_DATA_PATH])
    for candidate in candidates:
        if candidate.is_file():
            stat = candidate.stat()
            data = load_disk_data(str(candidate), stat.st_mtime_ns, stat.st_size)
            if candidate == ENRICHED_DATA_PATH:
                metadata = enrichment_metadata() or {}
                output = metadata.get("output", {})
                expected_rows = (
                    output.get("row_count") if isinstance(output, dict) else None
                )
                if expected_rows != len(data):
                    continue
            try:
                shown_path = candidate.relative_to(Path(__file__).resolve().parent)
            except ValueError:
                shown_path = candidate
            digest = cached_file_sha256(
                str(candidate),
                stat.st_mtime_ns,
                stat.st_size,
            )
            return data, f"Project snapshot · {shown_path}", digest

    expected = "<br>".join(f"- `{path}`" for path in candidates)
    show_data_error(f"None of the expected data files was found:<br>{expected}")
    raise RuntimeError("Streamlit should have stopped after a missing-data error")


def monthly_price_figure(trend: pd.DataFrame, metric: str) -> go.Figure:
    """Build the monthly price trend with an optional interquartile band."""
    figure = go.Figure()
    if metric in {"Median resale price", "Median inflation-adjusted price"}:
        is_real = metric == "Median inflation-adjusted price"
        median_column = "median_real_price" if is_real else "median_price"
        q25_column = "real_price_q25" if is_real else "price_q25"
        q75_column = "real_price_q75" if is_real else "price_q75"
        figure.add_trace(
            go.Scatter(
                x=trend["month"],
                y=trend[q75_column],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=trend["month"],
                y=trend[q25_column],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(15, 118, 110, 0.12)",
                name="Middle 50%",
                hoverinfo="skip",
            )
        )
        y_values = trend[median_column]
        y_title = (
            "Median real resale price" if is_real else "Median resale price"
        )
        hover = "%{x|%b %Y}<br>Median: S$%{y:,.0f}<extra></extra>"
        title = (
            "Monthly median resale price (inflation-adjusted)"
            if is_real
            else "Monthly median resale price"
        )
    else:
        y_values = trend["median_price_per_sqm"]
        y_title = "Median price per sqm"
        hover = "%{x|%b %Y}<br>Median: S$%{y:,.0f}/sqm<extra></extra>"
        title = "Monthly median price per square metre"

    figure.add_trace(
        go.Scatter(
            x=trend["month"],
            y=y_values,
            mode="lines",
            line=dict(color=TEAL, width=3),
            marker=dict(color=TEAL, size=5),
            name=metric,
            customdata=trend[["transactions"]],
            hovertemplate=hover.replace(
                "<extra>", "<br>Transactions: %{customdata[0]:,}<extra>"
            ),
        )
    )
    apply_chart_style(figure, height=430)
    figure.update_layout(title=dict(text=title, x=0.01, font=dict(size=18)))
    figure.update_yaxes(title=y_title, tickprefix="S$", tickformat=",")
    figure.update_xaxes(
        title=None,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#F1F5F9",
            activecolor="#CCFBF1",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return figure


def million_share_figure(summary: pd.DataFrame) -> go.Figure:
    """Build the annual share of million-dollar transactions."""
    figure = go.Figure(
        go.Scatter(
            x=summary["year"],
            y=summary["million_dollar_share"],
            mode="lines+markers",
            line=dict(color=AMBER, width=3),
            marker=dict(size=9, color=AMBER, line=dict(color="white", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(217, 119, 6, 0.10)",
            hovertemplate="%{x}<br>Share: %{y:.2%}<extra></extra>",
        )
    )
    apply_chart_style(figure, height=390, legend=False)
    figure.update_layout(
        title=dict(text="Million-dollar transaction share", x=0.01, font=dict(size=18))
    )
    figure.update_xaxes(dtick=1, title=None)
    figure.update_yaxes(
        title="Share of transactions", tickformat=".1%", rangemode="tozero"
    )
    return figure


def annual_market_figure(
    summary: pd.DataFrame,
    full_coverage: pd.DataFrame,
) -> go.Figure:
    """Build the annual transaction-volume and median-price view."""
    source_coverage = full_coverage[["year", "months_observed", "is_partial"]].rename(
        columns={"months_observed": "source_months_observed"}
    )
    chart_data = summary.drop(columns="is_partial", errors="ignore").merge(
        source_coverage,
        on="year",
        how="left",
        suffixes=("", "_source"),
    )
    partial = chart_data["is_partial"].fillna(False)
    bar_colors = np.where(partial, "#F59E0B", "#BAE6FD")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=chart_data["year"],
            y=chart_data["transactions"],
            marker_color=bar_colors,
            name="Transactions",
            customdata=chart_data[["source_months_observed"]],
            hovertemplate=(
                "%{x}<br>Transactions: %{y:,}<br>Months in source: "
                "%{customdata[0]:.0f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data["year"],
            y=chart_data["median_price"],
            mode="lines+markers",
            line=dict(color=TEAL, width=3),
            marker=dict(size=9, color=TEAL, line=dict(color="white", width=1.5)),
            name="Median price",
            hovertemplate="%{x}<br>Median: S$%{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_chart_style(figure, height=420)
    figure.update_layout(
        title=dict(text="Annual market direction", x=0.01, font=dict(size=18)),
        bargap=0.35,
    )
    figure.update_xaxes(dtick=1, title=None)
    figure.update_yaxes(title_text="Transactions", tickformat=",", secondary_y=False)
    figure.update_yaxes(
        title_text="Median resale price",
        tickprefix="S$",
        tickformat=",",
        secondary_y=True,
        showgrid=False,
    )
    return figure


def market_mix_figure(index: pd.DataFrame) -> go.Figure:
    """Compare the observed median-price index with a quality-adjusted index."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=index["month"],
            y=index["raw_price_index"],
            mode="lines",
            line=dict(color=SLATE, width=2.5, dash="dot"),
            name="Raw median-price index",
            customdata=index[["transactions"]],
            hovertemplate=(
                "%{x|%b %Y}<br>Raw index: %{y:.1f}"
                "<br>Transactions: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=index["month"],
            y=index["quality_adjusted_price_index"],
            mode="lines",
            line=dict(color=TEAL, width=3.2),
            name="Quality-adjusted index",
            customdata=index[["transactions"]],
            hovertemplate=(
                "%{x|%b %Y}<br>Adjusted index: %{y:.1f}"
                "<br>Transactions: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=100,
        line_width=1,
        line_dash="dash",
        line_color=GRID,
        annotation_text="Jan 2017 baseline",
        annotation_position="bottom right",
    )
    apply_chart_style(figure, height=470)
    figure.update_layout(
        title=dict(
            text="Raw vs quality-adjusted price index",
            subtitle=dict(
                text="January 2017 = 100 · whole-market model",
                font=dict(size=12, color=SLATE),
            ),
            x=0.01,
            font=dict(size=18),
        )
    )
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="Price index", tickformat=".0f")
    return figure


def town_ranking_figure(summary: pd.DataFrame, metric: str, top_n: int) -> go.Figure:
    """Build a sample-size-aware horizontal town ranking."""
    metric_map = {
        "Median resale price": "median_price",
        "Median price per sqm": "median_price_per_sqm",
        "Million-dollar share": "million_dollar_share",
        "Transaction volume": "transactions",
    }
    column = metric_map[metric]
    ranked = summary.nlargest(top_n, column).sort_values(column)
    customdata = np.column_stack(
        [
            ranked["transactions"],
            ranked["median_price"],
            ranked["median_price_per_sqm"],
            ranked["million_dollar_share"],
        ]
    )
    if column == "million_dollar_share":
        text = ranked[column].map(lambda value: f"{value:.1%}")
        x_tickformat = ".1%"
        x_prefix = ""
    elif column == "transactions":
        text = ranked[column].map(lambda value: f"{value:,.0f}")
        x_tickformat = ","
        x_prefix = ""
    else:
        text = ranked[column].map(lambda value: currency(value, compact=True))
        x_tickformat = ","
        x_prefix = "S$"

    figure = go.Figure(
        go.Bar(
            x=ranked[column],
            y=ranked["town"].map(readable_name),
            orientation="h",
            marker=dict(
                color=ranked[column],
                colorscale=[[0, "#99F6E4"], [1, TEAL]],
                line=dict(width=0),
            ),
            text=text,
            textposition="outside",
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>Transactions: %{customdata[0]:,.0f}"
                "<br>Median price: S$%{customdata[1]:,.0f}"
                "<br>Median S$/sqm: S$%{customdata[2]:,.0f}"
                "<br>Million-dollar share: %{customdata[3]:.2%}<extra></extra>"
            ),
        )
    )
    apply_chart_style(figure, height=max(390, 31 * len(ranked) + 105), legend=False)
    figure.update_layout(
        title=dict(text=f"Town ranking · {metric.lower()}", x=0.01, font=dict(size=18))
    )
    figure.update_xaxes(title=metric, tickprefix=x_prefix, tickformat=x_tickformat)
    figure.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    return figure


def histogram_figure(histogram: pd.DataFrame) -> go.Figure:
    """Build an aggregated transaction-price histogram."""
    widths = histogram["bin_right"] - histogram["bin_left"]
    figure = go.Figure(
        go.Bar(
            x=histogram["bin_mid"],
            y=histogram["transactions"],
            width=widths * 0.92,
            marker_color=TEAL,
            customdata=histogram[["bin_left", "bin_right"]],
            hovertemplate=(
                "S$%{customdata[0]:,.0f}–S$%{customdata[1]:,.0f}"
                "<br>Transactions: %{y:,}<extra></extra>"
            ),
        )
    )
    apply_chart_style(figure, height=400, legend=False)
    figure.update_layout(
        title=dict(text="Resale price distribution", x=0.01, font=dict(size=18)),
        bargap=0.02,
    )
    figure.update_xaxes(title="Resale price", tickprefix="S$", tickformat="~s")
    figure.update_yaxes(title="Transactions", tickformat=",")
    return figure


def flat_type_distribution_figure(summary: pd.DataFrame) -> go.Figure:
    """Build a compact 10th–90th percentile plot by flat type."""
    labels = summary["flat_type"].map(readable_name)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=summary["price_q75"] - summary["price_q25"],
            base=summary["price_q25"],
            y=labels,
            orientation="h",
            marker_color="rgba(15, 118, 110, 0.28)",
            name="Middle 50%",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=summary["median_price"],
            y=labels,
            mode="markers",
            marker=dict(color=TEAL, size=11, symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=summary["price_q90"] - summary["median_price"],
                arrayminus=summary["median_price"] - summary["price_q10"],
                color=SLATE,
                thickness=1.4,
                width=3,
            ),
            customdata=summary[["transactions", "price_q25", "price_q75"]],
            name="Median and 10th–90th percentile",
            hovertemplate=(
                "%{y}<br>Median: S$%{x:,.0f}"
                "<br>Middle 50%: S$%{customdata[1]:,.0f}–S$%{customdata[2]:,.0f}"
                "<br>Transactions: %{customdata[0]:,.0f}<extra></extra>"
            ),
        )
    )
    apply_chart_style(figure, height=max(400, 55 * len(summary) + 100))
    figure.update_layout(
        title=dict(text="Price range by flat type", x=0.01, font=dict(size=18)),
        barmode="overlay",
    )
    figure.update_xaxes(title="Resale price", tickprefix="S$", tickformat="~s")
    figure.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    return figure


def transaction_mix_figure(data: pd.DataFrame) -> go.Figure:
    """Build the annual percentage mix of flat types."""
    mix = data.groupby(["year", "flat_type"], as_index=False).agg(
        transactions=("resale_price", "size")
    )
    totals = mix.groupby("year")["transactions"].transform("sum")
    mix["share"] = mix["transactions"] / totals
    figure = go.Figure()
    palette = [TEAL, "#0284C7", AMBER, "#7C3AED", "#E11D48", "#64748B", "#14B8A6"]
    for index, flat_type in enumerate(sorted(mix["flat_type"].unique())):
        category = mix[mix["flat_type"].eq(flat_type)]
        figure.add_trace(
            go.Bar(
                x=category["year"],
                y=category["share"],
                name=readable_name(flat_type),
                marker_color=palette[index % len(palette)],
                customdata=category[["transactions"]],
                hovertemplate=(
                    "%{x}<br>Share: %{y:.1%}<br>Transactions: "
                    "%{customdata[0]:,}<extra></extra>"
                ),
            )
        )
    apply_chart_style(figure, height=410)
    figure.update_layout(
        title=dict(text="Transaction mix by flat type", x=0.01, font=dict(size=18)),
        barmode="stack",
    )
    figure.update_xaxes(dtick=1, title=None)
    figure.update_yaxes(title="Share of transactions", tickformat=".0%", range=[0, 1])
    return figure


def profile_figure(
    profile: pd.DataFrame,
    *,
    x_column: str,
    x_title: str,
    metric: str,
    title: str,
) -> go.Figure:
    """Build a binned descriptive relationship with sample-size cues."""
    metric_column = (
        "median_price" if metric == "Median resale price" else "median_price_per_sqm"
    )
    suffix = "" if metric == "Median resale price" else "/sqm"
    marker_sizes = np.clip(np.sqrt(profile["transactions"]) * 0.85, 7, 24)
    figure = go.Figure(
        go.Scatter(
            x=profile[x_column],
            y=profile[metric_column],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(
                color=profile["transactions"],
                colorscale=[[0, TEAL_LIGHT], [1, TEAL]],
                size=marker_sizes,
                line=dict(color="white", width=1),
                showscale=False,
            ),
            customdata=profile[["transactions"]],
            hovertemplate=(
                "%{x:.1f}<br>Median: S$%{y:,.0f}"
                + suffix
                + "<br>Transactions: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    apply_chart_style(figure, height=390, legend=False)
    figure.update_layout(title=dict(text=title, x=0.01, font=dict(size=18)))
    figure.update_xaxes(title=x_title)
    figure.update_yaxes(title=metric, tickprefix="S$", tickformat=",")
    return figure


def comparable_distribution_figure(
    data: pd.DataFrame,
    *,
    price_column: str,
    price_per_sqm_column: str,
    price_basis: str,
) -> go.Figure:
    """Show the observed price and price-per-sqm distributions for comparables."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{price_basis} transaction prices",
            f"{price_basis} price per sqm",
        ),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Histogram(
            x=data[price_column],
            nbinsx=28,
            marker_color=TEAL,
            opacity=0.86,
            name="Resale price",
            hovertemplate="Price: S$%{x:,.0f}<br>Transactions: %{y:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=data[price_per_sqm_column],
            nbinsx=28,
            marker_color=AMBER,
            opacity=0.86,
            name="Price per sqm",
            hovertemplate=(
                "Price per sqm: S$%{x:,.0f}<br>Transactions: "
                "%{y:,}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    apply_chart_style(figure, height=390, legend=False)
    figure.update_layout(
        title=dict(text="Observed comparable-sale distributions", x=0.01, font=dict(size=18)),
        bargap=0.05,
    )
    figure.update_xaxes(title="Transaction price", tickprefix="S$", tickformat=",", row=1, col=1)
    figure.update_xaxes(title="Price per sqm", tickprefix="S$", tickformat=",", row=1, col=2)
    figure.update_yaxes(title="Transactions", row=1, col=1)
    figure.update_yaxes(title="Transactions", row=1, col=2)
    return figure


def comparable_map_figure(
    comparable_sales: pd.DataFrame,
    mrt_exits: pd.DataFrame | None,
) -> go.Figure | None:
    """Map comparable blocks with an optional official MRT-station layer."""
    figure = go.Figure()
    centre_latitudes: list[float] = []
    centre_longitudes: list[float] = []
    has_sales_points = False

    if {"latitude", "longitude"}.issubset(comparable_sales.columns):
        sales_points = comparable_sales.dropna(subset=["latitude", "longitude"]).copy()
        if not sales_points.empty:
            group_columns = [
                column
                for column in ("block", "street_name", "latitude", "longitude")
                if column in sales_points.columns
            ]
            sales_points = (
                sales_points.groupby(group_columns, as_index=False, dropna=False)
                .agg(
                    transactions=("resale_price", "size"),
                    median_resale_price=("resale_price", "median"),
                )
                .sort_values("transactions", ascending=False)
            )
            labels = []
            for _, row in sales_points.iterrows():
                address = " ".join(
                    str(row[column])
                    for column in ("block", "street_name")
                    if column in group_columns and pd.notna(row[column])
                )
                labels.append(
                    f"{address or 'Comparable block'}"
                    f"<br>Median: {currency(float(row['median_resale_price']))}"
                    f"<br>Matches: {int(row['transactions']):,}"
                )
            figure.add_trace(
                go.Scattermap(
                    lat=sales_points["latitude"],
                    lon=sales_points["longitude"],
                    mode="markers",
                    marker=dict(size=11, color=TEAL, opacity=0.82),
                    text=labels,
                    hovertemplate="%{text}<extra>Comparable block</extra>",
                    name="Comparable blocks",
                )
            )
            has_sales_points = True
            centre_latitudes.extend(sales_points["latitude"].astype(float).tolist())
            centre_longitudes.extend(sales_points["longitude"].astype(float).tolist())

    if (
        mrt_exits is not None
        and {"latitude", "longitude"}.issubset(mrt_exits.columns)
    ):
        stations = mrt_exits.dropna(subset=["latitude", "longitude"]).copy()
        station_name_column = next(
            (
                column
                for column in ("station_name", "nearest_mrt_station", "name")
                if column in stations.columns
            ),
            None,
        )
        if not stations.empty:
            station_labels = (
                stations[station_name_column].astype(str)
                if station_name_column
                else pd.Series("MRT exit", index=stations.index)
            )
            figure.add_trace(
                go.Scattermap(
                    lat=stations["latitude"],
                    lon=stations["longitude"],
                    mode="markers",
                    marker=dict(size=7, color=AMBER, opacity=0.68),
                    text=station_labels,
                    hovertemplate="%{text}<extra>MRT station exit</extra>",
                    name="MRT exits",
                )
            )
            if not centre_latitudes:
                centre_latitudes.extend(stations["latitude"].astype(float).tolist())
                centre_longitudes.extend(stations["longitude"].astype(float).tolist())

    if not figure.data or not centre_latitudes:
        return None
    apply_chart_style(figure, height=500)
    figure.update_layout(
        title=dict(
            text=(
                "Comparable blocks with MRT reference layer"
                if has_sales_points
                else "MRT reference layer"
            ),
            x=0.01,
            font=dict(size=18),
        ),
        map=dict(
            style="open-street-map",
            center=dict(
                lat=float(np.median(centre_latitudes)),
                lon=float(np.median(centre_longitudes)),
            ),
            zoom=11,
        ),
        margin=dict(l=12, r=12, t=58, b=12),
    )
    return figure


def exact_storey_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prices by the midpoint of each source storey band."""
    valid = data.dropna(subset=["storey_mid"])
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby("storey_mid", as_index=False)
        .agg(
            median_price=("resale_price", "median"),
            median_price_per_sqm=("price_per_sqm", "median"),
            transactions=("resale_price", "size"),
        )
        .sort_values("storey_mid")
        .reset_index(drop=True)
    )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Singapore housing intelligence</div>
      <h1>HDB Resale Market Explorer</h1>
      <p>Explore price direction, transaction mix, town differences and the
      property characteristics associated with resale prices.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Explore the market")
    st.caption(
        "Filters update transaction views. Market vs mix uses the precomputed "
        "whole-market model."
    )
    uploaded_snapshot = st.file_uploader(
        "Use another processed snapshot",
        type=["parquet", "csv"],
        help="Optional. The project snapshot is used when no file is uploaded.",
    )

try:
    transactions, source_label, source_sha256 = selected_data_source(uploaded_snapshot)
except DashboardDataError as exc:
    show_data_error(str(exc))

snapshot_freshness = None
tracked_snapshot = tracked_snapshot_metadata()
verified_equivalent_hashes: tuple[str, ...] = ()
if uploaded_snapshot is None and tracked_snapshot is not None:
    if default_enriched_snapshot_is_verified():
        enrichment = enrichment_metadata() or {}
        enrichment_output = enrichment.get("output", {})
        enriched_hash = (
            enrichment_output.get("sha256")
            if isinstance(enrichment_output, dict)
            else None
        )
        if isinstance(enriched_hash, str):
            verified_equivalent_hashes = (enriched_hash,)
    snapshot_freshness = assess_snapshot_freshness(
        tracked_snapshot,
        source_sha256,
        equivalent_hashes=verified_equivalent_hashes,
    )

all_years = sorted(int(year) for year in transactions["year"].unique())
all_towns = sorted(str(town) for town in transactions["town"].dropna().unique())
all_flat_types = sorted(
    str(flat_type) for flat_type in transactions["flat_type"].dropna().unique()
)

with st.sidebar:
    st.divider()
    st.subheader("Filters")
    if len(all_years) > 1:
        selected_year_range = st.select_slider(
            "Transaction year",
            options=all_years,
            value=(all_years[0], all_years[-1]),
        )
    else:
        selected_year_range = (all_years[0], all_years[0])
        st.text_input("Transaction year", value=str(all_years[0]), disabled=True)
    selected_years = list(
        range(int(selected_year_range[0]), int(selected_year_range[1]) + 1)
    )
    selected_towns = st.multiselect(
        "Town",
        options=all_towns,
        default=[],
        format_func=readable_name,
        placeholder="All towns",
        help="Leave empty to include every town.",
    )
    selected_flat_types = st.multiselect(
        "Flat type",
        options=all_flat_types,
        default=[],
        format_func=readable_name,
        placeholder="All flat types",
        help="Leave empty to include every flat type.",
    )

filtered = filter_transactions(
    transactions,
    years=selected_years,
    towns=selected_towns or None,
    flat_types=selected_flat_types or None,
)

with st.sidebar:
    st.divider()
    st.metric("Matching transactions", f"{len(filtered):,}")
    st.caption(
        f"Source contains {len(transactions):,} rows from "
        f"{transactions['month'].min():%b %Y} to {transactions['month'].max():%b %Y}."
    )

if filtered.empty:
    st.warning(
        "No transactions match this combination. Widen the year range or clear "
        "one of the category filters.",
        icon="🔎",
    )
    st.stop()

full_coverage = coverage_by_year(transactions)
monthly = monthly_trend(filtered)
annual = annual_trend(filtered)
latest_annual = annual.iloc[-1]
previous_annual = annual.iloc[-2] if len(annual) > 1 else None

model_index: pd.DataFrame | None = None
model_metrics: dict[str, object] | None = None
model_artifact_error: str | None = None
if MODEL_INDEX_PATH.is_file() and MODEL_METRICS_PATH.is_file():
    try:
        model_index, model_metrics = load_model_artifacts(
            str(MODEL_INDEX_PATH),
            MODEL_INDEX_PATH.stat().st_mtime_ns,
            str(MODEL_METRICS_PATH),
            MODEL_METRICS_PATH.stat().st_mtime_ns,
        )
        model_input_hash = model_metrics.get("input_provenance", {}).get("sha256")
        if not source_matches_model_snapshot(source_sha256, model_input_hash):
            raise ValueError(
                "Model artifacts were generated from a different data snapshot"
            )
    except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        model_artifact_error = f"The model reports could not be read: {exc}"
else:
    model_artifact_error = (
        "The precomputed model reports are not present in this checkout."
    )

escaped_source = html.escape(source_label)
freshness_strip = ""
if snapshot_freshness is not None:
    observed_at = snapshot_freshness.observed_at_sgt
    freshness_strip = (
        f"<span>•</span><span>Observed {observed_at.day} "
        f"{observed_at:%b %Y} SGT</span><span>•</span>"
        f"<span>Age {snapshot_freshness.age_days} calendar days</span>"
    )
st.markdown(
    (
        '<div class="source-strip"><span class="source-pill">Data loaded</span>'
        f"<span>{escaped_source}</span><span>•</span>"
        f"<span>Filtered through {filtered['month'].max():%b %Y}</span>"
        f"{freshness_strip}</div>"
    ),
    unsafe_allow_html=True,
)

if snapshot_freshness is not None and snapshot_freshness.is_stale:
    observed_at = snapshot_freshness.observed_at_sgt
    st.warning(
        f"The official snapshot was last observed on {observed_at.day} "
        f"{observed_at:%B %Y} Singapore time and is "
        f"{snapshot_freshness.age_days} calendar days old. This is beyond the "
        f"{SNAPSHOT_STALE_AFTER_DAYS}-day freshness target; refresh the data "
        "before relying on recent-market conclusions.",
        icon="⏳",
    )

latest_source_year = int(full_coverage["year"].max())
latest_coverage = full_coverage.loc[full_coverage["year"].eq(latest_source_year)].iloc[
    0
]
provisional_month = provisional_source_month(
    transactions,
    source_sha256,
    equivalent_hashes=verified_equivalent_hashes,
)
coverage_notes: list[str] = []
if latest_source_year in selected_years and bool(latest_coverage["is_partial"]):
    coverage_notes.append(
        f"{latest_source_year} is partial ({int(latest_coverage['months_observed'])} "
        f"of 12 months, through {latest_coverage['last_month']:%B %Y}); annual "
        "comparisons need that context."
    )
if (
    provisional_month is not None
    and filtered["month"]
    .dt.to_period("M")
    .eq(pd.Period(str(provisional_month["month"]), freq="M"))
    .any()
):
    as_of = provisional_month["as_of"]
    model_note = (
        " The model index excludes it."
        if model_metrics is not None
        and model_metrics.get("input_provenance", {}).get(
            "provisional_latest_month_excluded"
        )
        else ""
    )
    coverage_notes.append(
        f"{provisional_month['month']} is provisional as of "
        f"{as_of.day} {as_of:%B %Y} Singapore time "
        f"({int(provisional_month['rows']):,} records); transaction views include "
        f"it.{model_note}"
    )
if coverage_notes:
    st.markdown(
        '<div class="coverage-notice"><strong>Data coverage</strong> · '
        + " ".join(coverage_notes)
        + "</div>",
        unsafe_allow_html=True,
    )
if len(filtered) < 100:
    transaction_word = "transaction" if len(filtered) == 1 else "transactions"
    st.info(
        f"Only {len(filtered):,} {transaction_word} match the filters. Medians and town "
        "rankings may move substantially with a few additional sales.",
        icon="ℹ️",
    )

price_delta = None
if previous_annual is not None and previous_annual["median_price"]:
    price_change = latest_annual["median_price"] / previous_annual["median_price"] - 1
    price_delta = f"{price_change:+.1%} vs {int(previous_annual['year'])}"

kpi_columns = st.columns(4)
with kpi_columns[0]:
    st.metric(
        "Transactions",
        f"{len(filtered):,}",
        help="Number of registered transactions matching the active filters.",
    )
with kpi_columns[1]:
    st.metric(
        "Median resale price",
        currency(float(filtered["resale_price"].median())),
        delta=price_delta,
        help="Median across the full filtered period; delta compares the latest two included years.",
    )
with kpi_columns[2]:
    st.metric(
        "Median S$/sqm",
        currency(float(filtered["price_per_sqm"].median())),
        help="Median price per square metre: resale price divided by floor area.",
    )
with kpi_columns[3]:
    st.metric(
        "Million-dollar share",
        percent(float(filtered["is_million_dollar"].mean())),
        help="Share of filtered transactions registered at S$1,000,000 or above.",
    )

st.caption(
    "Headline figures describe all matching transactions. Hover over charts for "
    "sample sizes and exact values."
)

(
    overview_tab,
    comparable_tab,
    market_mix_tab,
    mix_tab,
    drivers_tab,
    notes_tab,
) = st.tabs(
    [
        "Overview",
        "Comparable sales",
        "Market vs mix",
        "Price & mix",
        "Property profiles",
        "Data notes",
    ]
)

with overview_tab:
    st.markdown(
        '<p class="section-kicker">Direction over time</p>', unsafe_allow_html=True
    )
    trend_options = ["Median resale price"]
    if "median_real_price" in monthly.columns and monthly[
        "median_real_price"
    ].notna().any():
        trend_options.append("Median inflation-adjusted price")
    trend_options.append("Median price per sqm")
    trend_metric = st.radio(
        "Monthly trend metric",
        trend_options,
        horizontal=True,
        label_visibility="collapsed",
        key="monthly_metric",
    )
    trend_column, annual_column = st.columns([1.08, 1], gap="large")
    with trend_column:
        st.plotly_chart(
            monthly_price_figure(monthly, trend_metric),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
        if trend_metric == "Median inflation-adjusted price":
            last_real_month = monthly.loc[
                monthly["median_real_price"].notna(), "month"
            ].max()
            reference_values = (
                transactions["cpi_reference_month"].dropna()
                if "cpi_reference_month" in transactions.columns
                else pd.Series(dtype="string")
            )
            reference_note = (
                f", restated to {reference_values.iloc[0]}"
                if not reference_values.empty
                else ""
            )
            st.caption(
                f"Real-price series available through {last_real_month:%b %Y}"
                f"{reference_note}; later months await official CPI publication."
            )
    with annual_column:
        st.plotly_chart(
            annual_market_figure(annual, full_coverage),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )

    st.markdown(
        '<p class="section-kicker">Place and premium</p>', unsafe_allow_html=True
    )
    control_columns = st.columns([1.25, 1, 1])
    with control_columns[0]:
        ranking_metric = st.selectbox(
            "Rank towns by",
            [
                "Median resale price",
                "Median price per sqm",
                "Million-dollar share",
                "Transaction volume",
            ],
        )
    maximum_town_count = int(filtered.groupby("town").size().max())
    with control_columns[1]:
        minimum_town_count = int(
            st.number_input(
                "Minimum transactions per town",
                min_value=1,
                max_value=max(1, maximum_town_count),
                value=min(100, max(1, maximum_town_count)),
                step=10 if maximum_town_count >= 10 else 1,
                help="Excludes rankings based on very small samples.",
            )
        )
    eligible_towns = town_summary(filtered, minimum_transactions=minimum_town_count)
    with control_columns[2]:
        if eligible_towns.empty:
            towns_shown = 0
            st.text_input("Towns shown", value="None eligible", disabled=True)
        elif len(eligible_towns) == 1:
            towns_shown = 1
            st.text_input("Towns shown", value="1", disabled=True)
        else:
            towns_shown = st.slider(
                "Towns shown",
                min_value=1,
                max_value=min(20, len(eligible_towns)),
                value=min(12, len(eligible_towns)),
            )

    ranking_column, premium_column = st.columns([1.35, 1], gap="large")
    with ranking_column:
        if eligible_towns.empty:
            st.info(
                "No town reaches the selected sample threshold. Lower the minimum "
                "transaction count to display the ranking."
            )
        else:
            st.plotly_chart(
                town_ranking_figure(eligible_towns, ranking_metric, towns_shown),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
    with premium_column:
        st.plotly_chart(
            million_share_figure(annual),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
        st.markdown(
            '<p class="small-note">A million-dollar transaction is one registered '
            "at S$1,000,000 or above. The share is usually more comparable than the "
            "raw count when annual transaction volume changes.</p>",
            unsafe_allow_html=True,
        )


with comparable_tab:
    st.markdown(
        '<p class="section-kicker">Recent evidence for a similar flat</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Comparable Sales Explorer")
    st.write(
        "Choose a property profile to inspect registered transactions in the same "
        "town and flat type. Matching starts narrowly and widens only when fewer "
        "than 20 observations are available."
    )
    st.warning(
        "This is an exploratory comparison of historical transactions—not a formal "
        "valuation, appraisal, offer recommendation or guarantee of a future sale price.",
        icon="⚖️",
    )

    selector_columns = st.columns(2, gap="large")
    with selector_columns[0]:
        comparable_town = st.selectbox(
            "Town",
            options=all_towns,
            format_func=readable_name,
            key="comparable_town",
        )
    available_comparable_types = sorted(
        transactions.loc[
            transactions["town"].eq(comparable_town), "flat_type"
        ].dropna().unique()
    )
    preferred_type_index = (
        available_comparable_types.index("4 ROOM")
        if "4 ROOM" in available_comparable_types
        else 0
    )
    with selector_columns[1]:
        comparable_flat_type = st.selectbox(
            "Flat type",
            options=available_comparable_types,
            index=preferred_type_index,
            format_func=readable_name,
            key="comparable_flat_type",
        )

    comparable_pool = transactions.loc[
        transactions["town"].eq(comparable_town)
        & transactions["flat_type"].eq(comparable_flat_type)
    ]
    area_values = comparable_pool["floor_area_sqm"].dropna()
    default_area = float(area_values.median())
    detail_columns = st.columns(4, gap="medium")
    with detail_columns[0]:
        target_floor_area = float(
            st.number_input(
                "Floor area (sqm)",
                min_value=float(area_values.min()),
                max_value=float(area_values.max()),
                value=float(round(default_area)),
                step=1.0,
                key="comparable_area",
            )
        )

    storey_values = comparable_pool["storey_mid"].dropna()
    with detail_columns[1]:
        if storey_values.empty:
            target_storey = None
            st.text_input(
                "Approximate storey",
                value="Unavailable",
                disabled=True,
                key="comparable_storey_unavailable",
            )
        else:
            target_storey = float(
                st.number_input(
                    "Approximate storey",
                    min_value=float(storey_values.min()),
                    max_value=float(storey_values.max()),
                    value=float(round(float(storey_values.median()))),
                    step=1.0,
                    help="Midpoint of the source three-storey band.",
                    key="comparable_storey",
                )
            )

    lease_values = comparable_pool["remaining_lease_years"].dropna()
    with detail_columns[2]:
        if lease_values.empty:
            target_lease = None
            st.text_input(
                "Remaining lease",
                value="Unavailable",
                disabled=True,
                key="comparable_lease_unavailable",
            )
        else:
            target_lease = float(
                st.number_input(
                    "Remaining lease (years)",
                    min_value=float(max(0, np.floor(lease_values.min()))),
                    max_value=float(np.ceil(lease_values.max())),
                    value=float(round(float(lease_values.median()))),
                    step=1.0,
                    key="comparable_lease",
                )
            )

    with detail_columns[3]:
        comparable_months = int(
            st.selectbox(
                "Recent period",
                options=[12, 24, 36, 60],
                index=1,
                format_func=lambda months: f"Latest {months} months",
                key="comparable_months",
            )
        )

    comparable_result = find_comparable_sales(
        transactions,
        town=comparable_town,
        flat_type=comparable_flat_type,
        floor_area_sqm=target_floor_area,
        storey_mid=target_storey,
        remaining_lease_years=target_lease,
        recent_months=comparable_months,
        minimum_transactions=20,
    )
    comparable_sales = comparable_result.transactions
    if comparable_sales.empty:
        st.info(comparable_result.explanation, icon="🔎")
    else:
        st.info(comparable_result.explanation, icon="🧭")
        real_price_columns_available = {
            "resale_price_real_sgd",
            "price_per_sqm_real_sgd",
        }.issubset(comparable_sales.columns) and comparable_sales[
            ["resale_price_real_sgd", "price_per_sqm_real_sgd"]
        ].dropna().shape[0] > 0
        if real_price_columns_available:
            price_basis = st.radio(
                "Price basis",
                ["Nominal", "Inflation-adjusted"],
                horizontal=True,
                key="comparable_price_basis",
                help=(
                    "Inflation-adjusted values are expressed in the CPI reference "
                    "month documented by the enrichment report."
                ),
            )
        else:
            price_basis = "Nominal"

        if price_basis == "Inflation-adjusted":
            price_column = "resale_price_real_sgd"
            price_per_sqm_column = "price_per_sqm_real_sgd"
            summary_data = comparable_sales.dropna(
                subset=[price_column, price_per_sqm_column]
            )
        else:
            price_column = "resale_price"
            price_per_sqm_column = "price_per_sqm"
            summary_data = comparable_sales

        if price_basis == "Inflation-adjusted" and len(summary_data) < len(
            comparable_sales
        ):
            missing_real_prices = len(comparable_sales) - len(summary_data)
            reference_values = (
                comparable_sales["cpi_reference_month"].dropna()
                if "cpi_reference_month" in comparable_sales.columns
                else pd.Series(dtype="string")
            )
            reference_note = (
                f" Values are restated to {reference_values.iloc[0]}."
                if not reference_values.empty
                else ""
            )
            st.caption(
                f"{missing_real_prices:,} recent matches are omitted because official "
                f"CPI was not yet available for their transaction month.{reference_note}"
            )
        price_quantiles = summary_data[price_column].quantile([0.10, 0.90])
        comparable_kpis = st.columns(4)
        with comparable_kpis[0]:
            st.metric(
                "Comparable transactions",
                f"{len(summary_data):,}",
                help=f"Matching tier: {comparable_result.match_level}.",
            )
        with comparable_kpis[1]:
            st.metric(
                "Observed median",
                currency(float(summary_data[price_column].median())),
            )
        with comparable_kpis[2]:
            st.metric(
                "Observed middle 80%",
                (
                    f"{currency(float(price_quantiles.loc[0.10]), compact=True)}–"
                    f"{currency(float(price_quantiles.loc[0.90]), compact=True)}"
                ),
                help="10th to 90th percentile; this is not a prediction interval.",
            )
        with comparable_kpis[3]:
            st.metric(
                "Median price per sqm",
                f"{currency(float(summary_data[price_per_sqm_column].median()))}/sqm",
            )
        st.caption(
            f"Observed minimum to maximum: "
            f"{currency(float(summary_data[price_column].min()))} to "
            f"{currency(float(summary_data[price_column].max()))}. The displayed "
            "range describes past comparables and is not an estimate for a specific flat."
        )
        st.plotly_chart(
            comparable_distribution_figure(
                summary_data,
                price_column=price_column,
                price_per_sqm_column=price_per_sqm_column,
                price_basis=price_basis,
            ),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )

        interval_report = optional_report(MODEL_INTERVAL_METRICS_PATH)
        with st.expander("Model uncertainty and validation"):
            interval_columns = {"lower_price", "median_price", "upper_price"}
            valid_intervals = (
                comparable_sales.dropna(subset=list(interval_columns))
                if interval_columns.issubset(comparable_sales.columns)
                else comparable_sales.iloc[0:0]
            )
            if not valid_intervals.empty:
                interval_kpis = st.columns(3)
                with interval_kpis[0]:
                    st.metric(
                        "Median lower bound",
                        currency(float(valid_intervals["lower_price"].median())),
                    )
                with interval_kpis[1]:
                    st.metric(
                        "Median model estimate",
                        currency(float(valid_intervals["median_price"].median())),
                    )
                with interval_kpis[2]:
                    st.metric(
                        "Median upper bound",
                        currency(float(valid_intervals["upper_price"].median())),
                    )
                st.caption(
                    "These summarize row-level model intervals across the matching "
                    "transactions; they are not a valuation interval for the selected flat."
                )
            elif (
                interval_report is not None
                and not interval_report.empty
                and {
                    "empirical_coverage",
                    "target_coverage",
                    "median_interval_width",
                    "test_rows",
                }.issubset(interval_report.columns)
            ):
                valid_report = interval_report.dropna(
                    subset=[
                        "empirical_coverage",
                        "target_coverage",
                        "median_interval_width",
                        "test_rows",
                    ]
                ).copy()
                weights = pd.to_numeric(
                    valid_report["test_rows"], errors="coerce"
                ).fillna(0)
                if not valid_report.empty and float(weights.sum()) > 0:
                    rolling_coverage = float(
                        valid_report["empirical_coverage"].astype(float).mean()
                    )
                    rolling_width = float(
                        valid_report["median_interval_width"].astype(float).mean()
                    )
                    target_coverage = float(
                        valid_report["target_coverage"].astype(float).median()
                    )
                    latest_interval = valid_report.iloc[-1]
                    validation_kpis = st.columns(3)
                    with validation_kpis[0]:
                        st.metric(
                            "Rolling coverage",
                            percent(rolling_coverage),
                        )
                    with validation_kpis[1]:
                        st.metric(
                            "Target coverage",
                            percent(target_coverage),
                        )
                    with validation_kpis[2]:
                        st.metric(
                            "Typical interval width",
                            currency(rolling_width),
                        )
                    st.caption(
                        f"Mean across {len(valid_report):,} chronological folds and "
                        f"{int(weights.sum()):,} test transactions. Latest-fold coverage "
                        f"was {percent(float(latest_interval['empirical_coverage']))}. "
                        "These aggregate diagnostics do not provide a property-specific "
                        "valuation interval."
                    )
            else:
                st.caption(
                    "Calibrated prediction-interval diagnostics will appear here when "
                    "the optional model interval report is generated."
                )

        inflation_report = optional_report(INFLATION_REPORT_PATH)
        with st.expander("Inflation-adjusted context"):
            if real_price_columns_available:
                st.write(
                    "Use the price-basis control above to compare registered nominal "
                    "prices with values restated to a common CPI reference month."
                )
            elif inflation_report is not None and {
                "nominal_price_index",
                "real_price_index",
                "cpi_reference_month",
            }.issubset(inflation_report.columns):
                latest_inflation = inflation_report.iloc[-1]
                inflation_kpis = st.columns(2)
                with inflation_kpis[0]:
                    st.metric(
                        "Latest nominal index",
                        f"{float(latest_inflation['nominal_price_index']):.1f}",
                    )
                with inflation_kpis[1]:
                    st.metric(
                        "Latest real-price index",
                        f"{float(latest_inflation['real_price_index']):.1f}",
                    )
                st.caption(
                    "Whole-market context only. Upload or generate the enriched "
                    "transaction snapshot to switch individual comparables to real prices."
                )
            else:
                st.caption(
                    "Inflation-adjusted controls will appear here when CPI enrichment "
                    "has been generated."
                )

        mrt_exits = optional_report(MRT_EXITS_PATH)
        mrt_summary = optional_report(MRT_SUMMARY_PATH)
        mrt_metadata = (enrichment_metadata() or {}).get("mrt", {})
        mrt_coverage_pct = (
            mrt_metadata.get("coverage_pct")
            if isinstance(mrt_metadata, dict)
            else None
        )
        map_figure = comparable_map_figure(comparable_sales, mrt_exits)
        with st.expander(
            "MRT reference layer",
            expanded=map_figure is not None,
        ):
            if map_figure is not None:
                st.plotly_chart(
                    map_figure,
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )
                geocoded_sales = (
                    comparable_sales.dropna(subset=["latitude", "longitude"])
                    if {"latitude", "longitude"}.issubset(comparable_sales.columns)
                    else comparable_sales.iloc[0:0]
                )
                if geocoded_sales.empty:
                    coverage_text = (
                        f" Transaction-level MRT coverage is {float(mrt_coverage_pct):g}% "
                        "in this snapshot."
                        if isinstance(mrt_coverage_pct, (int, float))
                        else ""
                    )
                    st.caption(
                        "Reference layer only: the map shows official MRT exits, not "
                        "distances joined to individual resale transactions."
                        f"{coverage_text} Comparable blocks require validated OneMap "
                        "geocoding enrichment."
                    )
                elif "nearest_mrt_distance_m" in geocoded_sales.columns:
                    distance_values = geocoded_sales[
                        "nearest_mrt_distance_m"
                    ].dropna()
                    if not distance_values.empty:
                        st.metric(
                            "Median distance to nearest MRT exit",
                            f"{float(distance_values.median()):,.0f} m",
                        )
            else:
                st.caption(
                    "The MRT reference layer will appear when the optional official "
                    "station-exit report is available; comparable blocks additionally "
                    "require validated transaction coordinates."
                )

            if (
                mrt_summary is not None
                and not mrt_summary.empty
                and {
                    "mrt_distance_band",
                    "transaction_count",
                    "median_price_per_sqm",
                }.issubset(mrt_summary.columns)
            ):
                mrt_figure = go.Figure(
                    go.Bar(
                        x=mrt_summary["mrt_distance_band"],
                        y=mrt_summary["median_price_per_sqm"],
                        marker_color=TEAL,
                        customdata=mrt_summary[["transaction_count"]],
                        hovertemplate=(
                            "%{x}<br>Median: S$%{y:,.0f}/sqm<br>Transactions: "
                            "%{customdata[0]:,}<extra></extra>"
                        ),
                    )
                )
                apply_chart_style(mrt_figure, height=350, legend=False)
                mrt_figure.update_layout(
                    title=dict(
                        text="Price per sqm by MRT-distance band",
                        x=0.01,
                        font=dict(size=18),
                    )
                )
                mrt_figure.update_xaxes(title="Distance to nearest MRT exit")
                mrt_figure.update_yaxes(
                    title="Median price per sqm",
                    tickprefix="S$",
                    tickformat=",",
                )
                st.plotly_chart(
                    mrt_figure,
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )

        table_columns = [
            column
            for column in (
                "month",
                "block",
                "street_name",
                "address_key",
                "town",
                "flat_type",
                "storey_range",
                "floor_area_sqm",
                "remaining_lease_years",
                "resale_price",
                "price_per_sqm",
                "resale_price_real_sgd",
                "price_per_sqm_real_sgd",
                "cpi_all_items",
                "cpi_reference_month",
                "nearest_mrt_station",
                "nearest_mrt_exit_code",
                "nearest_mrt_distance_m",
            )
            if column in comparable_sales.columns
        ]
        export_sales = comparable_sales[table_columns].copy()
        export_sales["month"] = export_sales["month"].dt.strftime("%Y-%m")
        st.subheader("Matching transactions")
        st.dataframe(
            export_sales.head(250),
            hide_index=True,
            width="stretch",
            column_config={
                "resale_price": st.column_config.NumberColumn(format="dollar"),
                "price_per_sqm": st.column_config.NumberColumn(format="dollar"),
                "resale_price_real_sgd": st.column_config.NumberColumn(
                    "Real resale price",
                    format="dollar",
                ),
                "price_per_sqm_real_sgd": st.column_config.NumberColumn(
                    "Real price per sqm",
                    format="dollar",
                ),
                "nearest_mrt_distance_m": st.column_config.NumberColumn(
                    "Nearest MRT distance (m)",
                    format="localized",
                ),
            },
        )
        if len(export_sales) > 250:
            st.caption(
                "The table previews the 250 most recent matches; the download includes "
                f"all {len(export_sales):,}."
            )
        st.download_button(
            "Download comparable transactions (CSV)",
            data=export_sales.to_csv(index=False).encode("utf-8"),
            file_name="hdb_comparable_sales.csv",
            mime="text/csv",
            key="download_comparable_sales",
        )


with market_mix_tab:
    st.markdown(
        '<p class="section-kicker">The project’s differentiating result</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Did prices rise, or did the flats being sold change?")
    st.write(
        "The raw index follows the monthly median. The adjusted index estimates the "
        "market trend after accounting for town, flat type, model, floor area, "
        "remaining lease and storey."
    )
    if model_artifact_error or model_index is None or model_metrics is None:
        st.info(
            (model_artifact_error or "The model reports are unavailable.")
            + " Generate them to activate this view.",
            icon="🧭",
        )
        st.code(
            "python -m src.model_price --input data/processed/hdb_resale_clean.parquet",
            language="bash",
        )
    else:
        try:
            index_summary = model_metrics["index_summary"]
            holdout_metrics = model_metrics["holdout_metrics"]
            split = model_metrics["split"]
            methodology = model_metrics["methodology"]
            adjusted_metrics = holdout_metrics["hedonic_ridge"]
            baseline_metrics = holdout_metrics["training_median_baseline"]

            raw_growth = float(index_summary["raw_change_from_base_pct"])
            adjusted_growth = float(
                index_summary["quality_adjusted_change_from_base_pct"]
            )
            mix_gap = float(
                index_summary["raw_minus_adjusted_change_percentage_points"]
            )
            holdout_mae = float(adjusted_metrics["mae"])
            holdout_r2 = float(adjusted_metrics["r2"])
            baseline_mae = float(baseline_metrics["mae"])
            improvement = float(
                model_metrics["holdout_mae_improvement_vs_baseline_pct"]
            )

            model_kpis = st.columns(4)
            with model_kpis[0]:
                st.metric(
                    "Raw growth since Jan 2017",
                    f"{raw_growth:+.1f}%",
                )
            with model_kpis[1]:
                st.metric(
                    "Adjusted growth since Jan 2017",
                    f"{adjusted_growth:+.1f}%",
                )
            with model_kpis[2]:
                st.metric(
                    "Raw minus adjusted",
                    f"{mix_gap:.1f} pp",
                    help=(
                        "Difference between cumulative raw and quality-adjusted "
                        "changes; this is not a causal attribution."
                    ),
                )
            with model_kpis[3]:
                st.metric(
                    "Chronological holdout MAE",
                    currency(holdout_mae),
                    help="Typical absolute prediction error on the later holdout period.",
                )

            st.plotly_chart(
                market_mix_figure(model_index),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )

            interpretation_column, validation_column = st.columns(2, gap="large")
            with interpretation_column:
                st.subheader("What the comparison suggests")
                if abs(mix_gap) < 1:
                    gap_interpretation = (
                        "The gap is small, so cumulative raw and adjusted movement "
                        "is nearly identical under this specification. This does not "
                        "rule out mix effects in individual months or segments."
                    )
                else:
                    gap_interpretation = (
                        "The gap indicates that the recorded transaction mix changes "
                        "the cumulative headline movement under this specification."
                    )
                st.markdown(
                    f"The observed median-price index rose **{raw_growth:.1f}%**, "
                    f"while the quality-adjusted index rose **{adjusted_growth:.1f}%**. "
                    f"The difference is **{mix_gap:.1f} percentage points**. "
                    f"{gap_interpretation}"
                )
            with validation_column:
                st.subheader("Out-of-time validation")
                st.markdown(
                    f"The model was trained on **{split['training_start_month']} to "
                    f"{split['training_end_month']}** and tested on the later "
                    f"**{split['holdout_start_month']} to {split['holdout_end_month']}** "
                    f"period ({int(split['holdout_rows']):,} transactions). Holdout "
                    f"R² was **{holdout_r2:.3f}** and MAE was **{currency(holdout_mae)}**, "
                    f"a **{improvement:.1f}%** improvement over the training-median "
                    f"baseline MAE of {currency(baseline_mae)}."
                )

            provenance = model_metrics.get("input_provenance", {})
            coverage_note = (
                f"The incomplete {provenance['excluded_month']} source month was "
                "excluded from the index. "
                if provenance.get("provisional_latest_month_excluded")
                else ""
            )
            st.info(
                "This is a descriptive, whole-market transaction-mix adjustment—not "
                "a causal estimate or a formal property valuation. It uses precomputed "
                "artifacts and therefore does not change with the sidebar filters. "
                + coverage_note,
                icon="ℹ️",
            )
            with st.expander("Model controls and specification"):
                controls = methodology.get("quality_controls", [])
                st.write(
                    "Estimator:",
                    str(
                        methodology.get("estimator", "hedonic regression")
                    ).capitalize(),
                )
                st.write(
                    "Index estimator:",
                    str(methodology.get("index_estimator", "hedonic regression")),
                )
                st.write("Controls:", ", ".join(str(control) for control in controls))
                st.caption(str(methodology.get("interpretation", "")))
        except (KeyError, TypeError, ValueError) as exc:
            st.warning(
                f"The model reports use an unexpected schema ({exc}). Rebuild them "
                "with `python -m src.model_price --input "
                "data/processed/hdb_resale_clean.parquet`."
            )

with mix_tab:
    st.markdown(
        '<p class="section-kicker">Distribution, not just averages</p>',
        unsafe_allow_html=True,
    )
    st.write(
        "Medians describe the centre of the market; ranges and transaction mix show "
        "what sits underneath that headline."
    )
    distribution_column, flat_type_column = st.columns([1, 1.08], gap="large")
    with distribution_column:
        histogram = price_histogram(filtered, bins=38)
        st.plotly_chart(
            histogram_figure(histogram),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
    with flat_type_column:
        flat_summary = flat_type_distribution(filtered)
        st.plotly_chart(
            flat_type_distribution_figure(flat_summary),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )

    st.plotly_chart(
        transaction_mix_figure(filtered),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    st.markdown(
        '<p class="small-note"><strong>Why mix matters:</strong> a rise in the overall '
        "median can partly reflect a larger share of bigger flat types or transactions "
        "in higher-priced towns. Use the price-per-sqm and filtered views to compare "
        "more similar groups.</p>",
        unsafe_allow_html=True,
    )

with drivers_tab:
    st.markdown(
        '<p class="section-kicker">Descriptive price profiles</p>',
        unsafe_allow_html=True,
    )
    profile_controls = st.columns([1.4, 1])
    with profile_controls[0]:
        profile_metric = st.radio(
            "Profile metric",
            ["Median resale price", "Median price per sqm"],
            horizontal=True,
            key="profile_metric",
        )
    with profile_controls[1]:
        minimum_bin_size = int(
            st.number_input(
                "Minimum transactions per point",
                min_value=1,
                max_value=max(1, len(filtered)),
                value=min(50, len(filtered)),
                step=10 if len(filtered) >= 10 else 1,
                help="Hides thin bins whose medians are especially unstable.",
            )
        )

    area_profile = binned_price_profile(filtered, "floor_area_sqm", bin_width=10)
    area_profile = area_profile.loc[area_profile["transactions"] >= minimum_bin_size]
    lease_profile = binned_price_profile(filtered, "remaining_lease_years", bin_width=5)
    lease_profile = lease_profile.loc[lease_profile["transactions"] >= minimum_bin_size]
    storey_profile = exact_storey_profile(filtered)
    if not storey_profile.empty:
        storey_profile = storey_profile.loc[
            storey_profile["transactions"] >= minimum_bin_size
        ]

    area_column, lease_column = st.columns(2, gap="large")
    with area_column:
        if area_profile.empty:
            st.info("No floor-area bin reaches the selected sample threshold.")
        else:
            st.plotly_chart(
                profile_figure(
                    area_profile,
                    x_column="bin_mid",
                    x_title="Floor area (sqm; 10-sqm bins)",
                    metric=profile_metric,
                    title="Floor area profile",
                ),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
    with lease_column:
        if filtered["remaining_lease_years"].notna().sum() == 0:
            st.info(
                "The lease profile is unavailable because this snapshot has no "
                "usable remaining-lease values. Re-run the current cleaning pipeline."
            )
        elif lease_profile.empty:
            st.info("No remaining-lease bin reaches the selected sample threshold.")
        else:
            st.plotly_chart(
                profile_figure(
                    lease_profile,
                    x_column="bin_mid",
                    x_title="Remaining lease (years; 5-year bins)",
                    metric=profile_metric,
                    title="Remaining lease profile",
                ),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )

    if filtered["storey_mid"].notna().sum() == 0:
        st.info(
            "The storey profile is unavailable because this snapshot has no usable "
            "storey band values."
        )
    elif storey_profile.empty:
        st.info("No storey band reaches the selected sample threshold.")
    else:
        st.plotly_chart(
            profile_figure(
                storey_profile,
                x_column="storey_mid",
                x_title="Approximate storey (band midpoint)",
                metric=profile_metric,
                title="Storey profile",
            ),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
    st.markdown(
        '<p class="small-note"><strong>Read these as associations, not causal '
        "effects.</strong> Floor area, lease and storey also vary with town, flat type, "
        "model and transaction period. Point size reflects the number of transactions "
        "in each bin.</p>",
        unsafe_allow_html=True,
    )

with notes_tab:
    st.markdown(
        '<p class="section-kicker">Coverage and definitions</p>', unsafe_allow_html=True
    )
    st.subheader("Source coverage")
    coverage_display = full_coverage.copy()
    coverage_display["first_month"] = coverage_display["first_month"].dt.strftime(
        "%b %Y"
    )
    coverage_display["last_month"] = coverage_display["last_month"].dt.strftime("%b %Y")
    coverage_display["coverage"] = np.where(
        coverage_display["is_partial"], "Partial", "Complete"
    )
    coverage_display = coverage_display.rename(
        columns={
            "year": "Year",
            "months_observed": "Months observed",
            "first_month": "First month",
            "last_month": "Last month",
            "transactions": "Transactions",
            "coverage": "Coverage",
        }
    )[
        [
            "Year",
            "Months observed",
            "First month",
            "Last month",
            "Transactions",
            "Coverage",
        ]
    ]
    st.dataframe(
        coverage_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Transactions": st.column_config.NumberColumn(format="localized"),
        },
    )

    methodology_column, caution_column = st.columns(2, gap="large")
    with methodology_column:
        st.subheader("How the figures are calculated")
        st.markdown(
            """
            - **Median price** is the middle registered price after applying filters.
            - **Price per sqm** is transaction price divided by floor area, then summarised by its median.
            - **Million-dollar share** is the percentage of transactions at or above S$1,000,000.
            - **Storey** is represented by the midpoint of the source three-storey band.
            - **Remaining lease** is converted to months by the cleaning pipeline and shown in five-year bins.
            """
        )
    with caution_column:
        st.subheader("Interpretation cautions")
        st.markdown(
            """
            - A partial latest year should not be compared with a full year on transaction volume.
            - Small samples produce less stable medians; charts expose or filter by transaction count.
            - Composition changes can move aggregate prices even when like-for-like values are stable.
            - Profiles are descriptive and do not isolate the causal effect of a property characteristic.
            - Fully identical source rows remain because the public data has no transaction identifier.
            """
        )

    with st.expander("View and export the filtered town summary"):
        export_summary = town_summary(filtered).sort_values(
            "median_price", ascending=False
        )
        export_display = export_summary.copy()
        export_display["town"] = export_display["town"].map(readable_name)
        export_display = export_display.rename(
            columns={
                "town": "Town",
                "median_price": "Median resale price",
                "median_price_per_sqm": "Median price per sqm",
                "transactions": "Transactions",
                "million_dollar_share": "Million-dollar share",
            }
        )
        st.dataframe(
            export_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Median resale price": st.column_config.NumberColumn(format="dollar"),
                "Median price per sqm": st.column_config.NumberColumn(format="dollar"),
                "Transactions": st.column_config.NumberColumn(format="localized"),
                "Million-dollar share": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.download_button(
            "Download town summary (CSV)",
            data=export_summary.to_csv(index=False).encode("utf-8"),
            file_name="hdb_filtered_town_summary.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Source: Housing & Development Board resale flat prices via data.gov.sg. "
    "Registration month reflects when the transaction was registered, not necessarily "
    "the option-to-purchase date."
)
