"""Evidence-grounded Streamlit interface for the HDB Resale Decision Copilot.

The interface deliberately uses structured analysis modes. A user's optional
question can influence which records in an existing evidence bundle are
prioritised, but it cannot select tools, change filters, alter deterministic
wording, or introduce facts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.copilot import generate_answer
from src.copilot_tools import (
    build_comparable_sales_evidence,
    build_market_summary_evidence,
    build_model_diagnostics_evidence,
    build_town_comparison_evidence,
)
from src.dashboard_data import DashboardDataError, load_processed_data

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENRICHED_DATA_PATH = PROCESSED_DIR / "hdb_resale_enriched.parquet"
CLEAN_PARQUET_PATH = PROCESSED_DIR / "hdb_resale_clean.parquet"
CLEAN_CSV_PATH = PROCESSED_DIR / "hdb_resale_clean.csv"
SNAPSHOT_METADATA_PATH = PROCESSED_DIR / "snapshot_metadata.json"
ENRICHMENT_METADATA_PATH = PROJECT_ROOT / "reports" / "official_enrichment_metadata.json"
MODEL_METRICS_PATH = PROJECT_ROOT / "reports" / "price_model_metrics.json"
HDB_DATASET_URL = (
    "https://data.gov.sg/datasets/"
    "d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view"
)
OPEN_DATA_LICENCE_URL = "https://data.gov.sg/open-data-licence"
DEFAULT_REQUEST_LIMIT = 8
MARKET_MINIMUM_TRANSACTIONS = 20

MODE_MARKET = "Market brief"
MODE_COMPARABLES = "Comparable sales"
MODE_TOWN_COMPARISON = "Town comparison"
MODE_RELIABILITY = "Model reliability"

DEFAULT_QUESTIONS = {
    MODE_MARKET: (
        "What does the selected transaction evidence show, and what should I "
        "keep in mind when interpreting it?"
    ),
    MODE_COMPARABLES: (
        "What does this comparable-sales evidence show, and how broad were the "
        "matching criteria?"
    ),
    MODE_TOWN_COMPARISON: (
        "How do the two towns compare on observed prices and transaction volume, "
        "and what limits should I keep in mind?"
    ),
    MODE_RELIABILITY: (
        "Explain the model's tested reliability and limitations in plain language."
    ),
}


st.set_page_config(
    page_title="HDB Resale Decision Copilot",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f8fafc; }
      .block-container { max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
      [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 0.9rem 1rem; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
      }
      .copilot-hero {
        color: white; padding: 1.55rem 1.75rem; border-radius: 20px;
        background: linear-gradient(120deg, #0f766e 0%, #115e59 58%, #172033 100%);
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.16); margin-bottom: 1rem;
      }
      .copilot-hero h1 { color: white; margin: 0.12rem 0 0.38rem; font-size: 2.05rem; }
      .copilot-hero p { color: #ccfbf1; margin: 0; max-width: 850px; }
      .eyebrow { color: #99f6e4; font-size: 0.76rem; letter-spacing: 0.13em;
                 text-transform: uppercase; font-weight: 750; }
      .answer-card { background: #ffffff; border: 1px solid #dbeafe; border-left: 5px solid #0f766e;
                     border-radius: 14px; padding: 1rem 1.15rem; margin: 0.4rem 0 1rem; }
      .answer-card h3 { margin: 0 0 0.35rem; color: #172033; }
      .answer-card p { margin: 0; color: #334155; }
      .evidence-tag { color: #0f766e; font-size: 0.79rem; font-weight: 650; }
      .small-note { color: #64748b; font-size: 0.86rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for optional artifacts."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str:
    """Return a streaming digest for a data or report artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def _cached_file_sha256(
    path_string: str,
    modified_ns: int,
    file_size: int,
) -> str:
    """Cache a digest while invalidating it whenever the file changes."""
    del modified_ns, file_size
    return _file_sha256(Path(path_string))


def _actual_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_file_sha256(str(path), stat.st_mtime_ns, stat.st_size)


def _hash_matches(path: Path, expected_hash: object) -> bool:
    if not isinstance(expected_hash, str) or not expected_hash:
        return False
    try:
        return _actual_sha256(path) == expected_hash
    except OSError:
        return False


def _tracked_clean_snapshot_is_available(path: Path) -> bool:
    """Require clean CSV/Parquet bytes to match the tracked manifest."""
    if not path.is_file():
        return False
    snapshot = _read_json(SNAPSHOT_METADATA_PATH)
    if path == CLEAN_PARQUET_PATH:
        expected_file = snapshot.get("processed_parquet_file")
        expected_hash = snapshot.get("processed_parquet_sha256")
    elif path == CLEAN_CSV_PATH:
        expected_file = snapshot.get("processed_file")
        expected_hash = snapshot.get("processed_sha256")
    else:
        return False
    return (
        expected_file == path.name
        and _hash_matches(path, expected_hash)
    )


def _tracked_enriched_snapshot_is_available() -> bool:
    """Accept the enriched file only when its metadata matches the clean snapshot."""
    if not ENRICHED_DATA_PATH.is_file():
        return False
    snapshot = _read_json(SNAPSHOT_METADATA_PATH)
    enrichment = _read_json(ENRICHMENT_METADATA_PATH)
    enrichment_input = enrichment.get("input", {})
    enrichment_output = enrichment.get("output", {})
    if not isinstance(enrichment_input, Mapping) or not isinstance(
        enrichment_output, Mapping
    ):
        return False
    clean_hashes = {
        value
        for value in (
            snapshot.get("processed_sha256"),
            snapshot.get("processed_parquet_sha256"),
        )
        if isinstance(value, str) and value
    }
    return (
        enrichment_input.get("sha256") in clean_hashes
        and enrichment_output.get("file")
        == "data/processed/hdb_resale_enriched.parquet"
        and enrichment_input.get("row_count") == snapshot.get("row_count")
        and enrichment_output.get("row_count") == snapshot.get("row_count")
        and _hash_matches(ENRICHED_DATA_PATH, enrichment_output.get("sha256"))
    )


def _source_matches_model_snapshot(
    source_sha256: str,
    model_input_sha256: object,
) -> bool:
    """Accept only an exact or manifest-documented equivalent model input."""
    if source_sha256 == model_input_sha256:
        return True
    if not isinstance(model_input_sha256, str):
        return False
    snapshot = _read_json(SNAPSHOT_METADATA_PATH)
    canonical_hashes = {
        value
        for value in (
            snapshot.get("processed_sha256"),
            snapshot.get("processed_parquet_sha256"),
        )
        if isinstance(value, str) and value
    }
    if {source_sha256, model_input_sha256}.issubset(canonical_hashes):
        return True
    enrichment = _read_json(ENRICHMENT_METADATA_PATH)
    enrichment_input = enrichment.get("input", {})
    enrichment_output = enrichment.get("output", {})
    if not isinstance(enrichment_input, Mapping) or not isinstance(
        enrichment_output, Mapping
    ):
        return False
    return (
        source_sha256 == enrichment_output.get("sha256")
        and model_input_sha256 == enrichment_input.get("sha256")
    )


@st.cache_data(show_spinner=False)
def _load_disk_snapshot(
    path_string: str,
    modified_ns: int,
    file_size: int,
) -> pd.DataFrame:
    """Load with the shared dashboard validator and invalidate on file changes."""
    del modified_ns, file_size
    return load_processed_data(Path(path_string))


def load_copilot_snapshot() -> tuple[pd.DataFrame, Path]:
    """Load only checksum-verified tracked snapshots, in preferred order."""
    errors: list[str] = []
    for candidate in (ENRICHED_DATA_PATH, CLEAN_PARQUET_PATH, CLEAN_CSV_PATH):
        if not candidate.is_file():
            continue
        is_verified = (
            _tracked_enriched_snapshot_is_available()
            if candidate == ENRICHED_DATA_PATH
            else _tracked_clean_snapshot_is_available(candidate)
        )
        if not is_verified:
            errors.append(f"{candidate.name}: checksum or manifest mismatch")
            continue
        try:
            stat = candidate.stat()
            data = _load_disk_snapshot(
                str(candidate),
                stat.st_mtime_ns,
                stat.st_size,
            )
            expected_rows = _read_json(SNAPSHOT_METADATA_PATH).get("row_count")
            if isinstance(expected_rows, int) and len(data) != expected_rows:
                errors.append(f"{candidate.name}: row-count mismatch")
                continue
            return data, candidate
        except (DashboardDataError, OSError, ValueError) as exc:
            errors.append(f"{candidate.name}: {exc}")

    detail = "; ".join(errors) if errors else "no project snapshot was found"
    raise DashboardDataError(
        "The Copilot could not load an enriched, Parquet, or CSV snapshot ("
        + detail
        + ")."
    )


def _request_limit() -> int:
    """Return a defensive positive per-session generation limit."""
    try:
        configured = int(os.environ.get("HDB_COPILOT_MAX_REQUESTS", ""))
    except ValueError:
        return DEFAULT_REQUEST_LIMIT
    return configured if configured > 0 else DEFAULT_REQUEST_LIMIT


def _plain(value: Any) -> Any:
    """Convert Pydantic/dataclass answer objects into display-safe plain values."""
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _format_month(value: object) -> str:
    """Format a month-like value without implying greater date precision."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    return timestamp.strftime("%b %Y")


def _provenance(source_path: Path, data: pd.DataFrame) -> dict[str, Any]:
    """Build provenance passed unchanged to deterministic evidence builders."""
    snapshot = _read_json(SNAPSHOT_METADATA_PATH)
    try:
        relative_path = str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        relative_path = source_path.name
    return {
        "artifact": relative_path,
        "row_count": int(len(data)),
        "month_min": str(pd.Timestamp(data["month"].min()).to_period("M")),
        "month_max": str(pd.Timestamp(data["month"].max()).to_period("M")),
        "source_observed_at_utc": snapshot.get("source_modified_at_utc"),
        "sha256": _actual_sha256(source_path),
        "dataset_url": snapshot.get("dataset_url", HDB_DATASET_URL),
    }


def _build_evidence(
    mode: str,
    data: pd.DataFrame,
    source_path: Path,
    selections: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Route only the selected structured mode to its deterministic builder."""
    provenance = _provenance(source_path, data)
    if mode == MODE_MARKET:
        town = selections.get("town")
        flat_type = selections.get("flat_type")
        return build_market_summary_evidence(
            data,
            towns=None if town == "All towns" else [str(town)],
            flat_types=(
                None if flat_type == "All flat types" else [str(flat_type)]
            ),
            start_month=f"{int(selections['start_year'])}-01",
            end_month=f"{int(selections['end_year'])}-12",
            minimum_transactions=MARKET_MINIMUM_TRANSACTIONS,
            provenance=provenance,
        )
    if mode == MODE_COMPARABLES:
        return build_comparable_sales_evidence(
            data,
            town=str(selections["town"]),
            flat_type=str(selections["flat_type"]),
            floor_area_sqm=float(selections["floor_area_sqm"]),
            storey_mid=(
                float(selections["storey_mid"])
                if selections.get("use_storey")
                else None
            ),
            remaining_lease_years=(
                float(selections["remaining_lease_years"])
                if selections.get("use_lease")
                else None
            ),
            recent_months=int(selections["recent_months"]),
            minimum_transactions=int(selections["minimum_transactions"]),
            provenance=provenance,
        )
    if mode == MODE_TOWN_COMPARISON:
        flat_type = selections.get("flat_type")
        return build_town_comparison_evidence(
            data,
            towns=[str(selections["town_a"]), str(selections["town_b"])],
            flat_type=(
                None if flat_type == "All flat types" else str(flat_type)
            ),
            start_month=f"{int(selections['start_year'])}-01",
            end_month=f"{int(selections['end_year'])}-12",
            minimum_transactions=MARKET_MINIMUM_TRANSACTIONS,
            provenance=provenance,
        )
    if mode == MODE_RELIABILITY:
        metrics = _read_json(MODEL_METRICS_PATH)
        if not metrics:
            raise ValueError(
                "Model reliability artifacts are unavailable. Rebuild the price "
                "model reports before using this mode."
            )
        input_provenance = metrics.get("input_provenance", {})
        model_input_sha256 = (
            input_provenance.get("sha256")
            if isinstance(input_provenance, Mapping)
            else None
        )
        if not _source_matches_model_snapshot(
            str(provenance["sha256"]), model_input_sha256
        ):
            raise ValueError(
                "Model metrics were built from a different transaction snapshot."
            )
        model_provenance = {
            "artifact": "reports/price_model_metrics.json",
            "artifact_sha256": _actual_sha256(MODEL_METRICS_PATH),
            "transaction_snapshot_artifact": provenance["artifact"],
            "transaction_snapshot_sha256": provenance["sha256"],
        }
        return build_model_diagnostics_evidence(
            metrics,
            provenance=model_provenance,
        )
    raise ValueError(f"Unsupported analysis mode: {mode}")


def _render_sources(sources: list[Mapping[str, Any]]) -> None:
    """Show bundle sources plus the official dataset and licence links."""
    shown_urls: set[str] = set()
    for source in sources:
        url = source.get("url")
        title = str(source.get("title") or source.get("publisher") or "Source")
        publisher = str(source.get("publisher") or "").strip()
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            label = f"{title} — {publisher}" if publisher and publisher not in title else title
            st.markdown(f"- [{label}]({url})")
            shown_urls.add(url)
        else:
            st.markdown(f"- {title}" + (f" — {publisher}" if publisher else ""))
    if HDB_DATASET_URL not in shown_urls:
        st.markdown(f"- [HDB resale transaction records]({HDB_DATASET_URL})")
    if OPEN_DATA_LICENCE_URL not in shown_urls:
        st.markdown(f"- [Singapore Open Data Licence]({OPEN_DATA_LICENCE_URL})")


def _render_result(result: Mapping[str, Any]) -> None:
    """Render the answer and its inspectable evidence as separate layers."""
    answer = _plain(result.get("answer", {}))
    evidence = _plain(result.get("evidence", {}))
    if not isinstance(answer, Mapping) or not isinstance(evidence, Mapping):
        st.error("The Copilot returned an unreadable result.")
        return

    result_mode = str(result.get("mode") or "").strip()
    if result_mode:
        st.caption(f"Completed analysis: {result_mode}")

    generation_mode = str(answer.get("generation_mode", "deterministic_fallback"))
    if generation_mode == "deterministic_fallback":
        st.info(
            "Deterministic evidence summary shown. No generative model was needed "
            "for this answer."
        )
        fallback_reason = answer.get("fallback_reason")
        if fallback_reason:
            st.caption(f"Fallback reason: {fallback_reason}")
    else:
        st.success(
            "AI selected the evidence shown. All visible claim wording was "
            "rendered deterministically from verified records."
        )

    headline = str(answer.get("headline") or evidence.get("title") or "Result")
    summary = str(answer.get("summary") or "No summary was returned.")
    st.markdown(
        '<div class="answer-card"><h3>'
        + escape(headline)
        + "</h3><p>"
        + escape(summary)
        + "</p></div>",
        unsafe_allow_html=True,
    )

    points = answer.get("points", [])
    if isinstance(points, list) and points:
        st.subheader("Verified evidence selected for this question")
        for point in points:
            if not isinstance(point, Mapping):
                continue
            st.write(f"- {point.get('text', '')}")
            evidence_ids = point.get("evidence_ids", [])
            if isinstance(evidence_ids, list) and evidence_ids:
                st.caption("Evidence: " + ", ".join(map(str, evidence_ids)))

    facts = evidence.get("facts", [])
    if isinstance(facts, list) and facts:
        st.subheader("Verified figures")
        columns = st.columns(min(4, len(facts)))
        for index, fact in enumerate(facts[:8]):
            if not isinstance(fact, Mapping):
                continue
            display_value = fact.get("display_value", fact.get("raw_value", "—"))
            columns[index % len(columns)].metric(
                str(fact.get("label", fact.get("id", "Evidence"))),
                str(display_value),
            )

    evidence_rows = evidence.get("evidence_rows", [])
    st.subheader("Evidence table")
    if isinstance(evidence_rows, list) and evidence_rows:
        table = pd.DataFrame(evidence_rows)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            f"{len(table):,} evidence row{'s' if len(table) != 1 else ''} shown. "
            "These rows—not the optional question—anchor the answer."
        )
    else:
        st.warning("No transaction-level evidence rows were available for this selection.")

    limitations: list[str] = []
    for source in (answer.get("limitations", []), evidence.get("caveats", [])):
        if isinstance(source, list):
            limitations.extend(str(item) for item in source if str(item).strip())
    limitations = list(dict.fromkeys(limitations))
    if limitations:
        with st.expander("Caveats and responsible-use notes", expanded=True):
            for limitation in limitations:
                st.write(f"- {limitation}")

    st.subheader("Sources")
    raw_sources = evidence.get("sources", [])
    sources = [item for item in raw_sources if isinstance(item, Mapping)] if isinstance(raw_sources, list) else []
    _render_sources(sources)

    follow_ups = answer.get("follow_ups", [])
    if isinstance(follow_ups, list) and follow_ups:
        with st.expander("Questions you could explore next"):
            for follow_up in follow_ups:
                st.write(f"- {follow_up}")


st.markdown(
    """
    <div class="copilot-hero">
      <div class="eyebrow">Evidence-grounded decision support</div>
      <h1>Singapore HDB Resale Decision Copilot</h1>
      <p>Explore observed market activity, comparable transactions and model
      reliability. Deterministic calculations produce the evidence first; AI can
      select relevant records, while project code renders every visible claim.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    transactions, selected_path = load_copilot_snapshot()
except DashboardDataError as exc:
    st.error(str(exc), icon="📂")
    st.code("python -m src.download_data\npython -m src.clean_data", language="bash")
    st.stop()

snapshot_metadata = _read_json(SNAPSHOT_METADATA_PATH)
month_min = pd.Timestamp(transactions["month"].min())
month_max = pd.Timestamp(transactions["month"].max())
observed_at = snapshot_metadata.get("source_modified_at_utc")
observed_label = "unverified"
if observed_at:
    try:
        observed_timestamp = pd.Timestamp(observed_at)
        if observed_timestamp.tzinfo is None:
            observed_timestamp = observed_timestamp.tz_localize("UTC")
        observed_label = observed_timestamp.tz_convert("Asia/Singapore").strftime(
            "%d %b %Y SGT"
        )
    except (TypeError, ValueError):
        pass

coverage_columns = st.columns(4)
coverage_columns[0].metric("Transactions", f"{len(transactions):,}")
coverage_columns[1].metric("Coverage starts", _format_month(month_min))
coverage_columns[2].metric("Coverage ends", _format_month(month_max))
coverage_columns[3].metric("Source observed", observed_label)

latest_month_rows = int(
    transactions["month"].dt.to_period("M").eq(month_max.to_period("M")).sum()
)
st.warning(
    f"The latest observed month is {_format_month(month_max)} with "
    f"{latest_month_rows:,} recorded transactions and may be incomplete. "
    "Results describe this snapshot; they are not a formal valuation, forecast, "
    "financial recommendation or guarantee."
)

with st.sidebar:
    st.header("How this Copilot works")
    st.markdown(
        "1. You choose a structured analysis.\n"
        "2. Project code calculates a fixed evidence bundle.\n"
        "3. AI may select relevant evidence IDs; project code renders every "
        "visible claim."
    )
    st.caption(f"Data source: {selected_path.name}")
    st.markdown(f"[Official HDB dataset]({HDB_DATASET_URL})")
    st.markdown(f"[Open data licence]({OPEN_DATA_LICENCE_URL})")

    if os.environ.get("OPENAI_API_KEY", "").strip():
        st.success("AI evidence selection is available.")
    else:
        st.info(
            "No API key detected. The app remains fully usable with a "
            "deterministic evidence summary and makes no AI request."
        )

    request_limit = _request_limit()
    request_count = int(st.session_state.get("copilot_request_count", 0))
    st.caption(f"Session requests: {request_count}/{request_limit}")

st.subheader("Choose an analysis")
mode = st.radio(
    "Analysis mode",
    [MODE_MARKET, MODE_COMPARABLES, MODE_TOWN_COMPARISON, MODE_RELIABILITY],
    horizontal=True,
)
previous_mode = st.session_state.get("copilot_active_mode")
if isinstance(previous_mode, str) and previous_mode != mode:
    st.session_state.pop("copilot_result", None)
st.session_state["copilot_active_mode"] = mode

towns = sorted(str(value) for value in transactions["town"].dropna().unique())
flat_types = sorted(
    str(value) for value in transactions["flat_type"].dropna().unique()
)
years = sorted(int(value) for value in transactions["year"].dropna().unique())

with st.form("copilot_request_form", clear_on_submit=False):
    selections: dict[str, Any] = {}
    if mode == MODE_MARKET:
        st.caption(
            "Market briefs require at least 20 matching transactions before "
            "presenting distribution statistics."
        )
        first, second = st.columns(2)
        selections["town"] = first.selectbox("Town", ["All towns", *towns])
        selections["flat_type"] = second.selectbox(
            "Flat type", ["All flat types", *flat_types]
        )
        third, fourth = st.columns(2)
        default_start_index = max(0, len(years) - 5)
        selections["start_year"] = third.selectbox(
            "Start year", years, index=default_start_index
        )
        selections["end_year"] = fourth.selectbox(
            "End year", years, index=len(years) - 1
        )
    elif mode == MODE_COMPARABLES:
        first, second = st.columns(2)
        default_town = towns.index("TAMPINES") if "TAMPINES" in towns else 0
        default_type = flat_types.index("4 ROOM") if "4 ROOM" in flat_types else 0
        selections["town"] = first.selectbox("Town", towns, index=default_town)
        selections["flat_type"] = second.selectbox(
            "Flat type", flat_types, index=default_type
        )
        third, fourth = st.columns(2)
        selections["floor_area_sqm"] = third.number_input(
            "Floor area (sqm)", min_value=20.0, max_value=300.0, value=93.0, step=1.0
        )
        selections["recent_months"] = fourth.selectbox(
            "Recent period", [12, 24, 36, 60], index=1, format_func=lambda value: f"{value} months"
        )
        fifth, sixth = st.columns(2)
        selections["use_storey"] = fifth.checkbox(
            "Match storey", value=True
        )
        selections["storey_mid"] = fifth.number_input(
            "Storey midpoint", min_value=1.0, max_value=60.0, value=8.0, step=1.0,
            disabled=not selections["use_storey"],
        )
        selections["use_lease"] = sixth.checkbox(
            "Match remaining lease", value=True
        )
        selections["remaining_lease_years"] = sixth.number_input(
            "Remaining lease (years)", min_value=1.0, max_value=99.0, value=73.0, step=1.0,
            disabled=not selections["use_lease"],
        )
        selections["minimum_transactions"] = st.slider(
            "Preferred minimum matches", min_value=5, max_value=50, value=20, step=5
        )
    elif mode == MODE_TOWN_COMPARISON:
        st.caption(
            "Each town must have at least 20 matching transactions for a direct "
            "median-price comparison."
        )
        first, second = st.columns(2)
        default_a = towns.index("TAMPINES") if "TAMPINES" in towns else 0
        selections["town_a"] = first.selectbox(
            "First town", towns, index=default_a
        )
        comparison_towns = [
            town for town in towns if town != selections["town_a"]
        ]
        preferred_b = (
            comparison_towns.index("BEDOK")
            if "BEDOK" in comparison_towns
            else 0
        )
        selections["town_b"] = second.selectbox(
            "Second town", comparison_towns, index=preferred_b
        )
        third, fourth, fifth = st.columns(3)
        selections["flat_type"] = third.selectbox(
            "Flat type", ["All flat types", *flat_types]
        )
        default_start_index = max(0, len(years) - 5)
        selections["start_year"] = fourth.selectbox(
            "Start year", years, index=default_start_index
        )
        selections["end_year"] = fifth.selectbox(
            "End year", years, index=len(years) - 1
        )
    else:
        st.info(
            "This mode explains the tracked out-of-time holdout and rolling-backtest "
            "artifacts. It does not generate a price for an individual flat."
        )

    optional_question = st.text_area(
        "Optional question",
        placeholder=DEFAULT_QUESTIONS[mode],
        help=(
            "This can change which verified records are prioritised, but it cannot "
            "change the selected mode, filters, calculations or displayed wording."
        ),
        max_chars=800,
    )
    current_count = int(st.session_state.get("copilot_request_count", 0))
    submitted = st.form_submit_button(
        "Build evidence-grounded answer",
        type="primary",
        disabled=current_count >= request_limit,
        width="stretch",
    )

if submitted:
    if mode in {MODE_MARKET, MODE_TOWN_COMPARISON} and int(
        selections["start_year"]
    ) > int(
        selections["end_year"]
    ):
        st.error("Start year must be earlier than or equal to end year.")
    else:
        st.session_state["copilot_request_count"] = (
            int(st.session_state.get("copilot_request_count", 0)) + 1
        )
        try:
            with st.spinner("Building and checking the evidence..."):
                evidence_bundle = _build_evidence(
                    mode,
                    transactions,
                    selected_path,
                    selections,
                )
                question = optional_question.strip() or DEFAULT_QUESTIONS[mode]
                answer = generate_answer(question, evidence_bundle)
            st.session_state["copilot_result"] = {
                "mode": mode,
                "question": question,
                "evidence": _plain(evidence_bundle),
                "answer": _plain(answer),
            }
        except (DashboardDataError, KeyError, OSError, TypeError, ValueError) as exc:
            st.error(f"The selected evidence could not be built: {exc}")

if int(st.session_state.get("copilot_request_count", 0)) >= request_limit:
    st.error(
        "This session has reached its request limit. Start a new session to make "
        "another request, or raise HDB_COPILOT_MAX_REQUESTS for a private deployment."
    )

result = st.session_state.get("copilot_result")
if isinstance(result, Mapping):
    st.divider()
    _render_result(result)
else:
    st.caption(
        "No model call occurs while you change controls. Submit the form when you "
        "are ready to build the evidence and answer."
    )
