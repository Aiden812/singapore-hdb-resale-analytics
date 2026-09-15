"""Deterministic, citation-ready evidence builders for the HDB copilot.

The language model must never calculate market figures itself.  This module
turns validated transaction data and tracked model metrics into compact,
JSON-serialisable evidence packets.  Every material fact has a stable ID and
points to either a supporting evidence row or a named source.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.dashboard_data import (
    DashboardDataError,
    annual_trend,
    coverage_by_year,
    filter_transactions,
    find_comparable_sales,
    monthly_trend,
    prepare_dashboard_data,
    town_summary,
)

SCHEMA_VERSION = "1.0"
COMPARABLE_LIMIT = 5

HDB_SOURCE = {
    "id": "source.hdb_resale_transactions",
    "title": "HDB resale transaction registrations",
    "publisher": "Housing & Development Board via data.gov.sg",
    "url": (
        "https://data.gov.sg/datasets/"
        "d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view"
    ),
}
MODEL_METRICS_SOURCE = {
    "id": "source.price_model_metrics",
    "title": "Tracked price-model metrics",
    "publisher": "Singapore HDB Resale Analytics project",
    "path": "reports/price_model_metrics.json",
    "url": None,
}
MODEL_CARD_SOURCE = {
    "id": "source.model_card",
    "title": "Quality-adjusted price model card",
    "publisher": "Singapore HDB Resale Analytics project",
    "path": "docs/model_card.md",
    "url": None,
}


class EvidenceInputError(ValueError):
    """Raised when a copilot evidence request is malformed."""


def _json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values to strict JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise EvidenceInputError("data must be a pandas DataFrame")
    if data.empty:
        return data.copy()
    try:
        return prepare_dashboard_data(data)
    except DashboardDataError as exc:
        raise EvidenceInputError(str(exc)) from exc


def _normalise_labels(
    values: Sequence[str] | str | None,
    *,
    field: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, Sequence):
        candidates = list(values)
    else:
        raise EvidenceInputError(f"{field} must be text or a sequence of text")
    normalised: list[str] = []
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            raise EvidenceInputError(f"{field} cannot contain blank values")
        normalised.append(value.strip().upper())
    return tuple(sorted(set(normalised)))


def _normalise_required_label(value: str, *, field: str) -> str:
    labels = _normalise_labels(value, field=field)
    if not labels:
        raise EvidenceInputError(f"{field} is required")
    return labels[0]


def _normalise_month(value: object | None, *, field: str) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.number)):
        raise EvidenceInputError(f"{field} must be a calendar month")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceInputError(f"{field} must be a valid calendar month") from exc
    if pd.isna(timestamp):
        raise EvidenceInputError(f"{field} must be a valid calendar month")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.to_period("M").to_timestamp()


def _month_bounds(
    start_month: object | None,
    end_month: object | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = _normalise_month(start_month, field="start_month")
    end = _normalise_month(end_month, field="end_month")
    if start is not None and end is not None and start > end:
        raise EvidenceInputError("start_month cannot be after end_month")
    return start, end


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise EvidenceInputError(f"{field} must be a positive integer")
    result = int(value)
    if result < 1:
        raise EvidenceInputError(f"{field} must be a positive integer")
    return result


def _finite_number(
    value: object,
    *,
    field: str,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise EvidenceInputError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceInputError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise EvidenceInputError(f"{field} must be a finite number")
    return number


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".")
    return slug or "unknown"


def _month_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m")


def _format_count(value: int | float) -> str:
    return f"{int(value):,}"


def _format_sgd(value: int | float) -> str:
    return f"S${float(value):,.0f}"


def _format_sgd_per_sqm(value: int | float) -> str:
    return f"S${float(value):,.0f}/sqm"


def _format_percent_from_share(value: int | float) -> str:
    return f"{float(value) * 100:.1f}%"


def _format_percent(value: int | float) -> str:
    return f"{float(value):.1f}%"


def _fact(
    fact_id: str,
    label: str,
    raw_value: object,
    display_value: str,
    unit: str | None,
    *,
    evidence_ids: Sequence[str] = (),
    source_ids: Sequence[str] = ("source.hdb_resale_transactions",),
) -> dict[str, object]:
    return {
        "id": fact_id,
        "label": label,
        "raw_value": _json_safe(raw_value),
        "display_value": display_value,
        "unit": unit,
        "evidence_ids": list(evidence_ids),
        "source_ids": list(source_ids),
    }


def _snapshot_context(
    data: pd.DataFrame,
    provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    if provenance is not None and not isinstance(provenance, Mapping):
        raise EvidenceInputError("provenance must be a mapping")
    if data.empty:
        coverage: dict[str, object] = {
            "start_month": None,
            "end_month": None,
            "months_observed": 0,
            "partial_years": [],
        }
    else:
        year_coverage = coverage_by_year(data)
        coverage = {
            "start_month": _month_text(data["month"].min()),
            "end_month": _month_text(data["month"].max()),
            "months_observed": int(data["month"].nunique()),
            "partial_years": [
                int(year)
                for year in year_coverage.loc[
                    year_coverage["is_partial"], "year"
                ].tolist()
            ],
        }
    snapshot: dict[str, object] = {
        "coverage": coverage,
        "rows": int(len(data)),
    }
    if provenance is not None:
        snapshot["provenance"] = _json_safe(provenance)
    return snapshot


def _base_packet(
    *,
    analysis_type: str,
    title: str,
    filters: Mapping[str, object],
    snapshot: Mapping[str, object],
    sources: Sequence[Mapping[str, object]] = (HDB_SOURCE,),
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": analysis_type,
        "title": title,
        "status": "ok",
        "filters": _json_safe(filters),
        "snapshot": _json_safe(snapshot),
        "facts": [],
        "evidence_rows": [],
        "sources": [_json_safe(source) for source in sources],
        "caveats": [],
    }


def _apply_selection(
    data: pd.DataFrame,
    *,
    towns: tuple[str, ...] | None,
    flat_types: tuple[str, ...] | None,
    start_month: pd.Timestamp | None,
    end_month: pd.Timestamp | None,
) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    selected = filter_transactions(data, towns=towns, flat_types=flat_types)
    if start_month is not None:
        selected = selected.loc[selected["month"].ge(start_month)]
    if end_month is not None:
        selected = selected.loc[selected["month"].le(end_month)]
    return selected.copy()


def _annual_evidence_rows(data: pd.DataFrame, *, prefix: str) -> list[dict[str, object]]:
    if data.empty:
        return []
    rows: list[dict[str, object]] = []
    for record in annual_trend(data).to_dict(orient="records"):
        year = int(record["year"])
        rows.append(
            {
                "id": f"row.{prefix}.year.{year}",
                "year": year,
                "transactions": int(record["transactions"]),
                "months_observed": int(record["months_observed"]),
                "is_partial": bool(record["is_partial"]),
                "median_resale_price_sgd": float(record["median_price"]),
                "price_q25_sgd": float(record["price_q25"]),
                "price_q75_sgd": float(record["price_q75"]),
                "median_price_per_sqm_sgd": float(
                    record["median_price_per_sqm"]
                ),
                "million_dollar_share": float(record["million_dollar_share"]),
            }
        )
    return rows


def build_market_summary_evidence(
    data: pd.DataFrame,
    *,
    towns: Sequence[str] | str | None = None,
    flat_types: Sequence[str] | str | None = None,
    start_month: object | None = None,
    end_month: object | None = None,
    minimum_transactions: int = 3,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build grounded market-level price, volume and period evidence."""
    prepared = _prepare_input(data)
    towns_normalised = _normalise_labels(towns, field="towns")
    flat_types_normalised = _normalise_labels(flat_types, field="flat_types")
    start, end = _month_bounds(start_month, end_month)
    minimum = _positive_integer(
        minimum_transactions, field="minimum_transactions"
    )
    selected = _apply_selection(
        prepared,
        towns=towns_normalised,
        flat_types=flat_types_normalised,
        start_month=start,
        end_month=end,
    )
    filters = {
        "towns": list(towns_normalised) if towns_normalised is not None else [],
        "flat_types": (
            list(flat_types_normalised) if flat_types_normalised is not None else []
        ),
        "start_month": _month_text(start),
        "end_month": _month_text(end),
        "minimum_transactions": minimum,
    }
    packet = _base_packet(
        analysis_type="market_summary",
        title="HDB resale market summary",
        filters=filters,
        snapshot=_snapshot_context(prepared, provenance),
    )
    annual_evidence_rows = _annual_evidence_rows(selected, prefix="market")
    annual_evidence_ids = [str(row["id"]) for row in annual_evidence_rows]
    packet["evidence_rows"] = annual_evidence_rows
    facts: list[dict[str, object]] = [
        _fact(
            "fact.market.transaction_count",
            "Matching transactions",
            len(selected),
            _format_count(len(selected)),
            "transactions",
            evidence_ids=annual_evidence_ids,
        )
    ]
    packet["facts"] = facts

    if len(selected) < minimum:
        packet["status"] = "insufficient_evidence"
        packet["caveats"] = [
            f"The filters matched {len(selected):,} transactions; at least "
            f"{minimum:,} are required for a market summary."
        ]
        return packet

    trend = monthly_trend(selected)
    median_price = float(selected["resale_price"].median())
    median_price_per_sqm = float(selected["price_per_sqm"].median())
    price_q25 = float(selected["resale_price"].quantile(0.25))
    price_q75 = float(selected["resale_price"].quantile(0.75))
    million_share = float(selected["is_million_dollar"].mean())
    first_month = pd.Timestamp(trend.iloc[0]["month"])
    latest_month = pd.Timestamp(trend.iloc[-1]["month"])
    first_month_median = float(trend.iloc[0]["median_price"])
    latest_month_median = float(trend.iloc[-1]["median_price"])
    change_pct = (latest_month_median / first_month_median - 1) * 100
    selection_evidence_id = "row.market.selection"
    first_evidence_id = "row.market.endpoint.first"
    latest_evidence_id = "row.market.endpoint.latest"
    selection_evidence_row = {
        "id": selection_evidence_id,
        "transactions": int(len(selected)),
        "median_resale_price_sgd": median_price,
        "median_price_per_sqm_sgd": median_price_per_sqm,
        "price_q25_sgd": price_q25,
        "price_q75_sgd": price_q75,
        "million_dollar_share": million_share,
        "first_observed_month": _month_text(first_month),
        "latest_observed_month": _month_text(latest_month),
        "first_to_latest_monthly_median_change_pct": change_pct,
    }

    def endpoint_row(record: pd.Series, row_id: str, position: str) -> dict[str, object]:
        return {
            "id": row_id,
            "position": position,
            "month": _month_text(record["month"]),
            "transactions": int(record["transactions"]),
            "median_resale_price_sgd": float(record["median_price"]),
            "price_q25_sgd": float(record["price_q25"]),
            "price_q75_sgd": float(record["price_q75"]),
            "median_price_per_sqm_sgd": float(record["median_price_per_sqm"]),
            "million_dollar_share": float(record["million_dollar_share"]),
        }

    packet["evidence_rows"] = [
        selection_evidence_row,
        endpoint_row(trend.iloc[0], first_evidence_id, "first"),
        endpoint_row(trend.iloc[-1], latest_evidence_id, "latest"),
        *annual_evidence_rows,
    ]
    facts[0]["evidence_ids"] = [selection_evidence_id]
    facts.extend(
        [
            _fact(
                "fact.market.median_resale_price",
                "Median resale price",
                median_price,
                _format_sgd(median_price),
                "SGD",
                evidence_ids=[selection_evidence_id],
            ),
            _fact(
                "fact.market.median_price_per_sqm",
                "Median price per sqm",
                median_price_per_sqm,
                _format_sgd_per_sqm(median_price_per_sqm),
                "SGD_per_sqm",
                evidence_ids=[selection_evidence_id],
            ),
            _fact(
                "fact.market.price_q25",
                "25th-percentile resale price",
                price_q25,
                _format_sgd(price_q25),
                "SGD",
                evidence_ids=[selection_evidence_id],
            ),
            _fact(
                "fact.market.price_q75",
                "75th-percentile resale price",
                price_q75,
                _format_sgd(price_q75),
                "SGD",
                evidence_ids=[selection_evidence_id],
            ),
            _fact(
                "fact.market.million_dollar_share",
                "Million-dollar transaction share",
                million_share,
                _format_percent_from_share(million_share),
                "proportion",
                evidence_ids=[selection_evidence_id],
            ),
            _fact(
                "fact.market.first_observed_month",
                "First observed month",
                _month_text(first_month),
                str(_month_text(first_month)),
                "calendar_month",
                evidence_ids=[first_evidence_id],
            ),
            _fact(
                "fact.market.latest_observed_month",
                "Latest observed month",
                _month_text(latest_month),
                str(_month_text(latest_month)),
                "calendar_month",
                evidence_ids=[latest_evidence_id],
            ),
            _fact(
                "fact.market.first_month_median_price",
                "First observed monthly median resale price",
                first_month_median,
                _format_sgd(first_month_median),
                "SGD",
                evidence_ids=[first_evidence_id],
            ),
            _fact(
                "fact.market.latest_month_median_price",
                "Latest observed monthly median resale price",
                latest_month_median,
                _format_sgd(latest_month_median),
                "SGD",
                evidence_ids=[latest_evidence_id],
            ),
            _fact(
                "fact.market.first_to_latest_monthly_median_change",
                "First-to-latest monthly median price change",
                change_pct,
                _format_percent(change_pct),
                "percent",
                evidence_ids=[first_evidence_id, latest_evidence_id],
            ),
        ]
    )
    caveats = [
        "Registered transaction prices describe completed sales; they are not "
        "asking prices, forecasts or formal valuations.",
        "Changes in the mix of flats sold can affect unadjusted medians.",
    ]
    partial_years = [
        int(row["year"])
        for row in annual_evidence_rows
        if bool(row["is_partial"])
    ]
    if partial_years:
        caveats.append(
            "Partial-year evidence is present for: "
            + ", ".join(str(year) for year in partial_years)
            + "."
        )
    packet["caveats"] = caveats
    return packet


def _comparable_fingerprint(row: pd.Series) -> str:
    fields = (
        _month_text(row.get("month")) or "",
        str(row.get("town", "")),
        str(row.get("flat_type", "")),
        str(row.get("block", "")),
        str(row.get("street_name", "")),
        str(_json_safe(row.get("floor_area_sqm"))),
        str(_json_safe(row.get("storey_mid"))),
        str(_json_safe(row.get("remaining_lease_years"))),
        str(_json_safe(row.get("resale_price"))),
    )
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:12]


def _rank_comparables(
    transactions: pd.DataFrame,
    *,
    latest_month: pd.Timestamp,
    floor_area_sqm: float,
    storey_mid: float | None,
    remaining_lease_years: float | None,
    floor_area_tolerance: float,
    storey_tolerance: float | None,
    lease_tolerance: float | None,
) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()
    ranked = transactions.copy()
    ranked["_area_delta"] = (
        pd.to_numeric(ranked["floor_area_sqm"], errors="coerce") - floor_area_sqm
    ).abs()
    area_component = ranked["_area_delta"].fillna(float("inf")) / max(
        floor_area_tolerance, 1.0
    )

    if storey_mid is None:
        ranked["_storey_delta"] = np.nan
        storey_component = pd.Series(0.0, index=ranked.index)
    else:
        ranked["_storey_delta"] = (
            pd.to_numeric(ranked["storey_mid"], errors="coerce") - storey_mid
        ).abs()
        storey_scale = storey_tolerance or 6.0
        storey_component = ranked["_storey_delta"].fillna(storey_scale * 2) / max(
            storey_scale, 1.0
        )

    if remaining_lease_years is None:
        ranked["_lease_delta"] = np.nan
        lease_component = pd.Series(0.0, index=ranked.index)
    else:
        ranked["_lease_delta"] = (
            pd.to_numeric(ranked["remaining_lease_years"], errors="coerce")
            - remaining_lease_years
        ).abs()
        lease_scale = lease_tolerance or 10.0
        lease_component = ranked["_lease_delta"].fillna(lease_scale * 2) / max(
            lease_scale, 1.0
        )

    ranked["_distance_score"] = np.sqrt(
        area_component.pow(2)
        + storey_component.pow(2)
        + lease_component.pow(2)
    )
    ranked["_month_ordinal"] = ranked["month"].map(
        lambda value: pd.Timestamp(value).to_period("M").ordinal
    )
    ranked["_age_months"] = latest_month.to_period("M").ordinal - ranked[
        "_month_ordinal"
    ]
    ranked["_fingerprint"] = ranked.apply(_comparable_fingerprint, axis=1)
    ranked = ranked.sort_values(
        [
            "_distance_score",
            "_age_months",
            "_area_delta",
            "_storey_delta",
            "_lease_delta",
            "_fingerprint",
            "resale_price",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    ranked["_duplicate_number"] = ranked.groupby(
        "_fingerprint", sort=False
    ).cumcount() + 1
    ranked["_duplicate_count"] = ranked.groupby("_fingerprint", sort=False)[
        "_fingerprint"
    ].transform("size")
    return ranked.head(COMPARABLE_LIMIT).copy()


def _comparable_evidence_rows(ranked: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, (_, record) in enumerate(ranked.iterrows(), start=1):
        suffix = (
            f".{int(record['_duplicate_number'])}"
            if int(record["_duplicate_count"]) > 1
            else ""
        )
        row_id = f"row.comparable.{record['_fingerprint']}{suffix}"
        floor_area = float(record["floor_area_sqm"])
        resale_price = float(record["resale_price"])
        rows.append(
            {
                "id": row_id,
                "rank": rank,
                "month": _month_text(record["month"]),
                "town": str(record["town"]),
                "flat_type": str(record["flat_type"]),
                "block": (
                    None
                    if pd.isna(record.get("block"))
                    else str(record.get("block"))
                ),
                "street_name": (
                    None
                    if pd.isna(record.get("street_name"))
                    else str(record.get("street_name"))
                ),
                "floor_area_sqm": floor_area,
                "storey_mid": _json_safe(record.get("storey_mid")),
                "remaining_lease_years": _json_safe(
                    record.get("remaining_lease_years")
                ),
                "resale_price_sgd": resale_price,
                "price_per_sqm_sgd": resale_price / floor_area,
                "floor_area_delta_sqm": _json_safe(record["_area_delta"]),
                "storey_delta": _json_safe(record["_storey_delta"]),
                "remaining_lease_delta_years": _json_safe(
                    record["_lease_delta"]
                ),
                "age_months_from_snapshot_latest": int(record["_age_months"]),
                "distance_score": round(float(record["_distance_score"]), 6),
            }
        )
    return rows


def build_comparable_sales_evidence(
    data: pd.DataFrame,
    *,
    town: str,
    flat_type: str,
    floor_area_sqm: object,
    storey_mid: object | None = None,
    remaining_lease_years: object | None = None,
    recent_months: int = 24,
    minimum_transactions: int = 5,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the five closest deterministic comparable-sale records."""
    prepared = _prepare_input(data)
    town_normalised = _normalise_required_label(town, field="town")
    flat_type_normalised = _normalise_required_label(flat_type, field="flat_type")
    area = _finite_number(floor_area_sqm, field="floor_area_sqm")
    storey = _finite_number(storey_mid, field="storey_mid", allow_none=True)
    lease = _finite_number(
        remaining_lease_years,
        field="remaining_lease_years",
        allow_none=True,
    )
    assert area is not None
    if area <= 0:
        raise EvidenceInputError("floor_area_sqm must be positive")
    if storey is not None and storey <= 0:
        raise EvidenceInputError("storey_mid must be positive")
    if lease is not None and not 0 < lease <= 99:
        raise EvidenceInputError("remaining_lease_years must be between 0 and 99")
    recent = _positive_integer(recent_months, field="recent_months")
    minimum = _positive_integer(
        minimum_transactions, field="minimum_transactions"
    )
    filters = {
        "town": town_normalised,
        "flat_type": flat_type_normalised,
        "floor_area_sqm": area,
        "storey_mid": storey,
        "remaining_lease_years": lease,
        "requested_recent_months": recent,
        "minimum_transactions": minimum,
        "returned_limit": COMPARABLE_LIMIT,
    }
    packet = _base_packet(
        analysis_type="comparable_sales",
        title=f"Closest comparable sales for {town_normalised} {flat_type_normalised}",
        filters=filters,
        snapshot=_snapshot_context(prepared, provenance),
    )
    result = find_comparable_sales(
        prepared,
        town=town_normalised,
        flat_type=flat_type_normalised,
        floor_area_sqm=area,
        storey_mid=storey,
        remaining_lease_years=lease,
        recent_months=recent,
        minimum_transactions=minimum,
    )
    latest_month = (
        pd.Timestamp(prepared["month"].max())
        if not prepared.empty
        else pd.Timestamp("1970-01-01")
    )
    ranked = _rank_comparables(
        result.transactions,
        latest_month=latest_month,
        floor_area_sqm=area,
        storey_mid=storey,
        remaining_lease_years=lease,
        floor_area_tolerance=result.floor_area_tolerance,
        storey_tolerance=result.storey_tolerance,
        lease_tolerance=result.lease_tolerance,
    )
    transaction_rows = _comparable_evidence_rows(ranked)
    selection_id = "row.comparable.selection"
    selection_row = {
        "id": selection_id,
        "match_level": result.match_level,
        "matching_pool_transactions": int(len(result.transactions)),
        "returned_transactions": int(len(transaction_rows)),
        "requested_months": int(result.requested_months),
        "effective_months": int(result.effective_months),
        "floor_area_tolerance_sqm": float(result.floor_area_tolerance),
        "storey_tolerance": _json_safe(result.storey_tolerance),
        "remaining_lease_tolerance_years": _json_safe(result.lease_tolerance),
        "explanation": result.explanation,
    }
    packet["evidence_rows"] = [selection_row, *transaction_rows]
    transaction_ids = [str(row["id"]) for row in transaction_rows]
    facts: list[dict[str, object]] = [
        _fact(
            "fact.comparable.match_level",
            "Comparable matching tier",
            result.match_level,
            result.match_level,
            None,
            evidence_ids=[selection_id],
        ),
        _fact(
            "fact.comparable.matching_pool_size",
            "Transactions in matching pool",
            len(result.transactions),
            _format_count(len(result.transactions)),
            "transactions",
            evidence_ids=[selection_id],
        ),
        _fact(
            "fact.comparable.returned_count",
            "Closest comparable transactions returned",
            len(transaction_rows),
            _format_count(len(transaction_rows)),
            "transactions",
            evidence_ids=[selection_id],
        ),
        _fact(
            "fact.comparable.effective_history_months",
            "Effective matching history",
            result.effective_months,
            _format_count(result.effective_months),
            "months",
            evidence_ids=[selection_id],
        ),
    ]
    if transaction_rows:
        selected_prices = [float(row["resale_price_sgd"]) for row in transaction_rows]
        selected_price_per_sqm = [
            float(row["price_per_sqm_sgd"]) for row in transaction_rows
        ]
        selected_median = float(np.median(selected_prices))
        selected_q25 = float(np.quantile(selected_prices, 0.25))
        selected_q75 = float(np.quantile(selected_prices, 0.75))
        selected_minimum = float(min(selected_prices))
        selected_maximum = float(max(selected_prices))
        selected_ppsqm_median = float(np.median(selected_price_per_sqm))
        facts.extend(
            [
                _fact(
                    "fact.comparable.selected_median_price",
                    "Median price among returned comparables",
                    selected_median,
                    _format_sgd(selected_median),
                    "SGD",
                    evidence_ids=transaction_ids,
                ),
                _fact(
                    "fact.comparable.selected_price_q25",
                    "25th-percentile price among returned comparables",
                    selected_q25,
                    _format_sgd(selected_q25),
                    "SGD",
                    evidence_ids=transaction_ids,
                ),
                _fact(
                    "fact.comparable.selected_price_q75",
                    "75th-percentile price among returned comparables",
                    selected_q75,
                    _format_sgd(selected_q75),
                    "SGD",
                    evidence_ids=transaction_ids,
                ),
                _fact(
                    "fact.comparable.selected_minimum_price",
                    "Lowest price among returned comparables",
                    selected_minimum,
                    _format_sgd(selected_minimum),
                    "SGD",
                    evidence_ids=transaction_ids,
                ),
                _fact(
                    "fact.comparable.selected_maximum_price",
                    "Highest price among returned comparables",
                    selected_maximum,
                    _format_sgd(selected_maximum),
                    "SGD",
                    evidence_ids=transaction_ids,
                ),
                _fact(
                    "fact.comparable.selected_median_price_per_sqm",
                    "Median price per sqm among returned comparables",
                    selected_ppsqm_median,
                    _format_sgd_per_sqm(selected_ppsqm_median),
                    "SGD_per_sqm",
                    evidence_ids=transaction_ids,
                ),
            ]
        )
    packet["facts"] = facts
    caveats = [
        result.explanation,
        "The five returned rows are ranked by normalized floor-area, storey and "
        "remaining-lease distance; newer transactions break ties.",
        "Comparable sales are descriptive evidence, not a valuation or a forecast.",
        "Renovation quality, exact floor, view and other unobserved unit features "
        "are not present in the transaction data.",
    ]
    if len(result.transactions) < minimum:
        packet["status"] = "insufficient_evidence"
        caveats.append(
            f"Only {len(result.transactions):,} matching transactions were found, "
            f"below the {minimum:,}-transaction stability target."
        )
    packet["caveats"] = caveats
    return packet


def build_town_comparison_evidence(
    data: pd.DataFrame,
    *,
    towns: Sequence[str] | str,
    flat_type: str | None = None,
    start_month: object | None = None,
    end_month: object | None = None,
    minimum_transactions: int = 3,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare robust price and volume statistics for two or more towns."""
    prepared = _prepare_input(data)
    towns_normalised = _normalise_labels(towns, field="towns")
    if towns_normalised is None or len(towns_normalised) < 2:
        raise EvidenceInputError("towns must contain at least two distinct towns")
    flat_types_normalised = (
        None
        if flat_type is None
        else (_normalise_required_label(flat_type, field="flat_type"),)
    )
    start, end = _month_bounds(start_month, end_month)
    minimum = _positive_integer(
        minimum_transactions, field="minimum_transactions"
    )
    selected = _apply_selection(
        prepared,
        towns=towns_normalised,
        flat_types=flat_types_normalised,
        start_month=start,
        end_month=end,
    )
    filters = {
        "towns": list(towns_normalised),
        "flat_type": (
            flat_types_normalised[0] if flat_types_normalised is not None else None
        ),
        "start_month": _month_text(start),
        "end_month": _month_text(end),
        "minimum_transactions_per_town": minimum,
    }
    packet = _base_packet(
        analysis_type="town_comparison",
        title="HDB resale town comparison",
        filters=filters,
        snapshot=_snapshot_context(prepared, provenance),
    )
    summary = (
        town_summary(selected, minimum_transactions=1).set_index("town")
        if not selected.empty
        else pd.DataFrame()
    )
    evidence_rows: list[dict[str, object]] = []
    facts: list[dict[str, object]] = []
    eligible: list[tuple[str, dict[str, object]]] = []
    for town in towns_normalised:
        row_id = f"row.town.{_slug(town)}"
        if not summary.empty and town in summary.index:
            record = summary.loc[town]
            transactions = int(record["transactions"])
            evidence_row = {
                "id": row_id,
                "town": town,
                "transactions": transactions,
                "median_resale_price_sgd": float(record["median_price"]),
                "median_price_per_sqm_sgd": float(
                    record["median_price_per_sqm"]
                ),
                "million_dollar_share": float(record["million_dollar_share"]),
                "meets_minimum_sample": transactions >= minimum,
            }
        else:
            transactions = 0
            evidence_row = {
                "id": row_id,
                "town": town,
                "transactions": 0,
                "median_resale_price_sgd": None,
                "median_price_per_sqm_sgd": None,
                "million_dollar_share": None,
                "meets_minimum_sample": False,
            }
        evidence_rows.append(evidence_row)
        town_slug = _slug(town)
        facts.append(
            _fact(
                f"fact.town.{town_slug}.transaction_count",
                f"{town} matching transactions",
                transactions,
                _format_count(transactions),
                "transactions",
                evidence_ids=[row_id],
            )
        )
        if evidence_row["meets_minimum_sample"]:
            eligible.append((town, evidence_row))
            median_price = float(evidence_row["median_resale_price_sgd"])
            median_ppsqm = float(evidence_row["median_price_per_sqm_sgd"])
            million_share = float(evidence_row["million_dollar_share"])
            facts.extend(
                [
                    _fact(
                        f"fact.town.{town_slug}.median_resale_price",
                        f"{town} median resale price",
                        median_price,
                        _format_sgd(median_price),
                        "SGD",
                        evidence_ids=[row_id],
                    ),
                    _fact(
                        f"fact.town.{town_slug}.median_price_per_sqm",
                        f"{town} median price per sqm",
                        median_ppsqm,
                        _format_sgd_per_sqm(median_ppsqm),
                        "SGD_per_sqm",
                        evidence_ids=[row_id],
                    ),
                    _fact(
                        f"fact.town.{town_slug}.million_dollar_share",
                        f"{town} million-dollar transaction share",
                        million_share,
                        _format_percent_from_share(million_share),
                        "proportion",
                        evidence_ids=[row_id],
                    ),
                ]
            )
    packet["evidence_rows"] = evidence_rows
    packet["facts"] = facts
    caveats = [
        "Town medians do not adjust for differences in flat size, model, storey "
        "or remaining lease.",
        "Registered resale prices are not forecasts or formal valuations.",
    ]
    if len(eligible) < 2:
        packet["status"] = "insufficient_evidence"
        caveats.append(
            "Fewer than two requested towns meet the minimum sample of "
            f"{minimum:,} transactions."
        )
    else:
        ordered = sorted(
            eligible,
            key=lambda item: (
                -float(item[1]["median_resale_price_sgd"]),
                item[0],
            ),
        )
        highest_town, highest_row = ordered[0]
        lowest_town, lowest_row = ordered[-1]
        gap = float(highest_row["median_resale_price_sgd"]) - float(
            lowest_row["median_resale_price_sgd"]
        )
        facts.extend(
            [
                _fact(
                    "fact.town.highest_median_price_town",
                    "Town with highest median resale price",
                    highest_town,
                    highest_town,
                    None,
                    evidence_ids=[str(highest_row["id"])],
                ),
                _fact(
                    "fact.town.highest_to_lowest_median_price_gap",
                    "Highest-to-lowest town median price gap",
                    gap,
                    _format_sgd(gap),
                    "SGD",
                    evidence_ids=[
                        str(highest_row["id"]),
                        str(lowest_row["id"]),
                    ],
                ),
            ]
        )
    if not selected.empty and selected["year"].nunique():
        partial = coverage_by_year(selected)
        partial_years = partial.loc[partial["is_partial"], "year"].astype(int).tolist()
        if partial_years:
            caveats.append(
                "Partial-year evidence is present for: "
                + ", ".join(str(year) for year in partial_years)
                + "."
            )
    packet["caveats"] = caveats
    return packet


def _nested(mapping: Mapping[str, object], *path: str) -> object | None:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _metric_number(mapping: Mapping[str, object], *path: str) -> float | None:
    value = _nested(mapping, *path)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_integer(mapping: Mapping[str, object], *path: str) -> int | None:
    number = _metric_number(mapping, *path)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _model_snapshot(
    metrics: Mapping[str, object],
    provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    if provenance is not None and not isinstance(provenance, Mapping):
        raise EvidenceInputError("provenance must be a mapping")
    input_provenance = _nested(metrics, "input_provenance")
    input_mapping = input_provenance if isinstance(input_provenance, Mapping) else {}
    start = input_mapping.get("source_month_min") or input_mapping.get(
        "analysis_start_month"
    )
    end = input_mapping.get("source_month_max") or input_mapping.get(
        "analysis_latest_month"
    )
    start_timestamp = _normalise_month(start, field="source_month_min") if start else None
    end_timestamp = _normalise_month(end, field="source_month_max") if end else None
    months_observed = 0
    partial_years: list[int] = []
    if start_timestamp is not None and end_timestamp is not None:
        months_observed = (
            end_timestamp.to_period("M").ordinal
            - start_timestamp.to_period("M").ordinal
            + 1
        )
        if end_timestamp.month < 12:
            partial_years.append(int(end_timestamp.year))
    rows = _metric_integer(metrics, "input_provenance", "source_rows")
    if rows is None:
        rows = _metric_integer(metrics, "input_provenance", "analysis_rows")
    combined_provenance: dict[str, object] = dict(input_mapping)
    if provenance is not None:
        combined_provenance.update(provenance)
    snapshot: dict[str, object] = {
        "coverage": {
            "start_month": _month_text(start_timestamp),
            "end_month": _month_text(end_timestamp),
            "months_observed": months_observed,
            "partial_years": partial_years,
        },
        "rows": rows or 0,
    }
    if combined_provenance:
        snapshot["provenance"] = _json_safe(combined_provenance)
    return snapshot


def build_model_diagnostics_evidence(
    metrics: Mapping[str, object],
    *,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build an evidence packet from the tracked out-of-time model report."""
    if not isinstance(metrics, Mapping):
        raise EvidenceInputError("metrics must be a mapping")
    packet = _base_packet(
        analysis_type="model_diagnostics",
        title="HDB price-model diagnostics",
        filters={
            "model": "hedonic_ridge",
            "evaluation": "out_of_time_holdout",
        },
        snapshot=_model_snapshot(metrics, provenance),
        sources=(HDB_SOURCE, MODEL_METRICS_SOURCE, MODEL_CARD_SOURCE),
    )

    mae = _metric_number(metrics, "holdout_metrics", "hedonic_ridge", "mae")
    rmse = _metric_number(metrics, "holdout_metrics", "hedonic_ridge", "rmse")
    r2 = _metric_number(metrics, "holdout_metrics", "hedonic_ridge", "r2")
    mape = _metric_number(metrics, "holdout_metrics", "hedonic_ridge", "mape_pct")
    holdout_rows = _metric_integer(metrics, "split", "holdout_rows")
    holdout_start = _nested(metrics, "split", "holdout_start_month")
    holdout_end = _nested(metrics, "split", "holdout_end_month")
    improvement = _metric_number(
        metrics, "holdout_mae_improvement_vs_baseline_pct"
    )
    recent_segment_improvement = _metric_number(
        metrics, "holdout_mae_improvement_vs_prior_12m_segment_baseline_pct"
    )
    empirical_coverage = _metric_number(
        metrics, "holdout_prediction_interval", "empirical_coverage"
    )
    target_coverage = _metric_number(
        metrics, "holdout_prediction_interval", "target_coverage"
    )
    interval_width = _metric_number(
        metrics, "holdout_prediction_interval", "median_interval_width"
    )
    folds = _metric_integer(metrics, "rolling_backtests", "folds")
    rolling_mae = _metric_number(
        metrics,
        "rolling_backtests",
        "metrics_by_model",
        "hedonic_ridge",
        "mean_mae",
    )
    rolling_coverage = _metric_number(
        metrics, "rolling_backtests", "mean_empirical_interval_coverage"
    )
    adjusted_change = _metric_number(
        metrics, "index_summary", "quality_adjusted_change_from_base_pct"
    )
    base_month = _nested(metrics, "index_summary", "base_month")
    index_latest_month = _nested(metrics, "index_summary", "latest_month")

    holdout_id = "row.model.latest_holdout"
    interval_id = "row.model.latest_interval"
    rolling_id = "row.model.rolling_summary"
    index_id = "row.model.quality_adjusted_index"
    packet["evidence_rows"] = [
        {
            "id": holdout_id,
            "model": "hedonic_ridge",
            "test_start_month": _json_safe(holdout_start),
            "test_end_month": _json_safe(holdout_end),
            "test_rows": holdout_rows,
            "mae_sgd": mae,
            "rmse_sgd": rmse,
            "r2": r2,
            "mape_pct": mape,
            "mae_improvement_vs_training_median_pct": improvement,
            "mae_improvement_vs_prior_12m_segment_baseline_pct": (
                recent_segment_improvement
            ),
        },
        {
            "id": interval_id,
            "target_coverage": target_coverage,
            "empirical_coverage": empirical_coverage,
            "median_interval_width_sgd": interval_width,
        },
        {
            "id": rolling_id,
            "folds": folds,
            "mean_mae_sgd": rolling_mae,
            "mean_empirical_interval_coverage": rolling_coverage,
        },
        {
            "id": index_id,
            "base_month": _json_safe(base_month),
            "latest_month": _json_safe(index_latest_month),
            "quality_adjusted_change_from_base_pct": adjusted_change,
        },
    ]
    model_source_ids = ["source.price_model_metrics", "source.model_card"]
    facts: list[dict[str, object]] = []

    def add_numeric_fact(
        fact_id: str,
        label: str,
        value: float | int | None,
        formatter: Any,
        unit: str,
        evidence_id: str,
    ) -> None:
        if value is not None:
            facts.append(
                _fact(
                    fact_id,
                    label,
                    value,
                    formatter(value),
                    unit,
                    evidence_ids=[evidence_id],
                    source_ids=model_source_ids,
                )
            )

    add_numeric_fact(
        "fact.model.holdout_mae",
        "Latest holdout mean absolute error",
        mae,
        _format_sgd,
        "SGD",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.holdout_rmse",
        "Latest holdout root mean squared error",
        rmse,
        _format_sgd,
        "SGD",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.holdout_r2",
        "Latest holdout R-squared",
        r2,
        lambda value: f"{float(value):.3f}",
        "coefficient",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.holdout_mape",
        "Latest holdout mean absolute percentage error",
        mape,
        _format_percent,
        "percent",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.holdout_rows",
        "Latest holdout transactions",
        holdout_rows,
        _format_count,
        "transactions",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.mae_improvement_vs_training_median",
        "MAE improvement versus training-median baseline",
        improvement,
        _format_percent,
        "percent",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.mae_improvement_vs_prior_12m_segment_baseline",
        "MAE improvement versus prior-12-month town and flat-type baseline",
        recent_segment_improvement,
        _format_percent,
        "percent",
        holdout_id,
    )
    add_numeric_fact(
        "fact.model.interval_empirical_coverage",
        "Latest empirical interval coverage",
        empirical_coverage,
        _format_percent_from_share,
        "proportion",
        interval_id,
    )
    add_numeric_fact(
        "fact.model.interval_target_coverage",
        "Target interval coverage",
        target_coverage,
        _format_percent_from_share,
        "proportion",
        interval_id,
    )
    add_numeric_fact(
        "fact.model.interval_median_width",
        "Latest median interval width",
        interval_width,
        _format_sgd,
        "SGD",
        interval_id,
    )
    add_numeric_fact(
        "fact.model.rolling_folds",
        "Rolling backtest folds",
        folds,
        _format_count,
        "folds",
        rolling_id,
    )
    add_numeric_fact(
        "fact.model.rolling_mean_mae",
        "Rolling-fold mean absolute error",
        rolling_mae,
        _format_sgd,
        "SGD",
        rolling_id,
    )
    add_numeric_fact(
        "fact.model.rolling_mean_interval_coverage",
        "Rolling-fold mean interval coverage",
        rolling_coverage,
        _format_percent_from_share,
        "proportion",
        rolling_id,
    )
    add_numeric_fact(
        "fact.model.quality_adjusted_index_change",
        "Quality-adjusted index change from base month",
        adjusted_change,
        _format_percent,
        "percent",
        index_id,
    )
    packet["facts"] = facts

    required = (mae, rmse, r2, holdout_rows, empirical_coverage, target_coverage)
    caveats = [
        "The model is designed for market monitoring, not formal unit appraisal, "
        "causal inference or financial advice.",
        "Historical out-of-time performance does not guarantee accuracy in a new "
        "market regime or for a particular flat.",
        "Unobserved renovation quality, view, exact floor and other omitted "
        "attributes can affect transaction prices.",
    ]
    if any(value is None for value in required):
        packet["status"] = "insufficient_evidence"
        caveats.append(
            "The model report is missing one or more required holdout or interval "
            "metrics; rebuild the tracked model artifacts before interpretation."
        )
    if (
        empirical_coverage is not None
        and target_coverage is not None
        and empirical_coverage < target_coverage
    ):
        shortfall = (target_coverage - empirical_coverage) * 100
        caveats.append(
            f"Latest empirical interval coverage is {shortfall:.1f} percentage "
            "points below target; the ranges are diagnostics, not guaranteed "
            "confidence intervals."
        )
    packet["caveats"] = caveats
    return packet


# Concise aliases keep the domain vocabulary convenient for callers while the
# explicit build_* names make the return type clear at integration points.
market_summary = build_market_summary_evidence
comparable_sales = build_comparable_sales_evidence
town_comparison = build_town_comparison_evidence
model_diagnostics = build_model_diagnostics_evidence


__all__ = [
    "COMPARABLE_LIMIT",
    "SCHEMA_VERSION",
    "EvidenceInputError",
    "build_comparable_sales_evidence",
    "build_market_summary_evidence",
    "build_model_diagnostics_evidence",
    "build_town_comparison_evidence",
    "comparable_sales",
    "market_summary",
    "model_diagnostics",
    "town_comparison",
]
