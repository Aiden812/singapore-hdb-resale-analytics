# Singapore HDB Resale Market Analytics v1.2.0

This release adds an evidence-grounded HDB Resale Decision Copilot while
preserving the existing six-tab analytics dashboard and its market-monitoring
scope.

## Highlights

- A separate Streamlit copilot offers market briefs, comparable sales, town
  comparisons and model-reliability explanations.
- All analytics remain deterministic: compact evidence packets carry stable fact
  and row IDs, source provenance, filters, sample sizes and coverage dates.
- Optional OpenAI evidence selection uses the Responses API, strict Pydantic
  Structured Outputs, `store=False` and an environment-configured model.
- Model-written prose never reaches the display boundary. Only validated
  evidence-ID choices survive, and every visible claim is reconstructed by
  deterministic project code.
- A post-generation gate checks citations, numeric support, recognized units and
  tested valuation, forecast and financial-advice phrasing. Any failed check
  falls back to deterministic evidence rather than displaying the draft.
- Missing credentials, API errors and invalid model output fall back to a useful
  deterministic summary; the app is fully demonstrable offline.
- Comparable evidence ranks five closest observed transactions deterministically
  and exposes exact quartiles, matching tiers and widening criteria.
- Model evidence includes chronological holdout metrics, rolling folds, interval
  under-coverage and improvement over the prior-12-month segment baseline.
- Forty frozen cases route their actual inputs through the tracked HDB snapshot,
  deterministic evidence builders and model report. They cover normal, sparse,
  out-of-range, injection, unsupported-request and numerical-grounding behavior.
- All 147 automated tests pass, including real-data offline submissions through
  every copilot mode and the unchanged dashboard regression suite.
- The copilot loads only checksum-verified tracked snapshots and reports the
  model-metrics hash separately from the compatible transaction-snapshot hash.
- The original `app.py` remains unchanged; CI now compiles and tests both apps.
- Dependencies, environment templates, architecture notes and release checks are
  pinned and documented for a reproducible portfolio review.

## Snapshot results

- 239,977 transactions from January 2017 through provisional September 2026
- Latest 12-month Ridge MAE: S$52,844
- Ridge MAE improvement over the prior-12-month segment baseline: 35.5%
- Mean four-fold Ridge MAE: S$52,347
- Mean empirical interval coverage: 70.7% against an 80% target
- CPI-adjusted raw median growth through July 2026: 28.9%

## Important limits

The model and copilot are for descriptive market monitoring, not formal
valuation, future-price prediction or financial advice. The empirical price
range transparently under-covers its nominal 80% backtest target, so no
unit-price estimator is exposed.

Block-to-MRT distances remain unavailable until a user supplies a OneMap API
token and builds the optional block-coordinate cache. The included LTA data is a
validated station-exit reference layer, not transaction-level MRT enrichment.
