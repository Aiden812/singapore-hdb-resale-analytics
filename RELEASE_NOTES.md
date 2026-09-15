# Singapore HDB Resale Market Analytics v1.1.0

This release tightens the dashboard's data-provenance messaging and improves its
first-screen presentation without expanding the project's scope.

## Highlights

- Verified enriched snapshots now display the transaction source observation
  date rather than the artifact-generation date
- A regression check protects the 8 September 2026 source date from being shown
  as 15 September when the default enriched snapshot is loaded
- MRT content is labelled as an official **reference layer**; the dashboard does
  not imply that MRT distance is attached to transactions while block-coordinate
  coverage remains 0%
- The first screen uses one compact data-coverage notice and a readable
  price-per-square-metre KPI
- The README now leads with the intended user, problem, personal contribution,
  three evidence-backed findings, demo links and technology stack
- Dashboard preview assets were refreshed for the v1.1 interface
- Six-tab Streamlit market explorer with transparent comparable sales
- Official SingStat CPI enrichment and a separately validated LTA MRT-exit
  reference layer with provenance
- Four expanding-window model backtests and stronger segment baselines
- Error slices plus empirical price-range coverage diagnostics
- Resilient official-data downloads with bounded retry and rate-limit handling
- Verified Parquet snapshots, SQLite reconciliation and **92 automated tests**
- MIT-licensed code with complete third-party data attribution

## Snapshot results

- 239,977 transactions from January 2017 through provisional September 2026
- Latest 12-month Ridge MAE: S$52,844
- Ridge MAE improvement over the prior-12-month segment baseline: 35.5%
- Mean four-fold Ridge MAE: S$52,347
- Mean empirical interval coverage: 70.7% against an 80% target
- CPI-adjusted raw median growth through July 2026: 28.9%

## Important limits

The model is for market monitoring, not formal valuation or financial advice.
The empirical price range transparently under-covers its nominal 80% backtest
target.

Block-to-MRT distances remain unavailable until a user supplies a OneMap API
token and builds the optional block-coordinate cache. The included LTA data is a
validated station-exit reference layer, not transaction-level MRT enrichment.
