# Singapore HDB Resale Market Analytics v1.0.0

This is the first portfolio release of the reproducible HDB resale analytics
project, built from official Singapore public data.

## Highlights

- Six-tab Streamlit market explorer with transparent comparable sales
- Official SingStat CPI and LTA MRT-exit enrichment with provenance
- Four expanding-window model backtests and stronger segment baselines
- Error slices plus empirical price-range coverage diagnostics
- Resilient official-data downloads with bounded retry and rate-limit handling
- Verified Parquet snapshots, SQLite reconciliation and 88 automated tests
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

Block-to-MRT distances remain unavailable until the user supplies a OneMap API
