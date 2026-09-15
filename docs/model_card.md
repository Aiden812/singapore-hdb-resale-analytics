# Quality-adjusted price model

## Purpose

The model estimates a transaction-mix-adjusted HDB resale price trend. It asks
how the common time component changes when the recorded floor area, remaining
lease, storey, town, flat type and flat model are held constant in the model.
It is designed for market monitoring, not appraisal or causal inference.

## Data

- Source: HDB resale registrations published on data.gov.sg.
- Current source coverage: January 2017 through 8 September 2026.
- The 512 September 2026 records are provisional and excluded from modelling;
  the model and index use 239,465 records through August 2026.
- Target: natural logarithm of `resale_price`.
- Quality controls: log-quadratic floor area, quadratic remaining lease,
  quadratic storey midpoint, town, flat type and flat model.
- Time controls: a trend and annual seasonality for holdout evaluation;
  transaction-month fixed effects for the published index.
- Regularisation: Ridge (`alpha=1`) is used for predictive evaluation. The
  published index uses unpenalised month fixed effects (`alpha=0`) so the time
  movement is not mechanically shrunk by the prediction penalty.

The exact input snapshot is identified by
`data/processed/snapshot_metadata.json`. The model report records the input
SHA-256, row counts, as-of source, partial-month exclusion and output hashes.
The dashboard verifies those hashes before displaying the model artifacts.

## Validation design

The latest 12 complete transaction months, September 2025 through August 2026,
are held out. The model is trained only on earlier records and compared with a
global training median, an all-history town-by-flat-type median and a stronger
prior-12-month town-by-flat-type median with documented fallbacks.

| Holdout result | Median baseline | Hedonic ridge model |
| --- | ---: | ---: |
| MAE | S$206,276 | S$52,844 |
| RMSE | S$274,221 | S$69,635 |
| R² | -0.654 | 0.893 |

The model reduces holdout MAE by 74.4% relative to the deliberately simple
training-median baseline and by 35.5% relative to the prior-12-month segment
baseline. Grouped permutation importance is calculated only after evaluation.

Four non-overlapping 12-month expanding-window folds test performance across
multiple historical regimes rather than relying on the final holdout alone.

| Model | Mean MAE | Mean median AE | Mean p90 AE |
| --- | ---: | ---: | ---: |
| Training median | S$185,522 | S$138,472 | S$416,750 |
| Town and flat-type median | S$149,805 | S$128,222 | S$286,972 |
| Prior-12-month town and flat-type median | S$79,287 | S$50,889 | S$188,844 |
| Hedonic Ridge | S$52,347 | S$42,297 | S$107,902 |

Empirical 10th, 50th and 90th percentile estimates use a six-month calibration
window inside each training period. Mean four-fold coverage is 70.7% against an
80% target. The latest fold covers 75.0%, with a median interval width of
S$148,136. The dashboard discloses this undercoverage. These ranges are
diagnostics, not guaranteed confidence intervals.

Error slices report MAE, median and 90th-percentile absolute error, RMSE, R²,
MAPE and mean error by town, flat type, observed price band and remaining lease.

## Price-index result

From January 2017 through August 2026, the raw monthly median index changes
by 56.8%. The model's quality-adjusted index changes by 56.6%. The 0.2
percentage-point gap is small: cumulative raw and adjusted movements are nearly
identical under this unpenalised fixed-effects specification. This does not rule
out composition effects in individual months or segments.

## Limitations

- The index is hedonic, not a repeat-sales index. The public data does not
  identify individual units.
- Unobserved renovation quality, exact unit floor, view, proximity to transport
  and other omitted attributes can affect estimates.
- The current model omits MRT proximity because verified block coordinates are
  unavailable without a OneMap cache; station exits alone are not enough.
- The empirical uncertainty range undercovers the latest fold and must not be
  interpreted as a guaranteed 80% interval.
- Historical backtests cannot guarantee performance after a new market regime.
- The latest calendar year is partial. The incomplete latest transaction month
  is excluded using the source snapshot's Singapore as-of date.
- Ridge regularisation shrinks predictive-model coefficients; the published
  transaction-month fixed effects are unpenalised.
- Feature importance describes reliance by this fitted model. It is not a
  causal ranking.
- Predictions should not be used as formal valuations or financial advice.

## Reproduce

```powershell
python src/clean_data.py
python src/model_price.py --input data/processed/hdb_resale_clean.parquet
```

Machine-readable metrics are stored in `reports/price_model_metrics.json`.
Rolling fold metrics, interval diagnostics and segment errors are stored in
`reports/price_model_backtests.csv`, `reports/price_model_interval_metrics.csv`
and `reports/price_model_error_slices.csv`. The monthly index is stored in
`reports/quality_adjusted_price_index.csv`.
