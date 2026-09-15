# Advanced descriptive and robustness analysis

## Scope and coverage

The transaction snapshot spans 2017-01 to 2026-09 and contains 239,977 rows.

Years without all 12 month labels are disclosed as partial: 2026. They are retained in annual descriptive tables but are not used as town-growth endpoints.

The latest transaction month, 2026-09, is provisional: the snapshot was observed as of 8 September 2026 Singapore time and contains 512 rows for that month. Those rows remain in year-to-date descriptive totals.

## Price distribution

In 2026, the observed P25, median, P75 and P90 were S$515,000, S$630,000, S$780,000 and S$950,000. This period contains 9 month labels.

## Million-dollar transactions

The observed share in 2026 was 7.45% (1,334 of 17,910 records).

## Town endpoint comparison

Among eligible towns, TOA PAYOH had the largest change in raw median price between 2017 and 2025: 76.40%. Endpoint sample sizes were 749 and 1,063.

## Duplicate sensitivity

The exact-row sensitivity removes 318 rows. The overall median changes by S$0.00 (0.0000%).

Fully identical rows cannot be confirmed as errors because the source has no transaction identifier or unit number. Removal is shown only as a sensitivity check.

## Official HDB RPI benchmark

The benchmark covers 2017-Q1 through 2026-Q2. Both series are rebased to 100 at the first common quarter.

Official source: [d_14f63e595975691e7c24a27ae4c07c79](https://data.gov.sg/datasets/d_14f63e595975691e7c24a27ae4c07c79/view).

The raw-median series is composition-sensitive and is not equivalent to the official HDB Resale Price Index. Rebasing permits a visual comparison of movement, not a claim that the measures are interchangeable.

## Python/SQL reconciliation

Python and SQLite annual medians matched within S$0.01 for 10 of 10 compared years.

## Interpretation

Results are descriptive and unadjusted. Changes can reflect the mix of towns, flat types, sizes, leases and storeys sold; they do not identify causal effects.
