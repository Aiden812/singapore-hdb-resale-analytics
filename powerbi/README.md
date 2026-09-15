# Power BI dashboard guide

## Load the data

1. Run `python src/clean_data.py` from the project root.
2. In Power BI Desktop, select **Get data > Text/CSV**.
3. Open `data/processed/hdb_resale_clean.csv`.
4. Name the table `HDB Resale` and confirm these data types:

| Column | Data type |
| --- | --- |
| `month` | Date |
| `year`, `month_number`, `lease_commence_date`, `flat_age` | Whole number |
| `floor_area_sqm`, `resale_price`, `price_per_sqm`, `storey_mid` | Decimal number |
| Remaining columns | Text |

## Measures

Create these measures in the `HDB Resale` table:

```DAX
Total Transactions = COUNTROWS('HDB Resale')

Median Resale Price = MEDIAN('HDB Resale'[resale_price])

Average Resale Price = AVERAGE('HDB Resale'[resale_price])

Median Price per sqm = MEDIAN('HDB Resale'[price_per_sqm])

First Complete Year =
VAR YearCoverage =
    SUMMARIZE(
        ALL('HDB Resale'),
        'HDB Resale'[year],
        "MonthCount", CALCULATE(DISTINCTCOUNT('HDB Resale'[month_number]))
    )
RETURN
    MINX(
        FILTER(
            YearCoverage,
            [MonthCount] = 12 && 'HDB Resale'[year] < YEAR(TODAY())
        ),
        'HDB Resale'[year]
    )

Latest Complete Year =
VAR YearCoverage =
    SUMMARIZE(
        ALL('HDB Resale'),
        'HDB Resale'[year],
        "MonthCount", CALCULATE(DISTINCTCOUNT('HDB Resale'[month_number]))
    )
RETURN
    MAXX(
        FILTER(
            YearCoverage,
            [MonthCount] = 12 && 'HDB Resale'[year] < YEAR(TODAY())
        ),
        'HDB Resale'[year]
    )

Median Price First Complete Year =
VAR FirstYear = [First Complete Year]
RETURN
    CALCULATE(
        [Median Resale Price],
        REMOVEFILTERS('HDB Resale'[year]),
        'HDB Resale'[year] = FirstYear
    )

Median Price Latest Complete Year =
VAR LatestYear = [Latest Complete Year]
RETURN
    CALCULATE(
        [Median Resale Price],
        REMOVEFILTERS('HDB Resale'[year]),
        'HDB Resale'[year] = LatestYear
    )

Town Median Price Growth % =
DIVIDE(
    [Median Price Latest Complete Year] - [Median Price First Complete Year],
    [Median Price First Complete Year]
)
```

Format price measures as Singapore dollars and `Total Transactions` as a whole
number with a thousands separator. Format `Town Median Price Growth %` as a
percentage.

## Suggested report pages

### Market overview

- Cards: total transactions, median resale price, average resale price, and
  median price per square metre.
- Line chart: median resale price by `month`.
- Column chart: transaction count by `year`.
- Slicers: year, town, and flat type.

### Town comparison

- Bar chart: median resale price by town.
- Bar chart: transaction count by town.
- Matrix: town and flat type with transaction count and median price.

### Flat characteristics

- Bar chart: median resale price by flat type.
- Scatter plot: floor area against resale price.
- Scatter plot: flat age against resale price.
- Column chart: median resale price by `storey_mid`, sorted ascending.

### Town price growth

- Bar chart: `Town Median Price Growth %` by town, sorted descending.
- Tooltips: median price in the first and latest complete year.
- Cards: first complete year and latest complete year.
- Keep current-year filters off this page because the growth measures select
  complete-year endpoints automatically.

## Interpretation notes

- Treat the latest year as partial until all 12 months are present.
- Median price comparisons do not control for changes in town, flat type,
  floor area, age, or other transaction characteristics.
- Correlation describes association and does not establish causation.
- The source has no transaction ID or unit number, so fully identical rows are
  retained until they can be verified.

Save the Power BI report as `powerbi/hdb_resale_dashboard.pbix`. The PBIX file
is ignored by Git; add screenshots to `images/` for the portfolio README.
