-- Singapore HDB resale analytics queries (SQLite).
-- Run these against database/hdb_resale.db after running src/build_database.py.

-- Overall dataset summary.
SELECT
    COUNT(*) AS total_transactions,
    ROUND(AVG(resale_price), 2) AS average_resale_price,
    ROUND(AVG(price_per_sqm), 2) AS average_price_per_sqm,
    MIN(month) AS first_month,
    MAX(month) AS latest_month
FROM resale_transactions;

-- Q1: Median resale price by year.
WITH ranked AS (
    SELECT
        year,
        resale_price,
        ROW_NUMBER() OVER (
            PARTITION BY year
            ORDER BY resale_price
        ) AS price_position,
        COUNT(*) OVER (PARTITION BY year) AS group_size
    FROM resale_transactions
    WHERE resale_price IS NOT NULL
)
SELECT
    year,
    ROUND(AVG(resale_price), 2) AS median_resale_price
FROM ranked
WHERE price_position IN ((group_size + 1) / 2, (group_size + 2) / 2)
GROUP BY year
ORDER BY year;

-- Q2: Towns with the highest median resale prices.
WITH ranked AS (
    SELECT
        town,
        resale_price,
        ROW_NUMBER() OVER (
            PARTITION BY town
            ORDER BY resale_price
        ) AS price_position,
        COUNT(*) OVER (PARTITION BY town) AS group_size
    FROM resale_transactions
    WHERE resale_price IS NOT NULL
)
SELECT
    town,
    ROUND(AVG(resale_price), 2) AS median_resale_price
FROM ranked
WHERE price_position IN ((group_size + 1) / 2, (group_size + 2) / 2)
GROUP BY town
ORDER BY median_resale_price DESC;

-- Q3: Towns with the most transactions.
SELECT
    town,
    COUNT(*) AS total_transactions
FROM resale_transactions
GROUP BY town
ORDER BY total_transactions DESC, town;

-- Q4: Median resale price by flat type.
WITH ranked AS (
    SELECT
        flat_type,
        resale_price,
        ROW_NUMBER() OVER (
            PARTITION BY flat_type
            ORDER BY resale_price
        ) AS price_position,
        COUNT(*) OVER (PARTITION BY flat_type) AS group_size
    FROM resale_transactions
    WHERE resale_price IS NOT NULL
)
SELECT
    flat_type,
    ROUND(AVG(resale_price), 2) AS median_resale_price
FROM ranked
WHERE price_position IN ((group_size + 1) / 2, (group_size + 2) / 2)
GROUP BY flat_type
ORDER BY median_resale_price;

-- Q5: Resale price by 10-square-metre floor-area band.
WITH area_bands AS (
    SELECT
        CAST(floor_area_sqm / 10 AS INTEGER) * 10 AS floor_area_band,
        resale_price
    FROM resale_transactions
    WHERE floor_area_sqm IS NOT NULL
      AND resale_price IS NOT NULL
)
SELECT
    printf('%d-%d sqm', floor_area_band, floor_area_band + 9) AS area_range,
    COUNT(*) AS total_transactions,
    ROUND(AVG(resale_price), 2) AS average_resale_price
FROM area_bands
GROUP BY floor_area_band
ORDER BY floor_area_band;

-- Q6: Resale price by five-year flat-age band.
WITH age_bands AS (
    SELECT
        CAST(flat_age / 5 AS INTEGER) * 5 AS flat_age_band,
        resale_price
    FROM resale_transactions
    WHERE flat_age IS NOT NULL
      AND resale_price IS NOT NULL
)
SELECT
    printf('%d-%d years', flat_age_band, flat_age_band + 4) AS age_range,
    COUNT(*) AS total_transactions,
    ROUND(AVG(resale_price), 2) AS average_resale_price
FROM age_bands
GROUP BY flat_age_band
ORDER BY flat_age_band;

-- Q7: Median resale price by storey range, sorted by numeric midpoint.
WITH ranked AS (
    SELECT
        storey_range,
        storey_mid,
        resale_price,
        ROW_NUMBER() OVER (
            PARTITION BY storey_range
            ORDER BY resale_price
        ) AS price_position,
        COUNT(*) OVER (PARTITION BY storey_range) AS group_size
    FROM resale_transactions
    WHERE resale_price IS NOT NULL
)
SELECT
    storey_range,
    storey_mid,
    ROUND(AVG(resale_price), 2) AS median_resale_price
FROM ranked
WHERE price_position IN ((group_size + 1) / 2, (group_size + 2) / 2)
GROUP BY storey_range, storey_mid
ORDER BY storey_mid;

-- Q8: Town price growth between the earliest and latest complete years.
-- A complete year must contain transactions in all 12 calendar months.
WITH complete_years AS (
    SELECT year
    FROM resale_transactions
    WHERE year < CAST(strftime('%Y', 'now') AS INTEGER)
    GROUP BY year
    HAVING COUNT(DISTINCT month_number) = 12
),
year_bounds AS (
    SELECT
        MIN(year) AS first_year,
        MAX(year) AS last_year
    FROM complete_years
),
ranked AS (
    SELECT
        transactions.town,
        transactions.year,
        transactions.resale_price,
        ROW_NUMBER() OVER (
            PARTITION BY transactions.town, transactions.year
            ORDER BY transactions.resale_price
        ) AS price_position,
        COUNT(*) OVER (
            PARTITION BY transactions.town, transactions.year
        ) AS group_size
    FROM resale_transactions AS transactions
    INNER JOIN complete_years USING (year)
    WHERE transactions.resale_price IS NOT NULL
),
annual_medians AS (
    SELECT
        town,
        year,
        AVG(resale_price) AS median_resale_price,
        MAX(group_size) AS total_transactions
    FROM ranked
    WHERE price_position IN ((group_size + 1) / 2, (group_size + 2) / 2)
    GROUP BY town, year
),
town_endpoints AS (
    SELECT
        town,
        MAX(
            CASE WHEN year = (SELECT first_year FROM year_bounds)
                THEN median_resale_price END
        ) AS first_year_price,
        MAX(
            CASE WHEN year = (SELECT last_year FROM year_bounds)
                THEN median_resale_price END
        ) AS last_year_price,
        MAX(
            CASE WHEN year = (SELECT first_year FROM year_bounds)
                THEN total_transactions END
        ) AS first_year_transactions,
        MAX(
            CASE WHEN year = (SELECT last_year FROM year_bounds)
                THEN total_transactions END
        ) AS last_year_transactions
    FROM annual_medians
    GROUP BY town
)
SELECT
    endpoints.town,
    bounds.first_year,
    bounds.last_year,
    ROUND(endpoints.first_year_price, 2) AS first_year_median,
    ROUND(endpoints.last_year_price, 2) AS last_year_median,
    endpoints.first_year_transactions,
    endpoints.last_year_transactions,
    ROUND(
        100.0 * (endpoints.last_year_price - endpoints.first_year_price)
        / endpoints.first_year_price,
        2
    ) AS median_price_growth_pct
FROM town_endpoints AS endpoints
CROSS JOIN year_bounds AS bounds
WHERE endpoints.first_year_price IS NOT NULL
  AND endpoints.last_year_price IS NOT NULL
  AND endpoints.first_year_transactions >= 100
  AND endpoints.last_year_transactions >= 100
ORDER BY median_price_growth_pct DESC, endpoints.town;
