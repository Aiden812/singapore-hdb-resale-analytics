# Data dictionary

The clean dataset contains the 11 fields supplied by the Housing & Development
Board plus six fields created by the Python cleaning pipeline. The enriched
snapshot preserves every clean row and adds CPI fields plus optional verified
block and MRT proximity fields.

| Column | Type after cleaning | Origin | Description |
| --- | --- | --- | --- |
| `month` | date | Source | Transaction registration month, stored as the first day of the month. |
| `town` | text | Source | HDB town in which the flat is located. |
| `flat_type` | text | Source | Flat category, such as 3 Room, 4 Room or Executive. |
| `block` | text | Source | Block identifier. It remains text because identifiers can include letters. |
| `street_name` | text | Source | Street name supplied for the transaction. |
| `storey_range` | text | Source | Three-storey band containing the unit, such as `04 TO 06`. |
| `floor_area_sqm` | decimal | Source | Approximate floor area in square metres. |
| `flat_model` | text | Source | HDB flat-model classification. |
| `lease_commence_date` | whole number | Source | Calendar year in which the lease commenced. |
| `remaining_lease` | text | Source | Remaining lease reported as years and months. |
| `resale_price` | decimal | Source | Registered resale transaction price in Singapore dollars. |
| `year` | whole number | Engineered | Calendar year extracted from `month`. |
| `month_number` | whole number | Engineered | Calendar month number from 1 to 12. |
| `price_per_sqm` | decimal | Engineered | `resale_price / floor_area_sqm`. |
| `flat_age` | whole number | Engineered | Approximate age at sale: `year - lease_commence_date`. |
| `storey_mid` | decimal | Engineered | Midpoint of the lower and upper values in `storey_range`. |
| `remaining_lease_months` | whole number | Engineered | Remaining lease converted to total months, validated from 1 through 1,188. |
| `cpi_all_items` | decimal | SingStat | Monthly all-items CPI matched exactly to the transaction month. |
| `cpi_reference_month` | date | Engineered | Latest official CPI month used as the constant-dollar reference. |
| `resale_price_real_sgd` | decimal | Engineered | Resale price expressed in reference-month Singapore dollars. |
| `price_per_sqm_real_sgd` | decimal | Engineered | Price per sqm expressed in reference-month Singapore dollars. |
| `latitude` | decimal | Optional OneMap | Verified latitude for the block-and-street address. |
| `longitude` | decimal | Optional OneMap | Verified longitude for the block-and-street address. |
| `nearest_mrt_station` | text | Engineered | Nearest official MRT station exit name; LRT exits are excluded. |
| `nearest_mrt_exit_code` | text | LTA | Official exit feature code associated with the nearest MRT point. |
| `nearest_mrt_distance_m` | decimal | Engineered | Great-circle distance to the nearest MRT exit in metres. |
| `mrt_distance_band` | category | Engineered | Interpretable band derived from `nearest_mrt_distance_m`. |

Sources:

- [HDB resale transactions](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view),
  Housing & Development Board via data.gov.sg.
- [Monthly Consumer Price Index](https://data.gov.sg/datasets/d_bdaff844e3ef89d39fceb962ff8f0791/view),
  Singapore Department of Statistics via data.gov.sg.
- [LTA MRT Station Exit](https://data.gov.sg/datasets/d_b39d3a0871985372d7e1637193335da5/view),
  Land Transport Authority via data.gov.sg.

`flat_age` is an approximate whole-year feature, not the precise age on the
transaction date. `storey_mid` represents a band and is not an exact unit floor.
`remaining_lease_months` preserves month-level detail from the source text and is
the authoritative lease feature used by the model and dashboard.

CPI values are joined by exact transaction month and are never forward-filled.
When the newest CPI observation has not yet been published, the real-price fields
remain missing. Location and MRT fields appear only when a verified OneMap block
cache is supplied; no coordinates are inferred from ambiguous block numbers.
Straight-line distance is an accessibility proxy, not a travel-time measure.
The LTA layer is a current infrastructure snapshot and can include stations that
are not yet in passenger service, so it must not be interpreted as historical
operational access.
