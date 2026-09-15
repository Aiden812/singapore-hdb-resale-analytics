# Data dictionary

The clean dataset contains the 11 fields supplied by the Housing & Development
Board plus six fields created by the Python cleaning pipeline.

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

Source: [Resale flat prices based on registration date from Jan 2017
onwards](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view),
Housing & Development Board via data.gov.sg.

`flat_age` is an approximate whole-year feature, not the precise age on the
transaction date. `storey_mid` represents a band and is not an exact unit floor.
`remaining_lease_months` preserves month-level detail from the source text and is
the authoritative lease feature used by the model and dashboard.
