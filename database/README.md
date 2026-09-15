# SQLite database

The generated `hdb_resale.db` file is kept out of Git because it can be rebuilt
from the processed CSV.

From the project root, run:

```powershell
python src/clean_data.py
python src/build_database.py
```

The build creates the `resale_transactions` table with 17 columns and indexes
on `year`, `town`, `flat_type`, and `(town, year)`. Run the queries in
`sql/analysis_queries.sql` against the generated database.

The scripts validate that the CSV and database contain the same number of rows.
