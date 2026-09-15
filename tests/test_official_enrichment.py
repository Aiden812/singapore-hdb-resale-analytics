"""Focused tests for official CPI and MRT enrichment."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from src.official_enrichment import (
    CPI_DATASET_ID,
    MRT_DATASET_ID,
    add_cpi_adjustment,
    add_mrt_proximity,
    assign_mrt_distance_band,
    download_cpi_dataset,
    download_mrt_dataset,
    file_sha256,
    haversine_distance_m,
    monthly_price_inflation_summary,
    mrt_proximity_summary,
    nearest_mrt_exits,
    normalise_block_coordinates,
    normalise_cpi_data,
    normalise_mrt_geojson,
    run_enrichment,
)


def cpi_wide() -> pd.DataFrame:
    """Return a small fixture shaped like the official SingStat dataset."""
    return pd.DataFrame(
        [
            {
                "_id": 1,
                "DataSeries": "All Items",
                "2020Feb": "110.0",
                "2020Jan": "100.0",
            },
            {
                "_id": 2,
                "DataSeries": "Food",
                "2020Feb": "120.0",
                "2020Jan": "115.0",
            },
        ]
    )


def station_geojson() -> dict[str, object]:
    """Return synthetic MRT and LRT points in official GeoJSON shape."""
    return {
        "type": "FeatureCollection",
        "name": "MRT_EXITS",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.8000, 1.3000]},
                "properties": {
                    "OBJECTID": 1,
                    "STATION_NA": "ALPHA MRT STATION",
                    "EXIT_CODE": "Exit A",
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.8100, 1.3100]},
                "properties": {
                    "OBJECTID": 2,
                    "STATION_NA": "BETA MRT STATION",
                    "EXIT_CODE": "Exit B",
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.8001, 1.3001]},
                "properties": {
                    "OBJECTID": 3,
                    "STATION_NA": "GAMMA LRT STATION",
                    "EXIT_CODE": "Exit A",
                },
            },
        ],
    }


def transactions() -> pd.DataFrame:
    """Return a minimal clean-style transaction frame."""
    return pd.DataFrame(
        [
            {
                "month": "2020-01",
                "town": "ALPHA",
                "flat_type": "4 ROOM",
                "block": "1",
                "street_name": "TEST ROAD",
                "floor_area_sqm": 100.0,
                "resale_price": 1_100.0,
                "price_per_sqm": 11.0,
            },
            {
                "month": "2020-02",
                "town": "ALPHA",
                "flat_type": "4 ROOM",
                "block": "2",
                "street_name": "TEST ROAD",
                "floor_area_sqm": 100.0,
                "resale_price": 2_200.0,
                "price_per_sqm": 22.0,
            },
        ]
    )


class CpiTests(unittest.TestCase):
    def test_normalises_official_wide_all_items_series(self) -> None:
        result = normalise_cpi_data(cpi_wide())

        self.assertEqual(
            result["month"].dt.strftime("%Y-%m").tolist(), ["2020-01", "2020-02"]
        )
        self.assertEqual(result["cpi_all_items"].tolist(), [100.0, 110.0])

    def test_rejects_duplicate_months_in_cached_long_format(self) -> None:
        duplicated = pd.DataFrame(
            {
                "month": ["2020-01", "2020-01"],
                "cpi_all_items": [100.0, 101.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate months"):
            normalise_cpi_data(duplicated)

    def test_deflates_prices_to_an_explicit_reference_month(self) -> None:
        enriched, reference = add_cpi_adjustment(
            transactions(),
            cpi_wide(),
            reference_month="2020-02",
        )

        self.assertEqual(reference, "2020-02")
        self.assertAlmostEqual(enriched.loc[0, "resale_price_real_sgd"], 1_210.0)
        self.assertAlmostEqual(enriched.loc[1, "resale_price_real_sgd"], 2_200.0)
        self.assertAlmostEqual(enriched.loc[0, "price_per_sqm_real_sgd"], 12.1)

    def test_does_not_forward_fill_an_unpublished_cpi_month(self) -> None:
        frame = pd.concat(
            [
                transactions(),
                transactions().iloc[[0]].assign(month="2020-03"),
            ],
            ignore_index=True,
        )

        enriched, _ = add_cpi_adjustment(frame, cpi_wide())

        self.assertTrue(pd.isna(enriched.loc[2, "cpi_all_items"]))
        self.assertTrue(pd.isna(enriched.loc[2, "resale_price_real_sgd"]))

    def test_monthly_summary_rebases_nominal_and_real_series(self) -> None:
        enriched, _ = add_cpi_adjustment(transactions(), cpi_wide())
        summary = monthly_price_inflation_summary(enriched)

        self.assertEqual(summary["month"].tolist(), ["2020-01", "2020-02"])
        self.assertEqual(summary.loc[0, "nominal_price_index"], 100.0)
        self.assertEqual(summary.loc[0, "real_price_index"], 100.0)
        self.assertAlmostEqual(summary.loc[1, "nominal_price_index"], 200.0)
        self.assertAlmostEqual(summary.loc[1, "real_price_index"], 2_200 / 1_210 * 100)

    def test_downloads_only_the_official_all_items_series(self) -> None:
        response = Mock()
        response.json.return_value = {
            "success": True,
            "result": {"records": cpi_wide().to_dict("records")[:1]},
        }
        session = Mock()
        session.get.return_value = response

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "cpi.csv"
            row_count = download_cpi_dataset(output_path, session=session)
            saved = pd.read_csv(output_path)

        self.assertEqual(row_count, 2)
        self.assertEqual(saved["month"].tolist(), ["2020-01", "2020-02"])
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["resource_id"], CPI_DATASET_ID)
        self.assertIn("All Items", params["filters"])
        response.raise_for_status.assert_called_once()

    def test_retries_transient_cpi_network_failure_without_real_wait(self) -> None:
        response = Mock(status_code=200, headers={})
        response.json.return_value = {
            "success": True,
            "result": {"records": cpi_wide().to_dict("records")[:1]},
        }
        session = Mock()
        session.get.side_effect = [
            requests.ConnectionError("connection reset"),
            response,
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "cpi.csv"
            with patch("src.http_retry.time.sleep") as sleep:
                row_count = download_cpi_dataset(output_path, session=session)

        self.assertEqual(row_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(session.get.call_count, 2)


class MrtTests(unittest.TestCase):
    def test_normalises_mrt_points_and_excludes_lrt(self) -> None:
        result = normalise_mrt_geojson(station_geojson())

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["rail_mode"]), {"MRT"})
        self.assertNotIn("GAMMA LRT STATION", set(result["station_name"]))

    def test_canonicalises_official_features_labelled_by_mrt_code(self) -> None:
        payload = station_geojson()
        payload["features"].append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.7687, 1.3550]},
                "properties": {
                    "OBJECTID": 4,
                    "STATION_NA": "DT4",
                    "EXIT_CODE": "Exit 1",
                },
            }
        )

        result = normalise_mrt_geojson(payload)

        self.assertIn("HUME MRT STATION", set(result["station_name"]))
        self.assertEqual(set(result["rail_mode"]), {"MRT"})

    def test_haversine_distance_has_expected_scale(self) -> None:
        distance = float(haversine_distance_m(0.0, 0.0, 1.0, 0.0))

        self.assertAlmostEqual(distance / 1_000.0, 111.195, places=3)

    def test_distance_bands_have_explicit_boundaries(self) -> None:
        bands = assign_mrt_distance_band(
            pd.Series([0.0, 500.0, 500.1, 1_000.0, 1_500.0, 2_001.0])
        ).astype("string")

        self.assertEqual(
            bands.tolist(),
            [
                "0-500 m",
                "0-500 m",
                ">500-1,000 m",
                ">500-1,000 m",
                ">1,000-2,000 m",
                ">2,000 m",
            ],
        )

    def test_finds_nearest_exit_and_joins_transaction_address(self) -> None:
        blocks = pd.DataFrame(
            {
                "address_key": ["1|TEST ROAD"],
                "latitude": [1.3000],
                "longitude": [103.8000],
                "match_status": ["matched"],
            }
        )
        proximity = nearest_mrt_exits(
            blocks,
            normalise_mrt_geojson(station_geojson()),
        )
        result = add_mrt_proximity(transactions(), proximity)

        self.assertEqual(proximity.loc[0, "nearest_mrt_station"], "ALPHA MRT STATION")
        self.assertEqual(proximity.loc[0, "nearest_mrt_distance_m"], 0.0)
        self.assertEqual(result.loc[0, "mrt_distance_band"], "0-500 m")
        self.assertTrue(pd.isna(result.loc[1, "nearest_mrt_distance_m"]))

    def test_rejects_conflicting_coordinates_for_one_address(self) -> None:
        cache = pd.DataFrame(
            {
                "address_key": ["1|TEST ROAD", "1|TEST ROAD"],
                "latitude": [1.30, 1.31],
                "longitude": [103.80, 103.81],
            }
        )

        with self.assertRaisesRegex(ValueError, "conflicting keys"):
            normalise_block_coordinates(cache)

    def test_downloads_and_validates_official_geojson(self) -> None:
        poll_response = Mock()
        poll_response.json.return_value = {
            "code": 0,
            "data": {"url": "https://example.test/mrt.geojson"},
        }
        data_response = Mock()
        data_response.json.return_value = station_geojson()
        session = Mock()
        session.get.side_effect = [poll_response, data_response]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "mrt.geojson"
            with patch.dict(
                os.environ,
                {"DATA_GOV_SG_API_KEY": "test-secret"},
            ):
                row_count = download_mrt_dataset(output_path, session=session)
                saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(row_count, 2)
        self.assertEqual(saved["type"], "FeatureCollection")
        self.assertEqual(session.get.call_count, 2)
        self.assertIn(MRT_DATASET_ID, session.get.call_args_list[0].args[0])
        self.assertEqual(
            session.get.call_args_list[0].kwargs["headers"]["x-api-key"],
            "test-secret",
        )
        self.assertNotIn("headers", session.get.call_args_list[1].kwargs)

    def test_summarises_only_transactions_with_mrt_coverage(self) -> None:
        frame = transactions().assign(
            nearest_mrt_distance_m=[100.0, np.nan],
            mrt_distance_band=["0-500 m", pd.NA],
        )

        summary = mrt_proximity_summary(frame)

        self.assertEqual(summary["transaction_count"].tolist(), [1])
        self.assertEqual(summary["mrt_distance_band"].tolist(), ["0-500 m"])


class EndToEndEnrichmentTests(unittest.TestCase):
    def test_runs_without_onemap_coordinates_and_records_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "clean.csv"
            cpi_path = root / "cpi.csv"
            mrt_path = root / "mrt.geojson"
            output_path = root / "enriched.parquet"
            monthly_path = root / "monthly.csv"
            stations_path = root / "stations.csv"
            summary_path = root / "mrt_summary.csv"
            metadata_path = root / "metadata.json"
            missing_coordinates = root / "block_coordinates.csv"

            transactions().to_csv(input_path, index=False)
            input_sha256 = file_sha256(input_path)
            normalise_cpi_data(cpi_wide()).assign(
                month=lambda frame: frame["month"].dt.strftime("%Y-%m")
            ).to_csv(cpi_path, index=False)
            mrt_path.write_text(json.dumps(station_geojson()), encoding="utf-8")

            metadata = run_enrichment(
                input_path,
                cpi_cache_path=cpi_path,
                mrt_cache_path=mrt_path,
                block_coordinates_path=missing_coordinates,
                output_path=output_path,
                monthly_report_path=monthly_path,
                mrt_stations_report_path=stations_path,
                mrt_summary_path=summary_path,
                metadata_path=metadata_path,
                download_missing=False,
            )
            enriched = pd.read_parquet(output_path)
            summary = pd.read_csv(summary_path)

        self.assertEqual(
            metadata["mrt"]["status"], "unavailable_missing_block_coordinates"
        )
        self.assertEqual(metadata["input"]["sha256"], input_sha256)
        self.assertEqual(metadata["output"]["row_count"], 2)
        self.assertIn("resale_price_real_sgd", enriched.columns)
        self.assertNotIn("nearest_mrt_distance_m", enriched.columns)
        self.assertTrue(summary.empty)
        self.assertTrue(
            any("no block coordinates" in item for item in metadata["limitations"])
        )

    def test_adds_partial_mrt_coverage_when_coordinate_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "clean.csv"
            cpi_path = root / "cpi.csv"
            mrt_path = root / "mrt.geojson"
            coordinates_path = root / "block_coordinates.csv"
            output_path = root / "enriched.parquet"

            transactions().to_csv(input_path, index=False)
            normalise_cpi_data(cpi_wide()).assign(
                month=lambda frame: frame["month"].dt.strftime("%Y-%m")
            ).to_csv(cpi_path, index=False)
            mrt_path.write_text(json.dumps(station_geojson()), encoding="utf-8")
            pd.DataFrame(
                {
                    "address_key": ["1|TEST ROAD"],
                    "latitude": [1.3000],
                    "longitude": [103.8000],
                    "match_status": ["matched"],
                }
            ).to_csv(coordinates_path, index=False)

            metadata = run_enrichment(
                input_path,
                cpi_cache_path=cpi_path,
                mrt_cache_path=mrt_path,
                block_coordinates_path=coordinates_path,
                output_path=output_path,
                monthly_report_path=root / "monthly.csv",
                mrt_stations_report_path=root / "stations.csv",
                mrt_summary_path=root / "mrt_summary.csv",
                metadata_path=root / "metadata.json",
                download_missing=False,
            )
            enriched = pd.read_parquet(output_path)

        self.assertEqual(metadata["mrt"]["status"], "partial")
        self.assertEqual(metadata["mrt"]["matched_rows"], 1)
        self.assertEqual(metadata["mrt"]["coverage_pct"], 50.0)
        self.assertEqual(enriched.loc[0, "nearest_mrt_station"], "ALPHA MRT STATION")
        self.assertTrue(pd.isna(enriched.loc[1, "nearest_mrt_station"]))


if __name__ == "__main__":
    unittest.main()
