"""Unit tests for the optional OneMap block geocoder."""

from __future__ import annotations

import unittest

from src.geocode_blocks import normalise_address, parse_search_payload


class GeocodeBlockTests(unittest.TestCase):
    def test_normalises_an_hdb_address(self) -> None:
        key, search_value = normalise_address(" 123a ", " Ang Mo Kio Ave 3 ")

        self.assertEqual(key, "123A|ANG MO KIO AVE 3")
        self.assertEqual(search_value, "123A ANG MO KIO AVE 3 SINGAPORE")

    def test_parses_the_highest_ranked_match(self) -> None:
        payload = {
            "found": 1,
            "results": [
                {
                    "ADDRESS": "123A TEST ROAD SINGAPORE 123456",
                    "POSTAL": "123456",
                    "LATITUDE": "1.3000",
                    "LONGITUDE": "103.8000",
                }
            ],
        }

        record = parse_search_payload(
            payload,
            address_key="123A|TEST ROAD",
            search_address="123A TEST ROAD SINGAPORE",
        )

        self.assertEqual(record["match_status"], "matched")
        self.assertEqual(record["postal_code"], "123456")
        self.assertEqual(record["latitude"], 1.3)

    def test_marks_an_address_without_a_match(self) -> None:
        record = parse_search_payload(
            {"found": 0, "results": []},
            address_key="1|UNKNOWN ROAD",
            search_address="1 UNKNOWN ROAD SINGAPORE",
        )

        self.assertEqual(record["match_status"], "not_found")

    def test_rejects_authentication_errors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "token missing"):
            parse_search_payload(
                {"error": "Authentication token missing", "results": []},
                address_key="1|TEST ROAD",
                search_address="1 TEST ROAD SINGAPORE",
            )


if __name__ == "__main__":
    unittest.main()
