"""Contract and real-data offline tests for the HDB copilot eval catalogue."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_copilot import (
    DEFAULT_FIXTURE,
    DEFAULT_MODEL_METRICS,
    DEFAULT_SNAPSHOT,
    MINIMUM_CASES,
    REQUIRED_CATEGORIES,
    SUPPORTED_MODES,
    FixtureContractError,
    build_case_evidence,
    evaluate_case,
    evaluate_fixture,
    load_fixture,
    load_resources,
    main,
    validate_fixture,
)
from src.copilot import EvidencePoint, generate_answer


class CopilotEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_fixture(DEFAULT_FIXTURE)
        cls.cases = validate_fixture(cls.payload)

    def test_checked_in_fixture_has_full_unique_case_contract(self) -> None:
        self.assertGreaterEqual(len(self.cases), MINIMUM_CASES)
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            {case["category"] for case in self.cases}, REQUIRED_CATEGORIES
        )
        self.assertLessEqual({case["mode"] for case in self.cases}, SUPPORTED_MODES)
        for case in self.cases:
            self.assertTrue(case["question"].strip())
            self.assertIsInstance(case["expected"], dict)

    def test_contract_rejects_duplicate_ids(self) -> None:
        payload = deepcopy(self.payload)
        payload["cases"][1]["id"] = payload["cases"][0]["id"]
        with self.assertRaisesRegex(FixtureContractError, "Duplicate case id"):
            validate_fixture(payload)

    def test_contract_rejects_unsupported_mode(self) -> None:
        payload = deepcopy(self.payload)
        payload["cases"][0]["mode"] = "free_form_agent"
        with self.assertRaisesRegex(FixtureContractError, "unsupported mode"):
            validate_fixture(payload)

    def test_contract_rejects_blank_question_and_nonobject_expected(self) -> None:
        for field, value, message in (
            ("question", "  ", "blank question"),
            ("expected", [], "expected must be an object"),
        ):
            with self.subTest(field=field):
                payload = deepcopy(self.payload)
                payload["cases"][0][field] = value
                with self.assertRaisesRegex(FixtureContractError, message):
                    validate_fixture(payload)

    def test_contract_requires_every_safety_category(self) -> None:
        payload = deepcopy(self.payload)
        payload["cases"] = [
            case for case in payload["cases"] if case["category"] != "injection"
        ]
        with self.assertRaisesRegex(FixtureContractError, "missing required categories"):
            validate_fixture(payload)

    def test_contract_rejects_invalid_global_assertions(self) -> None:
        payload = deepcopy(self.payload)
        payload["global_assertions"]["maximum_unsupported_numbers"] = -1
        with self.assertRaisesRegex(
            FixtureContractError, "maximum_unsupported_numbers"
        ):
            validate_fixture(payload)

        payload = deepcopy(self.payload)
        del payload["global_assertions"]["must_disclose_snapshot_period"]
        with self.assertRaisesRegex(
            FixtureContractError, "must_disclose_snapshot_period"
        ):
            validate_fixture(payload)

    def test_contract_rejects_invalid_case_policy_fields(self) -> None:
        payload = deepcopy(self.payload)
        grounding = next(
            case for case in payload["cases"] if case["id"] == "grounding_market_median"
        )
        grounding["expected"]["maximum_unsupported_numbers"] = -1
        with self.assertRaisesRegex(
            FixtureContractError, "maximum_unsupported_numbers"
        ):
            validate_fixture(payload)

        payload = deepcopy(self.payload)
        injection = next(
            case for case in payload["cases"] if case["id"] == "injection_system_prompt"
        )
        injection["expected"]["must_ignore_instruction"] = "yes"
        with self.assertRaisesRegex(FixtureContractError, "must_ignore_instruction"):
            validate_fixture(payload)


class CopilotRealDataEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_fixture(DEFAULT_FIXTURE)
        cls.cases = validate_fixture(cls.payload)
        cls.data, cls.metrics = load_resources(
            DEFAULT_SNAPSHOT, DEFAULT_MODEL_METRICS
        )
        cls.summary = evaluate_fixture(cls.payload)

    @classmethod
    def case(cls, case_id: str) -> dict[str, object]:
        return next(case for case in cls.cases if case["id"] == case_id)

    def run_case(self, case_id: str) -> dict[str, object]:
        return evaluate_case(
            self.case(case_id),
            self.data,
            self.metrics,
            snapshot_path=DEFAULT_SNAPSHOT,
        )

    def test_complete_fixture_passes_without_api_access(self) -> None:
        self.assertEqual(self.summary["status"], "pass", self.summary["failures"])
        self.assertTrue(self.summary["offline"])
        self.assertEqual(self.summary["total"], len(self.cases))
        self.assertEqual(self.summary["passed"], len(self.cases))
        self.assertEqual(self.summary["failed"], 0)

    def test_existing_api_key_is_never_used_and_is_restored(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-be-used"}):
            result = self.run_case("normal_market_tampines_4room")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "must-not-be-used")
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["generation_mode"], "deterministic_fallback")

    def test_negative_area_is_validation_error_without_model_call(self) -> None:
        result = self.run_case("range_negative_floor_area")

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["observed_outcome"], "validation_error")
        self.assertFalse(result["model_called"])
        self.assertEqual(result["generation_mode"], "not_called")
        self.assertIn("positive", result["validation_error"])

    def test_future_no_data_differs_from_normal_real_selection(self) -> None:
        future = self.run_case("range_future_only")
        normal = self.run_case("normal_market_tampines_4room")

        self.assertTrue(future["passed"], future["failures"])
        self.assertTrue(normal["passed"], normal["failures"])
        self.assertEqual(future["observed_outcome"], "no_data")
        self.assertEqual(future["transaction_count"], 0)
        self.assertEqual(normal["observed_outcome"], "grounded")
        self.assertGreater(normal["transaction_count"], 0)
        self.assertNotEqual(future["evidence_digest"], normal["evidence_digest"])

    def test_numerical_case_requires_real_tolerance_surfaces(self) -> None:
        result = self.run_case("grounding_matching_tolerances")

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["observed_outcome"], "grounded")
        self.assertTrue(
            {
                "effective_months",
                "floor_area_tolerance",
                "storey_tolerance",
                "lease_tolerance",
            }.issubset(result["checked_surfaces"])
        )

    def test_tampered_expected_surface_fails(self) -> None:
        case = deepcopy(self.case("normal_market_tampines_4room"))
        case["expected"]["required_evidence"].append("invented_metric")

        result = evaluate_case(
            case,
            self.data,
            self.metrics,
            snapshot_path=DEFAULT_SNAPSHOT,
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("invented_metric" in failure for failure in result["failures"])
        )

    def test_injected_number_does_not_leak_into_fallback(self) -> None:
        result = self.run_case("injection_town_comparison_fake_gap")

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["generation_mode"], "deterministic_fallback")

    def test_all_enabled_global_assertions_are_executed(self) -> None:
        result = self.run_case("normal_market_tampines_4room")
        expected = {
            "maximum_unsupported_numbers",
            *(
                name
                for name, enabled in self.payload["global_assertions"].items()
                if name != "maximum_unsupported_numbers" and enabled
            ),
        }

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(set(result["checked_global_assertions"]), expected)

    def test_unsupported_number_limits_are_enforced_at_case_and_global_level(self) -> None:
        case = deepcopy(self.case("normal_market_tampines_4room"))
        case["expected"]["maximum_unsupported_numbers"] = 0
        evidence = build_case_evidence(case, self.data, self.metrics)
        with patch.dict(os.environ, {}, clear=True):
            safe_answer = generate_answer(case["question"], evidence)
        unsafe_answer = safe_answer.model_copy(
            update={"summary": safe_answer.summary + " Unsupported code 987654321."}
        )

        with patch(
            "scripts.evaluate_copilot.generate_answer", return_value=unsafe_answer
        ):
            result = evaluate_case(case, self.data, self.metrics)

        self.assertFalse(result["passed"])
        self.assertIn("987654321", result["unsupported_output_numbers"])
        self.assertTrue(
            any("case unsupported-number limit exceeded" in item for item in result["failures"])
        )
        self.assertTrue(
            any("global unsupported-number limit exceeded" in item for item in result["failures"])
        )

    def test_instruction_ignore_policy_rejects_an_echoed_injection(self) -> None:
        case = deepcopy(self.case("injection_system_prompt"))
        evidence = build_case_evidence(case, self.data, self.metrics)
        with patch.dict(os.environ, {}, clear=True):
            safe_answer = generate_answer(case["question"], evidence)
        unsafe_answer = safe_answer.model_copy(update={"summary": case["question"]})

        with patch(
            "scripts.evaluate_copilot.generate_answer", return_value=unsafe_answer
        ):
            result = evaluate_case(case, self.data, self.metrics)

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("instruction" in item.lower() for item in result["failures"])
        )

    def test_canonical_renderer_rejects_an_appended_advice_paraphrase(self) -> None:
        case = deepcopy(self.case("injection_role_override"))
        evidence = build_case_evidence(case, self.data, self.metrics)
        with patch.dict(os.environ, {}, clear=True):
            safe_answer = generate_answer(case["question"], evidence)
        unsafe_answer = safe_answer.model_copy(
            update={"summary": safe_answer.summary + " Buying now looks attractive."}
        )

        with patch(
            "scripts.evaluate_copilot.generate_answer", return_value=unsafe_answer
        ):
            result = evaluate_case(case, self.data, self.metrics)

        self.assertFalse(result["passed"])
        self.assertIn(
            "offline output differed from the canonical deterministic renderer",
            result["failures"],
        )

    def test_answer_surface_requires_the_value_not_only_a_citation(self) -> None:
        case = deepcopy(self.case("normal_market_tampines_4room"))
        evidence = build_case_evidence(case, self.data, self.metrics)
        with patch.dict(os.environ, {}, clear=True):
            safe_answer = generate_answer(case["question"], evidence)
        points = [
            EvidencePoint(
                text="The selected median-price fact is available.",
                evidence_ids=point.evidence_ids,
            )
            if "fact.market.median_resale_price" in point.evidence_ids
            else point
            for point in safe_answer.points
        ]
        incomplete_answer = safe_answer.model_copy(update={"points": points})

        with patch(
            "scripts.evaluate_copilot.generate_answer",
            return_value=incomplete_answer,
        ):
            result = evaluate_case(case, self.data, self.metrics)

        self.assertFalse(result["passed"])
        self.assertIn(
            "returned answer omitted required_evidence surface: median_price",
            result["failures"],
        )

    def test_global_snapshot_period_must_appear_in_the_answer(self) -> None:
        case = deepcopy(self.case("normal_market_tampines_4room"))
        evidence = build_case_evidence(case, self.data, self.metrics)
        with patch.dict(os.environ, {}, clear=True):
            safe_answer = generate_answer(case["question"], evidence)
        summary_without_coverage = safe_answer.summary.split(" Snapshot coverage:")[0]
        incomplete_answer = safe_answer.model_copy(
            update={"summary": summary_without_coverage}
        )

        with patch(
            "scripts.evaluate_copilot.generate_answer",
            return_value=incomplete_answer,
        ):
            result = evaluate_case(case, self.data, self.metrics)

        self.assertFalse(result["passed"])
        self.assertIn(
            "global snapshot-period disclosure is missing from answer",
            result["failures"],
        )

    def test_custom_model_report_path_is_preserved_in_provenance(self) -> None:
        case = deepcopy(self.case("normal_reliability_global"))
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "custom_metrics.json"
            metrics_path.write_text(
                json.dumps(self.metrics, ensure_ascii=False),
                encoding="utf-8",
            )
            evidence = build_case_evidence(
                case,
                self.data,
                self.metrics,
                metrics_path=metrics_path,
            )
            expected_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()

        provenance = evidence["snapshot"]["provenance"]
        metrics_source = next(
            source
            for source in evidence["sources"]
            if source["id"] == "source.price_model_metrics"
        )
        self.assertEqual(provenance["artifact"], str(metrics_path.resolve()))
        self.assertEqual(provenance["artifact_sha256"], expected_hash)
        self.assertEqual(metrics_source["path"], str(metrics_path.resolve()))

    def test_json_cli_is_machine_readable(self) -> None:
        fake_summary = {
            "status": "pass",
            "offline": True,
            "total": 40,
            "passed": 40,
            "failed": 0,
            "failures": [],
        }
        output = io.StringIO()
        with (
            patch(
                "scripts.evaluate_copilot.evaluate_fixture",
                return_value=fake_summary,
            ),
            redirect_stdout(output),
        ):
            exit_code = main(["--fixture", str(DEFAULT_FIXTURE), "--json"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["failed"], 0)

    def test_cli_returns_nonzero_for_invalid_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(
                '{"global_assertions": {}, "cases": []}',
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["--fixture", str(path), "--json"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["failures"][0]["id"], "evaluation_setup")


if __name__ == "__main__":
    unittest.main()
