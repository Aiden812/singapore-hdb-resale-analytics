"""Offline smoke tests for the standalone HDB Copilot Streamlit interface."""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - supports minimal non-UI test installs
    AppTest = None  # type: ignore[assignment,misc]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CopilotEvalFixtureTests(unittest.TestCase):
    def test_fixture_has_required_breadth_and_structure(self) -> None:
        fixture_path = PROJECT_ROOT / "evals" / "copilot_cases.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        cases = fixture["cases"]

        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(
            {case["category"] for case in cases},
            {
                "normal",
                "sparse",
                "out_of_range",
                "injection",
                "unsupported",
                "numerical_grounding",
            },
        )
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            self.assertIn(
                case["mode"],
                {
                    "market_brief",
                    "comparable_sales",
                    "town_comparison",
                    "model_reliability",
                },
            )
            self.assertIsInstance(case["inputs"], dict)
            self.assertTrue(case["question"].strip())
            self.assertIsInstance(case["expected"], dict)
            self.assertIn("outcome", case["expected"])


@unittest.skipIf(AppTest is None, "Streamlit AppTest is not installed")
class CopilotAppSmokeTests(unittest.TestCase):
    def run_app(self, *, request_limit: str = "12"):
        environment = {
            "OPENAI_API_KEY": "",
            "HDB_COPILOT_MAX_REQUESTS": request_limit,
        }
        with patch.dict(os.environ, environment, clear=False):
            return AppTest.from_file(
                str(PROJECT_ROOT / "copilot_app.py"),
                default_timeout=60,
            ).run(timeout=60)

    @staticmethod
    def visible_text(app) -> str:
        return " ".join(
            str(element.value)
            for element_type in (
                "markdown",
                "caption",
                "info",
                "warning",
                "success",
                "error",
                "subheader",
            )
            for element in app.get(element_type)
            if getattr(element, "value", None) is not None
        )

    def test_initial_render_is_offline_and_does_not_generate(self) -> None:
        app = self.run_app()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [radio.label for radio in app.radio],
            ["Analysis mode"],
        )
        self.assertEqual(
            list(app.radio[0].options),
            [
                "Market brief",
                "Comparable sales",
                "Town comparison",
                "Model reliability",
            ],
        )
        self.assertIn(
            "Build evidence-grounded answer",
            [button.label for button in app.button],
        )
        visible_text = self.visible_text(app)
        self.assertIn("No API key detected", visible_text)
        self.assertIn("deterministic evidence summary", visible_text)
        self.assertIn("not a formal valuation", visible_text)
        self.assertIn("Coverage starts", [metric.label for metric in app.metric])
        self.assertNotIn("Evidence table", visible_text)

    def test_each_mode_exposes_structured_controls(self) -> None:
        app = self.run_app()

        app.radio[0].set_value("Comparable sales")
        app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        selectbox_labels = [selectbox.label for selectbox in app.selectbox]
        number_input_labels = [item.label for item in app.number_input]
        self.assertIn("Town", selectbox_labels)
        self.assertIn("Flat type", selectbox_labels)
        self.assertIn("Recent period", selectbox_labels)
        self.assertIn("Floor area (sqm)", number_input_labels)
        self.assertIn("Remaining lease (years)", number_input_labels)

        app.radio[0].set_value("Town comparison")
        app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        comparison_labels = [selectbox.label for selectbox in app.selectbox]
        self.assertIn("First town", comparison_labels)
        self.assertIn("Second town", comparison_labels)
        self.assertIn("Flat type", comparison_labels)
        self.assertIn("Start year", comparison_labels)
        self.assertIn("End year", comparison_labels)

        app.radio[0].set_value("Model reliability")
        app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "does not generate a price for an individual flat",
            self.visible_text(app),
        )

    def test_submit_uses_deterministic_fallback_without_a_key(self) -> None:
        app = self.run_app(request_limit="1")
        submit = next(
            button
            for button in app.button
            if button.label == "Build evidence-grounded answer"
        )

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "HDB_COPILOT_MAX_REQUESTS": "1"},
            clear=False,
        ):
            submit.click()
            app.run(timeout=60)

        self.assertEqual(list(app.exception), [])
        visible_text = self.visible_text(app)
        self.assertIn("Deterministic evidence summary shown", visible_text)
        self.assertIn("Evidence table", visible_text)
        self.assertIn("Sources", visible_text)
        self.assertIn("reached its request limit", visible_text)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_specialist_modes_submit_and_render_offline(self) -> None:
        app = self.run_app(request_limit="3")

        app.radio[0].set_value("Comparable sales")
        app.run(timeout=60)
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "HDB_COPILOT_MAX_REQUESTS": "3"},
            clear=False,
        ):
            next(
                button
                for button in app.button
                if button.label == "Build evidence-grounded answer"
            ).click()
            app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        comparable_text = self.visible_text(app)
        self.assertIn("Deterministic evidence summary shown", comparable_text)
        self.assertIn("Evidence table", comparable_text)

        app.radio[0].set_value("Town comparison")
        app.run(timeout=60)
        self.assertNotIn("Evidence table", self.visible_text(app))
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "HDB_COPILOT_MAX_REQUESTS": "3"},
            clear=False,
        ):
            next(
                button
                for button in app.button
                if button.label == "Build evidence-grounded answer"
            ).click()
            app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        comparison_text = self.visible_text(app)
        self.assertIn("Deterministic evidence summary shown", comparison_text)
        self.assertIn("Town with highest median resale price", comparison_text)

        app.radio[0].set_value("Model reliability")
        app.run(timeout=60)
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "HDB_COPILOT_MAX_REQUESTS": "3"},
            clear=False,
        ):
            next(
                button
                for button in app.button
                if button.label == "Build evidence-grounded answer"
            ).click()
            app.run(timeout=60)
        self.assertEqual(list(app.exception), [])
        reliability_text = self.visible_text(app)
        self.assertIn("Deterministic evidence summary shown", reliability_text)
        self.assertIn("Latest holdout mean absolute error", reliability_text)
        result = app.session_state["copilot_result"]
        provenance = result["evidence"]["snapshot"]["provenance"]
        metrics_path = PROJECT_ROOT / "reports" / "price_model_metrics.json"
        actual_metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        self.assertEqual(provenance["artifact"], "reports/price_model_metrics.json")
        self.assertEqual(provenance["artifact_sha256"], actual_metrics_hash)
        self.assertNotEqual(
            provenance["artifact_sha256"],
            provenance["transaction_snapshot_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
