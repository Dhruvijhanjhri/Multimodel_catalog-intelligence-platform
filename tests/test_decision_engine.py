
import unittest

from services.inference.decision_engine import evaluate_decision


class TestCatalogDecisionEngine(unittest.TestCase):

    def test_clear_prediction_passes(self):
        result = evaluate_decision(
            confidence=0.95,
            mismatch=False,
            taxonomy_status="PASS",
            duplicate_score=0.10,
        )

        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["reasons"], [])

    def test_low_confidence_triggers_review(self):
        result = evaluate_decision(
            confidence=0.69,
            mismatch=False,
            taxonomy_status="PASS",
            duplicate_score=0.10,
        )

        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("Low Confidence", result["reasons"])

    def test_confidence_at_threshold_does_not_trigger_review(self):
        result = evaluate_decision(
            confidence=0.70,
            mismatch=False,
            taxonomy_status="PASS",
            duplicate_score=0.10,
        )

        self.assertEqual(result["decision"], "PASS")

    def test_image_text_mismatch_triggers_review(self):
        result = evaluate_decision(
            confidence=0.95,
            mismatch=True,
            taxonomy_status="PASS",
            duplicate_score=0.10,
        )

        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("Image-Text Mismatch", result["reasons"])

    def test_taxonomy_uncertainty_triggers_review(self):
        result = evaluate_decision(
            confidence=0.95,
            mismatch=False,
            taxonomy_status="REVIEW",
            duplicate_score=0.10,
        )

        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("Taxonomy Uncertainty", result["reasons"])

    def test_duplicate_score_above_threshold_triggers_review(self):
        result = evaluate_decision(
            confidence=0.95,
            mismatch=False,
            taxonomy_status="PASS",
            duplicate_score=0.91,
        )

        self.assertEqual(result["decision"], "REVIEW")
        self.assertIn("Possible Duplicate", result["reasons"])

    def test_duplicate_score_at_threshold_does_not_trigger_review(self):
        result = evaluate_decision(
            confidence=0.95,
            mismatch=False,
            taxonomy_status="PASS",
            duplicate_score=0.90,
        )

        self.assertEqual(result["decision"], "PASS")

    def test_multiple_signals_are_preserved_in_order(self):
        result = evaluate_decision(
            confidence=0.60,
            mismatch=True,
            taxonomy_status="REVIEW",
            duplicate_score=0.95,
        )

        self.assertEqual(result["decision"], "REVIEW")
        self.assertEqual(
            result["reasons"],
            [
                "Low Confidence",
                "Image-Text Mismatch",
                "Taxonomy Uncertainty",
                "Possible Duplicate",
            ],
        )


if __name__ == "__main__":
    unittest.main()
