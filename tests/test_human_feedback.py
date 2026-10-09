
import unittest
from unittest.mock import patch

from services.monitoring.human_feedback import (
    evaluate_human_feedback_quality,
)


def make_row(
    prediction_id=1,
    predicted_category="Footwear",
    decision="Approved",
    corrected_category=None,
):
    return (
        prediction_id,
        predicted_category,
        "multimodal_classifier_v1",
        decision,
        corrected_category,
        "2026-10-07",
    )


class TestHumanFeedbackQuality(unittest.TestCase):

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_approved_prediction_counts_as_correct(self, mock_data):
        mock_data.return_value = [
            make_row(decision="Approved")
        ]

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["reviewed_predictions"], 1)
        self.assertEqual(result["validated_predictions"], 1)
        self.assertEqual(result["validated_correct"], 1)
        self.assertEqual(result["validated_accuracy"], 1.0)
        self.assertEqual(result["corrections"], 0)

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_corrected_rejection_counts_as_correction(self, mock_data):
        mock_data.return_value = [
            make_row(
                decision="Rejected",
                corrected_category="Furniture",
            )
        ]

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["validated_predictions"], 1)
        self.assertEqual(result["validated_correct"], 0)
        self.assertEqual(result["corrections"], 1)
        self.assertEqual(result["validated_accuracy"], 0.0)
        self.assertEqual(result["correction_rate"], 1.0)

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_unresolved_rejection_is_excluded_from_accuracy(self, mock_data):
        mock_data.return_value = [
            make_row(decision="Rejected", corrected_category=None)
        ]

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["reviewed_predictions"], 1)
        self.assertEqual(result["validated_predictions"], 0)
        self.assertEqual(result["unresolved_reviews"], 1)
        self.assertIsNone(result["validated_accuracy"])
        self.assertIsNone(result["correction_rate"])

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_insufficient_data_below_minimum(self, mock_data):
        mock_data.return_value = [
            make_row(prediction_id=i)
            for i in range(29)
        ]

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["validated_predictions"], 29)
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_ready_at_minimum_validated_sample_count(self, mock_data):
        mock_data.return_value = [
            make_row(prediction_id=i)
            for i in range(30)
        ]

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["validated_predictions"], 30)
        self.assertEqual(result["status"], "READY")

    @patch("services.monitoring.human_feedback.get_human_feedback_quality")
    def test_empty_dataset(self, mock_data):
        mock_data.return_value = []

        result = evaluate_human_feedback_quality()

        self.assertEqual(result["reviewed_predictions"], 0)
        self.assertEqual(result["validated_predictions"], 0)
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["validated_accuracy"])


if __name__ == "__main__":
    unittest.main()
