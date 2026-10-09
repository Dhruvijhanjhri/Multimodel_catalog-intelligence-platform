
import importlib
import unittest
from unittest.mock import patch

import numpy as np


class FakeIndex:
    """Minimal FAISS index substitute for testing module initialization."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.ntotal = 0

    def add(self, vectors):
        self.ntotal = len(vectors)


def fake_load(path):
    """Return small, controlled arrays instead of reading real embeddings."""
    path = str(path)

    if "train_image_embeddings.npy" in path:
        return np.ones((6, 2), dtype=np.float32)

    if "train_text_embeddings.npy" in path:
        return np.ones((6, 2), dtype=np.float32)

    if "train_labels.npy" in path:
        return np.arange(6)

    raise ValueError(f"Unexpected embedding file: {path}")


class TestTaxonomyStatus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with (
            patch("numpy.load", side_effect=fake_load),
            patch("faiss.IndexFlatIP", FakeIndex),
        ):
            cls.validator = importlib.import_module(
                "services.inference.taxonomy_validator"
            )

    def test_confident_prediction_with_clear_margin_passes(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.90,
            support_margin=0.20,
            mismatch=False,
        )

        self.assertEqual(result, "PASS")

    def test_low_confidence_requires_review(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.69,
            support_margin=0.20,
            mismatch=False,
        )

        self.assertEqual(result, "REVIEW")

    def test_low_support_margin_requires_review(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.90,
            support_margin=0.09,
            mismatch=False,
        )

        self.assertEqual(result, "REVIEW")

    def test_image_text_mismatch_requires_review(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.90,
            support_margin=0.20,
            mismatch=True,
        )

        self.assertEqual(result, "REVIEW")

    def test_exact_thresholds_pass_when_no_mismatch(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.70,
            support_margin=0.10,
            mismatch=False,
        )

        self.assertEqual(result, "PASS")

    def test_all_uncertainty_signals_still_return_review(self):
        result = self.validator.determine_taxonomy_status(
            confidence=0.50,
            support_margin=0.02,
            mismatch=True,
        )

        self.assertEqual(result, "REVIEW")


if __name__ == "__main__":
    unittest.main()
