
import importlib
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient


class FakeTokens:
    def to(self, device):
        return self


class FakeClipModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def encode_text(self, tokens):
        return torch.ones((1, 512), dtype=torch.float32)


class FakeFaissIndex:
    def search(self, query, k):
        return (
            np.array([[0.10]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )


def fake_numpy_load(path, *args, **kwargs):
    path = str(path)

    if "text_embeddings.npy" in path:
        return np.zeros((2, 512), dtype=np.float32)

    return np.zeros((2, 2), dtype=np.float32)


def load_api_module():
    """Import the API without loading real ML models or database assets."""

    fake_predict_module = types.ModuleType(
        "services.inference.predict"
    )
    fake_predict_module.predict = MagicMock()
    fake_predict_module.MODEL_VERSION = "test_model"

    fake_taxonomy_module = types.ModuleType(
        "services.inference.taxonomy_validator"
    )
    fake_taxonomy_module.validate_taxonomy = MagicMock()
    fake_taxonomy_module.determine_taxonomy_status = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {
                "services.inference.predict": fake_predict_module,
                "services.inference.taxonomy_validator": (
                    fake_taxonomy_module
                ),
            },
        ),
        patch("joblib.load", return_value=MagicMock()),
        patch("numpy.load", side_effect=fake_numpy_load),
        patch(
            "pandas.read_parquet",
            return_value=pd.DataFrame(),
        ),
        patch(
            "faiss.read_index",
            return_value=FakeFaissIndex(),
        ),
        patch(
            "open_clip.create_model_and_transforms",
            return_value=(FakeClipModel(), None, None),
        ),
        patch(
            "open_clip.get_tokenizer",
            return_value=lambda texts: FakeTokens(),
        ),
    ):
        sys.modules.pop("services.api.main", None)
        api = importlib.import_module("services.api.main")

    return api


class TestPredictAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api = load_api_module()
        cls.client = TestClient(cls.api.app)

    def setUp(self):
        self.uploaded_paths = []

        self.api.predict.reset_mock()
        self.api.validate_taxonomy.reset_mock()
        self.api.determine_taxonomy_status.reset_mock()

        self.api.predict.return_value = {
            "category": "Footwear",
            "confidence": 0.95,
            "image_title_similarity": 0.82,
            "mismatch": False,
            "probabilities": [
                0.01, 0.01, 0.95, 0.01, 0.01, 0.01
            ],
            "image_embedding": np.ones(512, dtype=np.float32),
            "text_embedding": np.ones(512, dtype=np.float32),
        }

        self.api.validate_taxonomy.return_value = {
            "support_margin": 0.25,
            "catalog_support": 0.90,
            "best_alternative_category": "Fashion_Travel",
            "best_alternative_support": 0.30,
        }

        self.api.determine_taxonomy_status.return_value = "PASS"

    def tearDown(self):
        for path in self.uploaded_paths:
            if path.exists():
                path.unlink()

    def post_prediction(self):
        response = self.client.post(
            "/predict",
            data={"title": "Running shoes"},
            files={
                "image": (
                    "test_shoes.jpg",
                    b"mock image content",
                    "image/jpeg",
                )
            },
        )

        if response.status_code == 200:
            image_name = response.json().get("image_name")
            if image_name:
                self.uploaded_paths.append(
                    self.api.PROJECT_ROOT / "uploads" / image_name
                )

        return response

    def test_prediction_returns_expected_fields(self):
        with patch.object(
            self.api,
            "add_to_review_queue",
        ) as add_review:
            response = self.post_prediction()

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["category"], "Footwear")
        self.assertEqual(data["confidence"], 0.95)
        self.assertEqual(data["taxonomy_status"], "PASS")
        self.assertIn("taxonomy_support", data)
        self.assertIn("taxonomy_margin", data)
        self.assertIn("duplicate_score", data)
        self.assertIn("decision", data)
        self.assertIn("review_reasons", data)

        self.assertNotIn("image_embedding", data)
        self.assertNotIn("text_embedding", data)

        add_review.assert_not_called()

    def test_uncertain_prediction_is_added_to_review_queue(self):
        self.api.predict.return_value["confidence"] = 0.40

        with (
            patch.object(
                self.api,
                "evaluate_decision",
                return_value={
                    "decision": "REVIEW",
                    "reasons": ["low_confidence"],
                },
            ),
            patch.object(
                self.api,
                "add_to_review_queue",
            ) as add_review,
        ):
            response = self.post_prediction()

        self.assertEqual(response.status_code, 200)

        data = response.json()

        self.assertEqual(data["decision"], "REVIEW")
        self.assertIn("low_confidence", data["review_reasons"])
        add_review.assert_called_once()

        queue_item = add_review.call_args.kwargs
        self.assertEqual(
            queue_item["predicted_category"],
            "Footwear",
        )
        self.assertIn(
            "low_confidence",
            queue_item["reason"],
        )

    def test_duplicate_check_failure_does_not_crash_prediction(self):
        with (
            patch.object(
                self.api.clip_model,
                "encode_text",
                side_effect=RuntimeError("simulated failure"),
            ),
            patch.object(
                self.api,
                "add_to_review_queue",
            ),
        ):
            response = self.post_prediction()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["duplicate_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
