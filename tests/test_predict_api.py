import importlib
import sys
import types
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient
from pathlib import Path

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

    def post_prediction(self, enqueue_review=None):
        form_data = {"title": "Running shoes"}

        if enqueue_review is not None:
            form_data["enqueue_review"] = str(enqueue_review).lower()

        response = self.client.post(
            "/predict",
            data=form_data,
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

    def test_seller_analysis_opt_out_skips_review_queue(self):
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
            response = self.post_prediction(enqueue_review=False)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "REVIEW")
        add_review.assert_not_called()

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

    def test_missing_image_returns_validation_error(self):
        response = self.client.post(
            "/predict",
            data={"title": "Running shoes"},
        )

        self.assertEqual(response.status_code, 422)

    def test_approve_records_exact_prediction_id(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = (
            "seller-item-1",
            "Pending",
            101,
            202,
        )
        cursor.rowcount = 1

        with (
            patch.object(self.api, "get_connection", return_value=conn),
            patch.object(self.api, "add_product_review"),
        ):
            response = self.client.put("/review-queue/1/approve")


        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT INTO product_reviews" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)

        saved_values = insert_calls[0].args[1]
        self.assertEqual(saved_values[0], 101)  # product_id
        self.assertEqual(saved_values[1], 202)  # exact prediction_id
        self.assertEqual(saved_values[2], "admin")
        self.assertEqual(saved_values[3], "Approved")

        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    def test_reject_records_exact_prediction_id(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = (
            "seller-item-1",
            "Pending",
            101,
            202,
        )
        cursor.rowcount = 1

        with patch.object(
            self.api,
            "get_connection",
            return_value=conn,
        ):
            response = self.client.put(
                "/review-queue/1/reject",
                params={
                    "corrected_category": "Footwear",
                    "feedback": "Incorrect category",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT INTO product_reviews" in c.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)

        saved_values = insert_calls[0].args[1]
        self.assertEqual(saved_values[0], 101)
        self.assertEqual(saved_values[1], 202)
        self.assertEqual(saved_values[2], "admin")
        self.assertEqual(saved_values[3], "Rejected")
        self.assertEqual(saved_values[4], "Footwear")
        self.assertEqual(saved_values[5], "Incorrect category")

        conn.commit.assert_called_once()
        conn.close.assert_called_once()


    def test_processed_review_is_not_recorded_twice(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = (
            "seller-item-1",
            "Approved",
            101,
            202,
        )

        with patch.object(
            self.api,
            "get_connection",
            return_value=conn,
        ):
            response = self.client.put("/review-queue/1/approve")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT INTO product_reviews" in c.args[0]
        ]
        self.assertEqual(insert_calls, [])

        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()

    def test_missing_title_returns_validation_error(self):
        response = self.client.post(
            "/predict",
            files={
                "image": (
                    "test_shoes.jpg",
                    b"mock image content",
                    "image/jpeg",
                )
            },
        )

        self.assertEqual(response.status_code, 422)


    def test_seller_product_save_commits_product_image_and_prediction(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [(123,), (456,)]

        payload = {
            "title": "Running shoes",
            "brand": "Test Brand",
            "description": "Lightweight running shoes",
            "image_name": "shoe.jpg",
            "category": "Footwear",
            "confidence": 0.95,
            "image_title_similarity": 0.82,
            "mismatch": False,
            "duplicate_score": 0.0,
            "taxonomy_status": "PASS",
            "taxonomy_margin": 0.25,
            "model_version": "test_model",
        }

        with (
            patch.object(self.api, "get_connection", return_value=conn),
            patch.object(
                self.api,
                "evaluate_decision",
                return_value={"decision": "APPROVE", "reasons": []},
            ),
            patch.object(self.api, "add_to_review_queue") as add_review,
        ):
            response = self.client.post("/seller/products", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["product_id"], 123)

        insert_calls = [
            call for call in cursor.execute.call_args_list
            if "INSERT INTO" in call.args[0]
        ]
        self.assertEqual(len(insert_calls), 3)
        self.assertIn("INSERT INTO products", insert_calls[0].args[0])
        self.assertIn("INSERT INTO product_images", insert_calls[1].args[0])
        self.assertIn("INSERT INTO product_predictions", insert_calls[2].args[0])

        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        conn.close.assert_called_once()
        add_review.assert_not_called()

    def test_seller_product_with_review_reasons_is_queued(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [(123,), (456,)]

        payload = {
            "title": "Running shoes",
            "image_name": "shoe.jpg",
            "category": "Footwear",
            "confidence": 0.40,
            "image_title_similarity": 0.42,
            "mismatch": False,
            "duplicate_score": 0.0,
            "taxonomy_status": "PASS",
        }

        with (
            patch.object(self.api, "get_connection", return_value=conn),
            patch.object(
                self.api,
                "evaluate_decision",
                return_value={
                    "decision": "REVIEW",
                    "reasons": ["low_confidence"],
                },
            ),
            patch.object(self.api, "add_to_review_queue") as add_review,
        ):
            response = self.client.post("/seller/products", json=payload)

        self.assertEqual(response.status_code, 200)
        add_review.assert_called_once()

        queue_item = add_review.call_args.kwargs
        self.assertEqual(queue_item["prediction_id"], 456)
        self.assertEqual(queue_item["item_id"], response.json().get("item_id", queue_item["item_id"]))
        self.assertIn("low_confidence", queue_item["reason"])
        self.assertIs(queue_item["conn"], conn)

        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        conn.close.assert_called_once()

    def test_seller_product_database_failure_rolls_back(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.execute.side_effect = RuntimeError("simulated database failure")

        payload = self.api.SellerProductRequest(
            title="Running shoes",
            image_name="shoe.jpg",
            category="Footwear",
            confidence=0.95,
            image_title_similarity=0.82,
            mismatch=False,
            duplicate_score=0.0,
        )

        with (
            patch.object(self.api, "get_connection", return_value=conn),
            self.assertRaises(RuntimeError),
        ):
            self.api.create_seller_product(payload)

        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()
        conn.close.assert_called_once()


    def test_bulk_upload_rejects_csv_with_missing_columns(self):
        with (
            patch.object(
                self.api,
                "create_ingestion_event",
                return_value="test-event-id",
            ),
            patch.object(
                self.api,
                "update_ingestion_event",
            ) as update_event,
        ):
            response = self.client.post(
                "/seller/bulk-products",
                files=[
                    (
                        "catalog",
                        (
                            "catalog.csv",
                            b"product_id,title\np1,Running shoes\n",
                            "text/csv",
                        ),
                    ),
                    (
                        "images",
                        ("shoe.jpg", b"mock image content", "image/jpeg"),
                    ),
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])
        self.assertIn(
            "product_id, title, brand and image",
            response.json()["message"],
        )

        update_event.assert_called_once()
        self.assertEqual(
            update_event.call_args.kwargs["status"],
            "Failed",
        )


    def test_bulk_upload_skips_existing_product(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_root = Path(temp_dir)

            conn = MagicMock()
            cursor = MagicMock()
            conn.__enter__.return_value = conn
            conn.cursor.return_value.__enter__.return_value = cursor
            cursor.fetchone.return_value = (123,)

            with (
                patch.object(
                    self.api,
                    "BULK_UPLOAD_DIR",
                    batch_root,
                ),
                patch.object(
                    self.api,
                    "create_ingestion_event",
                    return_value="test-event-id",
                ),
                patch.object(
                    self.api,
                    "update_ingestion_event",
                ) as update_event,
                patch.object(
                    self.api,
                    "get_connection",
                    return_value=conn,
                ),
                patch.object(
                    self.api,
                    "create_seller_product",
                ) as save_product,

            ):
                response = self.client.post(
                    "/seller/bulk-products",
                    files=[
                        (
                            "catalog",
                            (
                                "catalog.csv",
                                (
                                    "product_id,title,brand,image\n"
                                    "existing-1,Running shoes,Test Brand,shoe.jpg\n"
                                ).encode(),
                                "text/csv",
                            ),
                        ),
                        (
                            "images",
                            ("shoe.jpg", b"mock image content", "image/jpeg"),
                        ),
                    ],
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["successful_rows"], 0)
        self.assertEqual(data["failed_rows"], 0)
        self.assertEqual(data["skipped_rows"], 1)
        self.assertTrue(data["results"][0]["skipped"])
        self.assertEqual(data["results"][0]["database_product_id"], 123)

        save_product.assert_not_called()
        self.api.predict.assert_not_called()
        update_event.assert_called_once()


    def test_bulk_upload_continues_when_prediction_fails_for_one_row(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_root = Path(temp_dir)

            conn = MagicMock()
            conn.__enter__.return_value = conn
            cursor = MagicMock()
            conn.cursor.return_value.__enter__.return_value = cursor
            cursor.fetchone.return_value = None

            prediction_result = {
                "category": "Footwear",
                "confidence": 0.95,
                "image_title_similarity": 0.90,
                "mismatch": False,
                "image_embedding": MagicMock(),
                "text_embedding": MagicMock(),
            }

            with (
                patch.object(self.api, "BULK_UPLOAD_DIR", batch_root),
                patch.object(
                    self.api,
                    "create_ingestion_event",
                    return_value="test-event-id",
                ),
                patch.object(self.api, "update_ingestion_event") as update_event,
                patch.object(
                    self.api,
                    "get_connection",
                    return_value=conn,
                ),
                patch.object(
                    self.api,
                    "translate_to_english",
                    side_effect=lambda title: title,
                ),
                patch.object(
                    self.api,
                    "predict",
                    side_effect=[
                        RuntimeError("Simulated prediction failure"),
                        prediction_result,
                    ],
                ) as mock_predict,
                patch.object(
                    self.api,
                    "validate_taxonomy",
                    return_value={"support_margin": 0.2},
                ),
                patch.object(
                    self.api,
                    "determine_taxonomy_status",
                    return_value="supported",
                ),
                patch.object(
                    self.api,
                    "evaluate_decision",
                    return_value={"decision": "APPROVE", "reasons": []},
                ),
                patch.object(
                    self.api,
                    "create_seller_product",
                    return_value={"product_id": 456, "reason": None},
                ) as save_product,
            ):
                response = self.client.post(
                    "/seller/bulk-products",
                    files=[
                        (
                            "catalog",
                            (
                                "catalog.csv",
                                (
                                    "product_id,title,brand,image\n"
                                    "p1,First shoe,BrandA,first.jpg\n"
                                    "p2,Second shoe,BrandB,second.jpg\n"
                                ).encode(),
                                "text/csv",
                            ),
                        ),
                        ("images", ("first.jpg", b"first image", "image/jpeg")),
                        ("images", ("second.jpg", b"second image", "image/jpeg")),
                    ],
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["failed_rows"], 1)
        self.assertEqual(data["successful_rows"], 1)
        self.assertEqual(data["results"][0]["product_id"], "p1")
        self.assertFalse(data["results"][0]["success"])
        self.assertIn("Simulated prediction failure", data["results"][0]["message"])
        self.assertEqual(data["results"][1]["product_id"], "p2")
        self.assertTrue(data["results"][1]["success"])
        self.assertEqual(mock_predict.call_count, 2)
        save_product.assert_called_once()
        update_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()
