import unittest
from unittest.mock import MagicMock, patch

from services.database.review_queue import add_to_review_queue


class TestAddToReviewQueue(unittest.TestCase):

    def setUp(self):
        self.conn = MagicMock()
        self.cursor = MagicMock()
        self.conn.cursor.return_value.__enter__.return_value = self.cursor

        self.queue_item = {
            "item_id": "SELLER-123",
            "image_name": "shoe.jpg",
            "title": "Running shoes",
            "predicted_category": "Footwear",
            "confidence": 0.65,
            "image_similarity": 0.42,
            "duplicate_score": 0.15,
            "reason": "low_confidence",
            "prediction_id": 202,
        }

    def test_supplied_connection_is_not_committed_or_closed(self):
        self.cursor.fetchone.return_value = None

        add_to_review_queue(
            **self.queue_item,
            conn=self.conn,
        )

        self.conn.commit.assert_not_called()
        self.conn.rollback.assert_not_called()
        self.conn.close.assert_not_called()

        insert_calls = [
            call
            for call in self.cursor.execute.call_args_list
            if "INSERT INTO review_queue" in call.args[0]
        ]
        self.assertEqual(len(insert_calls), 1)

        saved_values = insert_calls[0].args[1]
        self.assertEqual(saved_values[0], "SELLER-123")
        self.assertEqual(saved_values[-1], 202)

    def test_existing_pending_review_is_updated_to_latest_prediction(self):
        self.cursor.fetchone.return_value = (55,)

        add_to_review_queue(
            **self.queue_item,
            conn=self.conn,
        )

        update_calls = [
            call
            for call in self.cursor.execute.call_args_list
            if "UPDATE review_queue" in call.args[0]
        ]
        self.assertEqual(len(update_calls), 1)

        updated_values = update_calls[0].args[1]
        self.assertEqual(updated_values[0], 202)
        self.assertEqual(updated_values[-1], 55)

        self.conn.commit.assert_not_called()
        self.conn.rollback.assert_not_called()
        self.conn.close.assert_not_called()

    @patch("services.database.review_queue.get_connection")
    def test_helper_manages_its_own_connection_when_none_is_supplied(
        self,
        mock_get_connection,
    ):
        mock_get_connection.return_value = self.conn
        self.cursor.fetchone.return_value = None

        add_to_review_queue(**self.queue_item)

        self.conn.commit.assert_called_once()
        self.conn.rollback.assert_not_called()
        self.conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
