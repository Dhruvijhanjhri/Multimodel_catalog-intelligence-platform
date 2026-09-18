import numpy as np
import pandas as pd
import torch

from services.database.postgres import get_connection


IMAGE_EMBEDDINGS = "embeddings/train_image_embeddings.npy"
TEXT_EMBEDDINGS = "embeddings/train_text_embeddings.npy"
METADATA_FILE = "embeddings/embedding_metadata.parquet"
MODEL_FILE = "models/multimodal_classifier.pt"

LABELS = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen",
]

MODEL_VERSION = "multimodal_classifier_v1"
PREDICTION_SOURCE = "saved_embeddings"


class MultimodalClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.network(x)


def generate_predictions():
    print("Loading embeddings...")

    image_embeddings = np.load(IMAGE_EMBEDDINGS)
    text_embeddings = np.load(TEXT_EMBEDDINGS)
    metadata = pd.read_parquet(METADATA_FILE)

    print("Image embeddings:", image_embeddings.shape)
    print("Text embeddings:", text_embeddings.shape)
    print("Metadata rows:", len(metadata))

    features = np.concatenate(
        [image_embeddings, text_embeddings],
        axis=1,
    )

    x = torch.tensor(features, dtype=torch.float32)

    print("Loading classifier...")

    model = MultimodalClassifier()

    state_dict = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
        predictions = probabilities.argmax(dim=1)
        confidences = probabilities.max(dim=1).values

    return metadata, predictions.numpy(), confidences.numpy()


def insert_predictions(metadata, predictions, confidences):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT id, item_id
                FROM products
                """
            )

            product_map = {
                item_id: product_id
                for product_id, item_id in cursor.fetchall()
            }

            records = []

            for i, row in enumerate(metadata.itertuples(index=False)):
                product_id = product_map.get(row.item_id)

                if product_id is None:
                    raise ValueError(
                        f"Product not found for item_id: {row.item_id}"
                    )

                records.append(
                    (
                        product_id,
                        LABELS[int(predictions[i])],
                        float(confidences[i]),
                        MODEL_VERSION,
                        PREDICTION_SOURCE,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO product_predictions
                (
                    product_id,
                    predicted_category,
                    confidence,
                    model_version,
                    prediction_source
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                records,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    print(f"Inserted {len(records)} predictions into PostgreSQL.")


if __name__ == "__main__":
    metadata, predictions, confidences = generate_predictions()

    insert_predictions(
        metadata,
        predictions,
        confidences,
    )