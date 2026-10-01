from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"


class MultimodalClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)


def load_split(split_name):

    image_embeddings = np.load(
        EMBEDDING_DIR / f"{split_name}_image_embeddings.npy"
    )

    text_embeddings = np.load(
        EMBEDDING_DIR / f"{split_name}_text_embeddings.npy"
    )

    labels = np.load(
        EMBEDDING_DIR / f"{split_name}_labels.npy"
    )

    features = np.concatenate(
        [image_embeddings, text_embeddings],
        axis=1
    )

    X = torch.tensor(
        features,
        dtype=torch.float32
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    return X, y


def evaluate_model(model_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultimodalClassifier().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    results = {}

    for split_name in ["validation", "test"]:

        X, y = load_split(split_name)

        with torch.no_grad():

            outputs = model(
                X.to(device)
            )

            predictions = outputs.argmax(
                dim=1
            )

        correct = (
            predictions.cpu() == y
        ).sum().item()

        total = len(y)

        accuracy = correct / total

        results[split_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

    return results


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/evaluate_model.py <model_path>"
        )
        raise SystemExit(1)

    model_path = Path(sys.argv[1])

    results = evaluate_model(model_path)

    print()
    print("-" * 50)
    print("Model Evaluation")
    print("-" * 50)

    print("Model :", model_path)
    print("Metrics :", results)

    for split_name, metrics in results.items():

        print(
            f"{split_name.capitalize()} Accuracy : "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"{split_name.capitalize()} Samples  : "
            f"{metrics['total']}"
        )