from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates"

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


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

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


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


def train_candidate():

    print("-" * 50)
    print("Candidate Model Training")
    print("-" * 50)

    X_train, y_train = load_split("train")

    active_image_embeddings = np.load(
        EMBEDDING_DIR / "active_learning" / "active_learning_image_embeddings.npy"
    )

    active_text_embeddings = np.load(
        EMBEDDING_DIR / "active_learning" / "active_learning_text_embeddings.npy"
    )

    active_labels = np.load(
        EMBEDDING_DIR / "active_learning" / "active_learning_labels.npy"
    )

    active_features = np.concatenate(
        [active_image_embeddings, active_text_embeddings],
        axis=1
    )

    X_active = torch.tensor(
        active_features,
        dtype=torch.float32
    )

    y_active = torch.tensor(
        active_labels,
        dtype=torch.long
    )

    X_train = torch.cat(
        [X_train, X_active],
        dim=0
    )

    y_train = torch.cat(
        [y_train, y_active],
        dim=0
    )

    X_val, y_val = load_split("validation")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultimodalClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 15
    best_accuracy = 0.0

    candidate_version = "multimodal_classifier_candidate_20261007"

    candidate_path = (
        CANDIDATE_DIR / f"{candidate_version}.pt"
    )

    start = time.time()

    for epoch in range(epochs):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                predictions = outputs.argmax(1)

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Validation Accuracy {accuracy * 100:.2f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                candidate_path
            )

    elapsed = (time.time() - start) / 60

    print("-" * 50)
    print("Candidate Training Finished")
    print("-" * 50)

    print(
        f"Best Validation Accuracy : "
        f"{best_accuracy * 100:.2f}%"
    )

    print(f"Training Time : {elapsed:.1f} minutes")
    print("Candidate Artifact :", candidate_path)


if __name__ == "__main__":
    train_candidate()
