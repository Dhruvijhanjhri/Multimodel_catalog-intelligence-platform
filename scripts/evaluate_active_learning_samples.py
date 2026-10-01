from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from services.database.model_registry import get_active_model_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = PROJECT_ROOT / "embeddings" / "active_learning"

ACTIVE_MODEL = get_active_model_version("multimodal_classifier")
MODEL_VERSION = ACTIVE_MODEL["version"]
MODEL_PATH = PROJECT_ROOT / ACTIVE_MODEL["artifact_path"]


IMAGE_EMBEDDINGS_FILE = (
    EMBEDDING_DIR / "active_learning_image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = (
    EMBEDDING_DIR / "active_learning_text_embeddings.npy"
)

LABELS_FILE = (
    EMBEDDING_DIR / "active_learning_labels.npy"
)


LABELS = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen"
]


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


image_embeddings = np.load(
    IMAGE_EMBEDDINGS_FILE
)

text_embeddings = np.load(
    TEXT_EMBEDDINGS_FILE
)

labels = np.load(
    LABELS_FILE
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


print()
print("-" * 50)
print("Active Learning Data Loaded")
print("-" * 50)

print("Image Embeddings :", image_embeddings.shape)
print("Text Embeddings  :", text_embeddings.shape)
print("Combined Features:", X.shape)
print("Labels           :", y.shape)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = MultimodalClassifier().to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


print()
print("-" * 50)
print("Baseline Model Loaded")
print("-" * 50)

print("Model Path :", MODEL_PATH)
print("Device     :", device)


with torch.no_grad():

    X_device = X.to(device)

    outputs = model(X_device)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_ids = probabilities.argmax(
        dim=1
    )


print()
print("-" * 50)
print("Baseline Predictions")
print("-" * 50)

for index in range(len(y)):

    actual_label = LABELS[
        y[index].item()
    ]

    predicted_label = LABELS[
        predicted_ids[index].item()
    ]

    confidence = probabilities[
        index,
        predicted_ids[index]
    ].item()

    status = (
        "CORRECT"
        if actual_label == predicted_label
        else "INCORRECT"
    )

    print(
        f"{index + 1}. "
        f"Actual: {actual_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {confidence:.4f} | "
        f"{status}"
    )