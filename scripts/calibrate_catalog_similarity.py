import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

IMAGE_FILE = "embeddings/validation_image_embeddings.npy"
TEXT_FILE = "embeddings/validation_text_embeddings.npy"
LABEL_FILE = "embeddings/validation_labels.npy"
MODEL_FILE = "models/multimodal_classifier.pt"

TOP_K = 5

LABELS = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen"
]

print()
print("-" * 70)
print("Predicted-Category Catalog Support Calibration")
print("-" * 70)

image_embeddings = np.load(
    IMAGE_FILE
)

text_embeddings = np.load(
    TEXT_FILE
)

labels = np.load(
    LABEL_FILE
)

print()
print("Image embeddings :", image_embeddings.shape)
print("Text embeddings  :", text_embeddings.shape)
print("Labels           :", labels.shape)

features = np.concatenate(
    [
        image_embeddings,
        text_embeddings
    ],
    axis=1
)


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

            torch.nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)


model = MultimodalClassifier()

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location="cpu"
    )
)

model.eval()

features_tensor = torch.tensor(
    features,
    dtype=torch.float32
)

with torch.no_grad():

    logits = model(
        features_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, predictions = torch.max(
        probabilities,
        dim=1
    )

predictions = predictions.numpy()

confidence = confidence.numpy()

accuracy = (
    predictions == labels
).mean()

print()
print(
    "Validation accuracy :",
    f"{accuracy * 100:.2f}%"
)

print()
print("Creating normalized multimodal embeddings...")

combined_embeddings = (
    image_embeddings
    + text_embeddings
) / 2.0

norms = np.linalg.norm(
    combined_embeddings,
    axis=1,
    keepdims=True
)

combined_embeddings = (
    combined_embeddings
    / np.maximum(
        norms,
        1e-12
    )
)

print()
print("Calculating similarity matrix...")

similarity_matrix = cosine_similarity(
    combined_embeddings
)

correct_support = []
incorrect_support = []

all_support = []
all_correct_flags = []

print()
print("Calculating predicted-category Top-5 support...")

for i in range(
    len(labels)
):

    predicted_category = predictions[i]

    category_indices = np.where(
        labels == predicted_category
    )[0]

    category_indices = (
        category_indices[
            category_indices != i
        ]
    )

    if len(category_indices) < TOP_K:
        continue

    scores = similarity_matrix[
        i,
        category_indices
    ]

    top_scores = np.sort(
        scores
    )[-TOP_K:]

    support_score = (
        top_scores.mean()
    )

    all_support.append(
        support_score
    )

    is_correct = (
        predictions[i] == labels[i]
    )

    all_correct_flags.append(
        is_correct
    )

    if is_correct:

        correct_support.append(
            support_score
        )

    else:

        incorrect_support.append(
            support_score
        )


correct_support = np.array(
    correct_support
)

incorrect_support = np.array(
    incorrect_support
)

all_support = np.array(
    all_support
)

all_correct_flags = np.array(
    all_correct_flags
)

print()
print("-" * 70)
print("Correct Predictions")
print("-" * 70)

print(
    "Samples     :",
    len(correct_support)
)

print(
    "Mean        :",
    f"{correct_support.mean():.4f}"
)

print(
    "10th %ile   :",
    f"{np.percentile(correct_support, 10):.4f}"
)

print(
    "25th %ile   :",
    f"{np.percentile(correct_support, 25):.4f}"
)

print(
    "Median      :",
    f"{np.median(correct_support):.4f}"
)

print(
    "75th %ile   :",
    f"{np.percentile(correct_support, 75):.4f}"
)

print(
    "Minimum     :",
    f"{correct_support.min():.4f}"
)

print()
print("-" * 70)
print("Incorrect Predictions")
print("-" * 70)

print(
    "Samples     :",
    len(incorrect_support)
)

print(
    "Mean        :",
    f"{incorrect_support.mean():.4f}"
)

print(
    "10th %ile   :",
    f"{np.percentile(incorrect_support, 10):.4f}"
)

print(
    "25th %ile   :",
    f"{np.percentile(incorrect_support, 25):.4f}"
)

print(
    "Median      :",
    f"{np.median(incorrect_support):.4f}"
)

print(
    "75th %ile   :",
    f"{np.percentile(incorrect_support, 75):.4f}"
)

print(
    "Maximum     :",
    f"{incorrect_support.max():.4f}"
)

print()
print("-" * 70)
print("Support Threshold Analysis")
print("-" * 70)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

for threshold in thresholds:

    low_support = (
        all_support < threshold
    )

    total = len(
        all_support
    )

    flagged = (
        low_support.sum()
    )

    incorrect_flagged = (
        low_support
        & (~all_correct_flags)
    ).sum()

    correct_flagged = (
        low_support
        & all_correct_flags
    ).sum()

    if flagged > 0:

        incorrect_precision = (
            incorrect_flagged
            / flagged
        )

    else:

        incorrect_precision = 0.0

    print()
    print(
        f"Threshold < {threshold:.2f}"
    )

    print(
        "Flagged       :",
        f"{flagged}/{total}",
        f"({flagged / total * 100:.2f}%)"
    )

    print(
        "Incorrect in flagged :",
        incorrect_flagged
    )

    print(
        "Correct in flagged   :",
        correct_flagged
    )

    print(
        "Flag precision       :",
        f"{incorrect_precision * 100:.2f}%"
    )

print()
print("-" * 70)
print("Calibration Complete")
print("-" * 70)