from pathlib import Path

import numpy as np
import torch

from services.inference.predict import classifier
from services.inference.taxonomy_validator import (
    validate_taxonomy,
    LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_FILE = PROJECT_ROOT / "embeddings" / "validation_image_embeddings.npy"
TEXT_FILE = PROJECT_ROOT / "embeddings" / "validation_text_embeddings.npy"
LABEL_FILE = PROJECT_ROOT / "embeddings" / "validation_labels.npy"


print()
print("=" * 60)
print("Taxonomy Threshold Calibration")
print("=" * 60)


print()
print("Loading validation embeddings...")

images = np.load(IMAGE_FILE)
texts = np.load(TEXT_FILE)
labels = np.load(LABEL_FILE)

print("Images :", images.shape)
print("Texts  :", texts.shape)
print("Labels :", labels.shape)


print()
print("Running classifier...")

features = np.concatenate(
    [images, texts],
    axis=1
).astype(np.float32)

with torch.no_grad():

    tensor = torch.from_numpy(features)

    logits = classifier(tensor)

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

correct = predictions == labels


print(
    "Validation accuracy:",
    round(correct.mean() * 100, 2),
    "%"
)


print()
print("Calculating taxonomy signals...")


supports = []
margins = []


for i in range(len(labels)):

    predicted_category = LABELS[
        int(predictions[i])
    ]

    result = validate_taxonomy(
        images[i],
        texts[i],
        predicted_category
    )

    supports.append(
        result["catalog_support"]
    )

    margins.append(
        result["support_margin"]
    )

    if (i + 1) % 500 == 0:

        print(
            "Processed:",
            i + 1,
            "/",
            len(labels)
        )


supports = np.array(supports)
margins = np.array(margins)


print()
print("=" * 60)
print("GLOBAL THRESHOLD ANALYSIS")
print("=" * 60)


for threshold in [0.05, 0.10, 0.15, 0.20]:

    flagged = margins < threshold

    total_flagged = int(
        flagged.sum()
    )

    errors_caught = int(
        ((~correct) & flagged).sum()
    )

    correct_flagged = int(
        (correct & flagged).sum()
    )

    if total_flagged > 0:

        error_rate = (
            errors_caught
            / total_flagged
            * 100
        )

    else:

        error_rate = 0.0


    print()
    print(
        f"Margin < {threshold:.2f}"
    )

    print(
        "Flagged:",
        total_flagged
    )

    print(
        "Actual errors:",
        errors_caught
    )

    print(
        "Correct products:",
        correct_flagged
    )

    print(
        "Flag precision:",
        round(
            error_rate,
            2
        ),
        "%"
    )


print()
print("=" * 60)
print("CATEGORY-WISE THRESHOLD ANALYSIS")
print("=" * 60)


for category_id, category_name in enumerate(LABELS):

    category_mask = (
        labels == category_id
    )

    print()
    print(
        category_name
    )

    print("-" * 50)

    for threshold in [0.05, 0.10, 0.15, 0.20]:

        flagged = (
            category_mask
            & (margins < threshold)
        )

        total_flagged = int(
            flagged.sum()
        )

        errors_caught = int(
            ((~correct) & flagged).sum()
        )

        correct_flagged = int(
            (correct & flagged).sum()
        )

        if total_flagged > 0:

            precision = (
                errors_caught
                / total_flagged
                * 100
            )

        else:

            precision = 0.0


        print(
            f"  < {threshold:.2f}: "
            f"flagged={total_flagged}, "
            f"errors={errors_caught}, "
            f"correct={correct_flagged}, "
            f"precision={precision:.2f}%"
        )


print()
print("=" * 60)
print("Calibration Complete")
print("=" * 60)