from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.parquet"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Maximum number of titles sampled from each supported product type.
SAMPLES_PER_TYPE = 50

print()
print("-" * 60)
print("Taxonomy Validation Calibration")
print("-" * 60)

print("Device :", DEVICE)

df = pd.read_parquet(TRAIN_FILE)

print("Training samples :", len(df))
print(
    "Product types    :",
    df["product_type"].nunique()
)

PRODUCT_TYPES = sorted(
    df["product_type"]
    .dropna()
    .unique()
)

print()
print("Loading OpenCLIP...")

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model = clip_model.to(DEVICE)
clip_model.eval()

print("OpenCLIP loaded.")


def clean_product_type(product_type):
    return (
        str(product_type)
        .replace("_", " ")
        .lower()
    )


# ------------------------------------------------------------
# Create embeddings for the 51 supported product types
# ------------------------------------------------------------

type_texts = [
    clean_product_type(product_type)
    for product_type in PRODUCT_TYPES
]

with torch.no_grad():
    tokens = tokenizer(type_texts).to(DEVICE)

    type_embeddings = clip_model.encode_text(tokens)

    type_embeddings = (
        type_embeddings
        / type_embeddings.norm(
            dim=-1,
            keepdim=True
        )
    )

print()
print("Product type embeddings created.")
print("Supported types :", len(PRODUCT_TYPES))


# ------------------------------------------------------------
# Sample representative titles
# ------------------------------------------------------------

sample_df = (
    df[
        [
            "title",
            "product_type"
        ]
    ]
    .dropna()
    .groupby("product_type", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(
                SAMPLES_PER_TYPE,
                len(group)
            ),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

print()
print("Calibration samples :", len(sample_df))
print(
    "Maximum per type    :",
    SAMPLES_PER_TYPE
)

print()
print("Calculating taxonomy similarity...")
print("-" * 60)


# ------------------------------------------------------------
# Encode sampled titles
# ------------------------------------------------------------

similarities = []

batch_size = 32

for start in range(
    0,
    len(sample_df),
    batch_size
):

    batch = sample_df.iloc[
        start:start + batch_size
    ]

    titles = (
        batch["title"]
        .astype(str)
        .tolist()
    )

    with torch.no_grad():

        tokens = tokenizer(titles).to(DEVICE)

        title_embeddings = clip_model.encode_text(
            tokens
        )

        title_embeddings = (
            title_embeddings
            / title_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        scores = (
            title_embeddings
            @ type_embeddings.T
        )

        max_scores, predicted_indices = torch.max(
            scores,
            dim=1
        )

    for index in range(len(batch)):

        actual_type = batch.iloc[index][
            "product_type"
        ]

        predicted_type = PRODUCT_TYPES[
            predicted_indices[index].item()
        ]

        similarity = max_scores[
            index
        ].item()

        similarities.append(
            {
                "title": batch.iloc[index]["title"],
                "actual_product_type": actual_type,
                "matched_product_type": predicted_type,
                "similarity": similarity
            }
        )

    processed = min(
        start + batch_size,
        len(sample_df)
    )

    print(
        f"Processed {processed}/{len(sample_df)} titles"
    )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

result_df = pd.DataFrame(similarities)

print()
print("-" * 60)
print("Calibration Results")
print("-" * 60)

print()
print("Similarity statistics:")

print(
    result_df["similarity"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Product-type matching accuracy
# ------------------------------------------------------------

correct_matches = (
    result_df[
        result_df["actual_product_type"]
        ==
        result_df["matched_product_type"]
    ]
)

print()
print(
    "Correct product-type matches :",
    len(correct_matches)
)

print(
    "Total calibration samples    :",
    len(result_df)
)

print(
    "Product-type match rate      :",
    f"{100 * len(correct_matches) / len(result_df):.2f}%"
)


# ------------------------------------------------------------
# Similarity distribution by actual product type
# ------------------------------------------------------------

print()
print("Similarity by product type:")
print("-" * 60)

type_stats = (
    result_df
    .groupby("actual_product_type")["similarity"]
    .agg(
        ["count", "min", "mean", "median", "max"]
    )
    .sort_values("min")
)

print(
    type_stats.to_string()
)


# ------------------------------------------------------------
# Lowest similarity examples
# ------------------------------------------------------------

print()
print("Lowest similarity examples:")
print("-" * 60)

print(
    result_df
    .sort_values("similarity")
    .head(20)
    [
        [
            "actual_product_type",
            "matched_product_type",
            "similarity",
            "title"
        ]
    ]
    .to_string(index=False)
)


# ------------------------------------------------------------
# Highest similarity examples
# ------------------------------------------------------------

print()
print("Highest similarity examples:")
print("-" * 60)

print(
    result_df
    .sort_values(
        "similarity",
        ascending=False
    )
    .head(10)
    [
        [
            "actual_product_type",
            "matched_product_type",
            "similarity",
            "title"
        ]
    ]
    .to_string(index=False)
)


print()
print("-" * 60)
print("Calibration Complete")
print("-" * 60)