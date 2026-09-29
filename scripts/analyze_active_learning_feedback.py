from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "active_learning"
    / "reviewed_training_data.parquet"
)


EXPECTED_CATEGORIES = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen"
]


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Active Learning feedback file not found: {INPUT_FILE}"
    )


df = pd.read_parquet(INPUT_FILE)


print()
print("=" * 60)
print("ACTIVE LEARNING FEEDBACK ANALYSIS")
print("=" * 60)

print()
print("Total eligible human-labeled samples:", len(df))

print()
print("Samples by feedback source:")
print(df["feedback_source"].value_counts().to_string())

print()
print("Human-labeled samples by category:")

category_counts = (
    df["training_label"]
    .value_counts()
    .reindex(EXPECTED_CATEGORIES, fill_value=0)
)

print(category_counts.to_string())

print()
print("Category coverage:")
print(
    f"{(category_counts > 0).sum()} "
    f"of {len(EXPECTED_CATEGORIES)} categories represented"
)

print()
print("Feedback records:")
print(
    df[
        [
            "product_id",
            "decision",
            "predicted_category",
            "corrected_category",
            "training_label",
            "feedback_source"
        ]
    ].to_string(index=False)
)

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)