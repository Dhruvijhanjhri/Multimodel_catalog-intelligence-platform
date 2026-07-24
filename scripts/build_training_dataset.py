"""
Build Training Dataset

This script:

1. Loads the production dataset
2. Creates a balanced 30K dataset
3. Splits into Train / Validation / Test
4. Saves all datasets
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

SPLIT_DIR = DATA_DIR / "splits"

SPLIT_DIR.mkdir(exist_ok=True)

print("Project Root :", PROJECT_ROOT)
print("Split Folder :", SPLIT_DIR)

# ---------------------------------------------------
# Load Production Dataset
# ---------------------------------------------------

production_file = PROCESSED_DIR / "production_products.parquet"

if not production_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {production_file}"
    )

df = pd.read_parquet(production_file)

print()
print("----------------------------------------")
print("Production Dataset Loaded")
print("----------------------------------------")
print("Shape :", df.shape)

print()
print(df.head())

print()
print(df["category"].value_counts())

# ---------------------------------------------------
# Target Distribution (30K Dataset)
# ---------------------------------------------------

TARGET_DISTRIBUTION = {
    "Electronics_Accessories": 10000,
    "Footwear": 8000,
    "Furniture": 6000,
    "Home_Kitchen": 6000,
    "Fashion_Travel": 2000,
    "Hardware_HomeImprovement": 1600
}

print()
print("----------------------------------------")
print("Target Distribution")
print("----------------------------------------")

total = 0

for category, count in TARGET_DISTRIBUTION.items():
    print(f"{category:<30} {count}")
    total += count

print("----------------------------------------")
print("Total Samples :", total)

# ---------------------------------------------------
# Create Balanced Dataset
# ---------------------------------------------------

balanced_parts = []

print()
print("----------------------------------------")
print("Sampling Dataset")
print("----------------------------------------")

for category, target_count in TARGET_DISTRIBUTION.items():

    category_df = df[df["category"] == category]

    available = len(category_df)

    print(f"{category:<30} Available: {available:6}  Target: {target_count:6}")

    selected_count = min(target_count, available)

    sampled = category_df.sample(
        n=selected_count,
        random_state=42
    )

    if available < target_count:
        print(
            f"  -> Only {available} available. Using all available products."
        )

    balanced_parts.append(sampled)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print()
print("----------------------------------------")
print("Balanced Dataset")
print("----------------------------------------")
print("Shape :", balanced_df.shape)

print()
print(balanced_df["category"].value_counts())

# ---------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------

train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.30,
    stratify=balanced_df["category"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["category"],
    random_state=42
)

print()
print("----------------------------------------")
print("Dataset Split")
print("----------------------------------------")

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")

# ---------------------------------------------------
# Save Train / Validation / Test
# ---------------------------------------------------

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_path = SPLIT_DIR / "train.parquet"
val_path = SPLIT_DIR / "validation.parquet"
test_path = SPLIT_DIR / "test.parquet"

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

print()
print("----------------------------------------")
print("Datasets Saved")
print("----------------------------------------")
print("Train      :", train_path)
print("Validation :", val_path)
print("Test       :", test_path)

# ---------------------------------------------------
# Verify Dataset Distribution
# ---------------------------------------------------

print()
print("----------------------------------------")
print("Train Distribution")
print("----------------------------------------")
print(train_df["category"].value_counts())

print()
print("----------------------------------------")
print("Validation Distribution")
print("----------------------------------------")
print(val_df["category"].value_counts())

print()
print("----------------------------------------")
print("Test Distribution")
print("----------------------------------------")
print(test_df["category"].value_counts())