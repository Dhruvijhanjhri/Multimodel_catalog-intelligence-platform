"""
Build Production Dataset

This script:
1. Loads the cleaned ABO metadata
2. Maps product types to final categories
3. Creates the production dataset
4. Saves the processed dataset
"""

from pathlib import Path
import pandas as pd
import yaml

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)

print("Project Root :", PROJECT_ROOT)
print("Processed Dir:", PROCESSED_DIR)

# ---------------------------------------------------
# Load Category Mapping
# ---------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs"

mapping_file = CONFIG_DIR / "category_mapping.yaml"

with open(mapping_file, "r", encoding="utf-8") as f:
    CATEGORY_MAPPING = yaml.safe_load(f)

print()
print("Category Mapping Loaded Successfully")

for category, product_types in CATEGORY_MAPPING.items():
    print(f"{category:<30} {len(product_types)} product types")

# ---------------------------------------------------
# Load ABO Metadata
# ---------------------------------------------------

METADATA_DIR = (
    RAW_DIR
    / "abo-listings"
    / "listings"
    / "metadata"
)

metadata_files = sorted(
    METADATA_DIR.glob("listings_*.json.gz")
)

print()
print(f"Metadata Files Found : {len(metadata_files)}")

all_data = []

for file in metadata_files:

    df = pd.read_json(
        file,
        lines=True
    )

    all_data.append(df)

    print(f"{file.name:<20} {len(df):>8,} rows")

combined_df = pd.concat(
    all_data,
    ignore_index=True
)

print("\n------------------------------------")
print("Combined Dataset Shape :", combined_df.shape)

# ---------------------------------------------------
# Extract Required Columns
# ---------------------------------------------------

def extract_value(value):
    """
    Extract readable value from ABO metadata.
    """

    # Handle None
    if value is None:
        return None

    # Handle missing float values (NaN)
    if isinstance(value, float) and pd.isna(value):
        return None

    # Handle lists
    if isinstance(value, list):

        if len(value) == 0:
            return None

        first = value[0]

        if isinstance(first, dict):
            return first.get("value")

        return first

    # Handle dictionaries
    if isinstance(value, dict):
        return value.get("value")

    # Everything else
    return value


clean_df = pd.DataFrame()

clean_df["item_id"] = combined_df["item_id"]

clean_df["title"] = combined_df["item_name"].apply(extract_value)

clean_df["product_type"] = combined_df["product_type"].apply(extract_value)

clean_df["brand"] = combined_df["brand"].apply(extract_value)

clean_df["color"] = combined_df["color"].apply(extract_value)

print()
print("------------------------------------")
print("Clean Dataset Shape :", clean_df.shape)

print()
print(clean_df.head())

# ---------------------------------------------------
# Basic Data Cleaning
# ---------------------------------------------------

# Remove rows with missing title or product type
clean_df = clean_df.dropna(
    subset=["title", "product_type"]
)

# Remove duplicate products based on item_id
clean_df = clean_df.drop_duplicates(
    subset="item_id"
)

# Remove extra spaces
clean_df["title"] = clean_df["title"].astype(str).str.strip()
clean_df["product_type"] = clean_df["product_type"].astype(str).str.strip()

# Remove empty titles
clean_df = clean_df[
    clean_df["title"] != ""
]

print()
print("------------------------------------")
print("After Cleaning")
print("------------------------------------")
print("Dataset Shape :", clean_df.shape)

print()
print(clean_df.isnull().sum())

# ---------------------------------------------------
# Assign Final Category
# ---------------------------------------------------

# Reverse mapping:
# Product Type  -->  Final Category

product_type_to_category = {}

for category, product_types in CATEGORY_MAPPING.items():

    for product_type in product_types:

        product_type_to_category[product_type] = category


clean_df["category"] = clean_df["product_type"].map(
    product_type_to_category
)

# Keep only mapped categories
production_df = clean_df.dropna(
    subset=["category"]
).reset_index(drop=True)

print()
print("------------------------------------")
print("Production Dataset")
print("------------------------------------")
print("Shape :", production_df.shape)

print()
print(production_df["category"].value_counts())

# ---------------------------------------------------
# Save Clean Dataset
# ---------------------------------------------------

output_file = PROCESSED_DIR / "production_products.parquet"

production_df.to_parquet(
    output_file,
    index=False
)

print()
print("------------------------------------")
print("Production Dataset Saved")
print("------------------------------------")
print("Location :", output_file)
print(f"Rows     : {len(production_df):,}")
print(f"Columns  : {production_df.shape[1]}")