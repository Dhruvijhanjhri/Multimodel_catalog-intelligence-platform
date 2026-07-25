from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR = PROJECT_ROOT / "data" / "raw"

IMAGES_DIR = (
    RAW_DIR
    / "abo-images-small"
    / "images"
    / "small"
)

print("Project Root :", PROJECT_ROOT)
print("Images Folder:", IMAGES_DIR)

# --------------------------------------------------
# Load production dataset
# --------------------------------------------------

products = pd.read_parquet(
    PROCESSED_DIR / "production_products.parquet"
)

print()
print("--------------------------------")
print("Production Products")
print("--------------------------------")

print(products.shape)
print(products.head())

# --------------------------------------------------
# Load image metadata
# --------------------------------------------------

images_df = pd.read_csv(
    RAW_DIR / "images.csv"
)

print()
print("--------------------------------")
print("Image Metadata")
print("--------------------------------")

print(images_df.shape)
print(images_df.head())

# --------------------------------------------------
# Load listing-image mapping
# --------------------------------------------------

mapping_file = (
    RAW_DIR
    / "abo-images-small"
    / "images"
    / "metadata"
    / "images.csv.gz"
)

mapping_df = pd.read_csv(mapping_file)

print()
print("--------------------------------")
print("Listing Image Mapping")
print("--------------------------------")

print(mapping_df.shape)
print(mapping_df.head())
print(mapping_df.columns.tolist())

# --------------------------------------------------
# Inspect one listing
# --------------------------------------------------

sample_listing = pd.read_json(
    RAW_DIR
    / "abo-listings"
    / "listings"
    / "metadata"
    / "listings_0.json.gz",
    lines=True
)

print(sample_listing.columns.tolist())

print()
print("--------------------------------")
print("First Listing")
print("--------------------------------")

print(sample_listing.iloc[0])

# --------------------------------------------------
# Build Image Mapping
# --------------------------------------------------

metadata_dir = (
    RAW_DIR
    / "abo-listings"
    / "listings"
    / "metadata"
)

metadata_files = sorted(metadata_dir.glob("listings_*.json.gz"))

mapping_parts = []

for file in metadata_files:

    df = pd.read_json(file, lines=True)

    temp = df[["item_id", "main_image_id"]].copy()

    mapping_parts.append(temp)

    print(f"{file.name:<20} {len(temp):>6,} rows")

image_mapping = pd.concat(
    mapping_parts,
    ignore_index=True
)

# Keep only one image per product
image_mapping = image_mapping.drop_duplicates(
    subset="item_id",
    keep="first"
)

print()
print("--------------------------------")
print("Image Mapping")
print("--------------------------------")

print(image_mapping.shape)
print(image_mapping.head())

# --------------------------------------------------
# Keep only production products
# --------------------------------------------------

manifest = products.merge(
    image_mapping,
    on="item_id",
    how="left"
)

print()
print("--------------------------------")
print("After Merge")
print("--------------------------------")

print(manifest.shape)

print()

print(
    "Missing Images:",
    manifest["main_image_id"].isna().sum()
)

# ----------------------------------------
# Merge with image metadata
# ----------------------------------------

manifest = manifest.merge(
    images_df[
        ["image_id", "path"]
    ],
    left_on="main_image_id",
    right_on="image_id",
    how="left"
)

print()
print("--------------------------------")
print("Image Paths Added")
print("--------------------------------")

print(manifest.shape)

print()

print(
    "Missing Paths:",
    manifest["path"].isna().sum()
)

# ----------------------------------------
# Create full image path
# ----------------------------------------

manifest["image_path"] = manifest["path"].apply(
    lambda x: str(IMAGES_DIR / x)
    if pd.notna(x)
    else None
)

manifest = manifest.dropna(
    subset=["image_path"]
)

print()
print("--------------------------------")
print("Final Manifest")
print("--------------------------------")

print(manifest.shape)

print(manifest.head())

output_file = (
    PROCESSED_DIR
    / "image_manifest.parquet"
)

manifest.to_parquet(
    output_file,
    index=False
)

print()
print("--------------------------------")
print("Image Manifest Saved")
print("--------------------------------")

print(output_file)