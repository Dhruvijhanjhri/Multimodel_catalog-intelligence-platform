from pathlib import Path
import sys
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

EMBEDDING_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Split
# --------------------------------------------------

if len(sys.argv) != 2 or sys.argv[1] not in {"validation", "test"}:
    raise SystemExit("Usage: python scripts/generate_split_embeddings.py validation|test")

split_name = sys.argv[1]

print("-" * 40)
print("OpenCLIP Split Embedding Generator")
print("-" * 40)
print("Split:", split_name)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

split_df = pd.read_parquet(
    SPLIT_DIR / f"{split_name}.parquet"
)

print()
print("Official Dataset:", split_df.shape)

# --------------------------------------------------
# Load Image Manifest
# --------------------------------------------------

manifest_df = pd.read_parquet(
    PROCESSED_DIR / "image_manifest.parquet"
)

# Keep only products with images
split_df = split_df.merge(
    manifest_df[["item_id", "image_path"]],
    on="item_id",
    how="inner"
)

print("Dataset With Images:", split_df.shape)

# --------------------------------------------------
# Load OpenCLIP
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class ProductDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = preprocess(image)

        return image, row["title"], row["category"]

dataset = ProductDataset(split_df)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# Label Mapping
# --------------------------------------------------

label_to_id = {
    "Electronics_Accessories": 0,
    "Fashion_Travel": 1,
    "Footwear": 2,
    "Furniture": 3,
    "Hardware_HomeImprovement": 4,
    "Home_Kitchen": 5
}

print("Label Mapping:", label_to_id)

# --------------------------------------------------
# Generate Embeddings
# --------------------------------------------------

image_embeddings = []
text_embeddings = []
labels = []

with torch.no_grad():

    for images, texts, batch_labels in tqdm(loader):

        images = images.to(device)

        tokens = tokenizer(list(texts)).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)

        image_features = (
            image_features
            / image_features.norm(dim=-1, keepdim=True)
        )

        text_features = (
            text_features
            / text_features.norm(dim=-1, keepdim=True)
        )

        image_embeddings.append(
            image_features.cpu().numpy()
        )

        text_embeddings.append(
            text_features.cpu().numpy()
        )

        labels.extend(
            label_to_id[x]
            for x in batch_labels
        )

image_embeddings = np.vstack(image_embeddings)
text_embeddings = np.vstack(text_embeddings)
labels = np.array(labels)

print()
print("Image embeddings:", image_embeddings.shape)
print("Text embeddings :", text_embeddings.shape)
print("Labels           :", labels.shape)

# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    EMBEDDING_DIR / f"{split_name}_image_embeddings.npy",
    image_embeddings
)

np.save(
    EMBEDDING_DIR / f"{split_name}_text_embeddings.npy",
    text_embeddings
)

np.save(
    EMBEDDING_DIR / f"{split_name}_labels.npy",
    labels
)

# Save metadata so IDs remain traceable
metadata = split_df[
    ["item_id", "title", "category", "image_path"]
].copy()

metadata.to_parquet(
    EMBEDDING_DIR / f"{split_name}_embedding_metadata.parquet",
    index=False
)

print()
print("-" * 40)
print("Embeddings Saved")
print("-" * 40)
print("Split:", split_name)
print("Output:", EMBEDDING_DIR)
