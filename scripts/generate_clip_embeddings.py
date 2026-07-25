from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch

print("-" * 40)
print("OpenCLIP Embedding Generator")
print("-" * 40)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"

EMBEDDING_DIR.mkdir(exist_ok=True)

print("Project Root :", PROJECT_ROOT)
print("Embedding Dir:", EMBEDDING_DIR)

# --------------------------------------------------
# Load Train Dataset
# --------------------------------------------------

train_df = pd.read_parquet(
    SPLIT_DIR / "train.parquet"
)

print()
print("-" * 40)
print("Training Dataset")
print("-" * 40)

print(train_df.shape)
print(train_df.head())

# --------------------------------------------------
# Load Image Manifest
# --------------------------------------------------

manifest_df = pd.read_parquet(
    PROCESSED_DIR / "image_manifest.parquet"
)

print()
print("-" * 40)
print("Image Manifest")
print("-" * 40)

print(manifest_df.shape)

# --------------------------------------------------
# Merge Train Dataset with Image Manifest
# --------------------------------------------------

train_df = train_df.merge(
    manifest_df[
        ["item_id", "image_path"]
    ],
    on="item_id",
    how="inner"
)

print()
print("-" * 40)
print("Training Dataset With Images")
print("-" * 40)

print(train_df.shape)

print()

print(
    train_df[
        ["item_id", "title", "category", "image_path"]
    ].head()
)

# --------------------------------------------------
# Load OpenCLIP Model
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print()
print("Device :", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

print()
print("-" * 40)
print("OpenCLIP Model Loaded")
print("-" * 40)

print("Model : ViT-B-32")
print("Weights : laion2b_s34b_b79k")

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Image Dataset
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

        text = row["title"]

        label = row["category"]

        return image, text, label

# --------------------------------------------------
# DataLoader
# --------------------------------------------------

dataset = ProductDataset(train_df)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print()
print("-" * 40)
print("DataLoader Ready")
print("-" * 40)

print("Dataset Size :", len(dataset))
print("Batches      :", len(loader))

# --------------------------------------------------
# Test One Batch
# --------------------------------------------------

images, texts, labels = next(iter(loader))

print()
print("-" * 40)
print("First Batch")
print("-" * 40)

print("Images :", images.shape)
print("Texts  :", len(texts))
print("Labels :", len(labels))

print()
print("Sample Title:")
print(texts[0])

print()
print("Sample Label:")
print(labels[0])

# --------------------------------------------------
# Encode Labels
# --------------------------------------------------

label_to_id = {
    label: idx
    for idx, label in enumerate(
        sorted(train_df["category"].unique())
    )
}

id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}

print()
print("-" * 40)
print("Label Mapping")
print("-" * 40)

print(label_to_id)

# --------------------------------------------------
# Storage
# --------------------------------------------------

image_embeddings = []
text_embeddings = []
labels = []

# --------------------------------------------------
# Generate Embeddings
# --------------------------------------------------

from tqdm import tqdm

print()
print("-" * 40)
print("Generating Embeddings")
print("-" * 40)

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
            [
                label_to_id[x]
                for x in batch_labels
            ]
        )

# --------------------------------------------------
# Combine Arrays
# --------------------------------------------------

image_embeddings = np.vstack(image_embeddings)

text_embeddings = np.vstack(text_embeddings)

labels = np.array(labels)

print()
print("-" * 40)
print("Embedding Shapes")
print("-" * 40)

print("Images :", image_embeddings.shape)
print("Texts  :", text_embeddings.shape)
print("Labels :", labels.shape)

# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    EMBEDDING_DIR / "train_image_embeddings.npy",
    image_embeddings
)

np.save(
    EMBEDDING_DIR / "train_text_embeddings.npy",
    text_embeddings
)

np.save(
    EMBEDDING_DIR / "train_labels.npy",
    labels
)

print()
print("-" * 40)
print("Embeddings Saved")
print("-" * 40)

print(EMBEDDING_DIR)