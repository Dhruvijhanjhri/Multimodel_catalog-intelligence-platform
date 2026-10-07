from pathlib import Path
import sys

import numpy as np
import pandas as pd
import open_clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_LEARNING_DIR = PROJECT_ROOT / "data" / "active_learning"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings" / "active_learning"

EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILE = ACTIVE_LEARNING_DIR / "reviewed_training_data.parquet"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Active Learning dataset not found: {INPUT_FILE}"
    )


df = pd.read_parquet(INPUT_FILE)
df = df[df["dataset_membership"] == "new_reviewed"].copy()


print()
print("-" * 50)
print("Active Learning Embedding Generation")
print("-" * 50)

print("Input File :", INPUT_FILE)
print("Samples    :", len(df))
print()

print(
    df[
        ["product_id", "training_label"]
    ].to_string(index=False)
)


def resolve_image_path(image_path):

    path = Path(image_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


device = "cuda" if torch.cuda.is_available() else "cpu"


print()
print("Loading OpenCLIP model...")
print("Device :", device)


clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model = clip_model.to(device)
clip_model.eval()


print("OpenCLIP model loaded successfully.")


class ActiveLearningDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = resolve_image_path(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = preprocess(image)

        return (
            image,
            str(row["title"]),
            str(row["training_label"])
        )


dataset = ActiveLearningDataset(df)


loader = DataLoader(
    dataset,
    batch_size=7,
    shuffle=False,
    num_workers=0
)


print()
print("DataLoader Ready")
print("Samples :", len(dataset))
print("Batches :", len(loader))


label_to_id = {
    "Electronics_Accessories": 0,
    "Fashion_Travel": 1,
    "Footwear": 2,
    "Furniture": 3,
    "Hardware_HomeImprovement": 4,
    "Home_Kitchen": 5
}


image_embeddings = []
text_embeddings = []
labels = []


print()
print("Generating Active Learning embeddings...")
print("-" * 50)


with torch.no_grad():

    for images, texts, batch_labels in tqdm(loader):

        images = images.to(device)

        tokens = clip_tokenizer(
            list(texts)
        ).to(device)


        image_features = clip_model.encode_image(
            images
        )

        text_features = clip_model.encode_text(
            tokens
        )


        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )


        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True
            )
        )


        image_embeddings.append(
            image_features.cpu().numpy()
        )


        text_embeddings.append(
            text_features.cpu().numpy()
        )


        labels.extend(
            label_to_id[label]
            for label in batch_labels
        )


image_embeddings = np.vstack(
    image_embeddings
)

text_embeddings = np.vstack(
    text_embeddings
)

labels = np.array(
    labels
)


np.save(
    EMBEDDING_DIR
    / "active_learning_image_embeddings.npy",
    image_embeddings
)


np.save(
    EMBEDDING_DIR
    / "active_learning_text_embeddings.npy",
    text_embeddings
)


np.save(
    EMBEDDING_DIR
    / "active_learning_labels.npy",
    labels
)


metadata = df[
    [
        "product_id",
        "title",
        "training_label",
        "image_path"
    ]
].copy()


metadata.to_parquet(
    EMBEDDING_DIR
    / "active_learning_embedding_metadata.parquet",
    index=False
)


print()
print("-" * 50)
print("Embeddings Saved")
print("-" * 50)

print(
    "Image Embeddings :",
    image_embeddings.shape
)

print(
    "Text Embeddings  :",
    text_embeddings.shape
)

print(
    "Labels           :",
    labels.shape
)

print(
    "Output Directory :",
    EMBEDDING_DIR
)