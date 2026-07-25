from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

print("-" * 50)
print("Multimodal Classifier Training")
print("-" * 50)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("Project Root :", PROJECT_ROOT)
print("Embedding Dir:", EMBEDDING_DIR)
print("Model Dir    :", MODEL_DIR)

# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------

image_embeddings = np.load(
    EMBEDDING_DIR / "train_image_embeddings.npy"
)

text_embeddings = np.load(
    EMBEDDING_DIR / "train_text_embeddings.npy"
)

labels = np.load(
    EMBEDDING_DIR / "train_labels.npy"
)

print()
print("-" * 50)
print("Embeddings Loaded")
print("-" * 50)

print("Image Embeddings :", image_embeddings.shape)
print("Text Embeddings  :", text_embeddings.shape)
print("Labels           :", labels.shape)

# --------------------------------------------------
# Feature Fusion
# --------------------------------------------------

features = np.concatenate(
    [image_embeddings, text_embeddings],
    axis=1
)

print()
print("-" * 50)
print("Fused Features")
print("-" * 50)

print(features.shape)

# --------------------------------------------------
# Convert to Tensors
# --------------------------------------------------

X = torch.tensor(
    features,
    dtype=torch.float32
)

y = torch.tensor(
    labels,
    dtype=torch.long
)

print()
print("-" * 50)
print("Tensor Shapes")
print("-" * 50)

print("Features :", X.shape)
print("Labels   :", y.shape)

# --------------------------------------------------
# Train / Validation Split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("-" * 50)
print("Train / Validation Split")
print("-" * 50)

print("Train :", X_train.shape)
print("Validation :", X_val.shape)

# --------------------------------------------------
# Tensor Dataset
# --------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

print()
print("-" * 50)
print("DataLoaders Ready")
print("-" * 50)

print("Train Batches :", len(train_loader))
print("Validation Batches :", len(val_loader))

# --------------------------------------------------
# Multimodal Classifier
# --------------------------------------------------

import torch.nn as nn


class MultimodalClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 6)

        )

    def forward(self, x):

        return self.network(x)

# --------------------------------------------------
# Initialize Model
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalClassifier().to(device)

print()
print("-" * 50)
print("Model Created")
print("-" * 50)

print(model)
print()
print("Device :", device)

import time

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 15

train_losses = []
val_losses = []
val_accs = []

best_acc = 0.0

print("-"*50)
print("Training Started")
print("-"*50)

start = time.time()

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)

    ######################################

    model.eval()

    running_val_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            running_val_loss += loss.item()

            preds = outputs.argmax(1)

            correct += (preds == y_batch).sum().item()

            total += y_batch.size(0)

    val_loss = running_val_loss / len(val_loader)

    accuracy = 100 * correct / total

    val_losses.append(val_loss)

    val_accs.append(accuracy)

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {accuracy:.2f}%"
    )

    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            MODEL_DIR/"multimodal_classifier.pt"
        )

end = time.time()

print("-"*50)
print("Training Finished")
print("-"*50)

print(f"Best Validation Accuracy : {best_acc:.2f}%")

print(f"Training Time : {(end-start)/60:.1f} minutes")