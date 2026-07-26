from pathlib import Path
import numpy as np
import open_clip
import torch
from PIL import Image
from PIL import Image
from services.database.review_queue import add_to_review_queue

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "multimodal_classifier.pt"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("-" * 50)
print("Production Predictor")
print("-" * 50)
print("Project :", PROJECT_ROOT)
print("Device  :", DEVICE)

print("\nLoading OpenCLIP...")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model.to(DEVICE)
clip_model.eval()

print("OpenCLIP Loaded")

# --------------------------------------------------
# Multimodal Classifier
# --------------------------------------------------

class MultimodalClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)

print("\nLoading Multimodal Classifier...")

classifier = MultimodalClassifier()

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

classifier.to(DEVICE)
classifier.eval()

print("Classifier Loaded")

LABELS = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen"
]

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(DEVICE)

def encode_image(image_path):
    image = load_image(image_path)

    with torch.no_grad():
        image_embedding = clip_model.encode_image(image)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding

def encode_text(title):

    tokens = tokenizer([title]).to(DEVICE)

    with torch.no_grad():
        text_embedding = clip_model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding

def predict_category(image_path, title):

    image_embedding = encode_image(image_path)
    text_embedding = encode_text(title)

    fused = torch.cat(
        [image_embedding, text_embedding],
        dim=1
    )

    with torch.no_grad():
        logits = classifier(fused)

        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "category": LABELS[prediction.item()],
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze().cpu().numpy()
    }

def predict(image_path, title):

    result = predict_category(image_path, title)

    similarity = image_text_similarity(
        image_path,
        title
    )

    mismatch = similarity < 0.20

    confidence = float(result["confidence"])

    duplicate_score = 0.0

    if (
        confidence < 0.70
        or mismatch
    ):

        add_to_review_queue(
            item_id="UNKNOWN",
            title=title,
            predicted_category=result["category"],
            confidence=confidence,
            image_similarity=similarity,
            duplicate_score=duplicate_score,
            reason="Low confidence or image-text mismatch"
        )

    return {
        "category": result["category"],
        "confidence": round(confidence, 4),
        "image_title_similarity": round(similarity, 4),
        "mismatch": mismatch,
        "probabilities": result["probabilities"].tolist()
    }

def image_text_similarity(image_path, title):

    image_embedding = encode_image(image_path)
    text_embedding = encode_text(title)

    similarity = torch.sum(
        image_embedding * text_embedding,
        dim=1
    ).item()

    return similarity

