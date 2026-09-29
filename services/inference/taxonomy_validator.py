from pathlib import Path

import numpy as np
import faiss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "train_image_embeddings.npy"
)

TEXT_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "train_text_embeddings.npy"
)

LABELS_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "train_labels.npy"
)


LABELS = [
    "Electronics_Accessories",
    "Fashion_Travel",
    "Footwear",
    "Furniture",
    "Hardware_HomeImprovement",
    "Home_Kitchen"
]


TOP_K = 5

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.10


print()
print("-" * 60)
print("Taxonomy Validator")
print("-" * 60)


print()
print("Loading catalog embeddings...")

image_embeddings = np.load(
    IMAGE_EMBEDDINGS_FILE
)

text_embeddings = np.load(
    TEXT_EMBEDDINGS_FILE
)

catalog_labels = np.load(
    LABELS_FILE
)


print(
    "Image embeddings :",
    image_embeddings.shape
)

print(
    "Text embeddings  :",
    text_embeddings.shape
)

print(
    "Catalog labels    :",
    catalog_labels.shape
)


print()
print("Creating multimodal catalog embeddings...")

catalog_embeddings = (
    image_embeddings
    + text_embeddings
) / 2.0


norms = np.linalg.norm(
    catalog_embeddings,
    axis=1,
    keepdims=True
)

catalog_embeddings = (
    catalog_embeddings
    / np.maximum(
        norms,
        1e-12
    )
)

catalog_embeddings = np.ascontiguousarray(
    catalog_embeddings,
    dtype=np.float32
)


print(
    "Catalog embeddings :",
    catalog_embeddings.shape
)


print()
print("Building category-specific FAISS indexes...")

CATEGORY_INDEXES = {}

for category_id, category_name in enumerate(LABELS):

    category_indices = np.where(
        catalog_labels == category_id
    )[0]

    category_vectors = (
        catalog_embeddings[
            category_indices
        ]
    )

    category_index = faiss.IndexFlatIP(
        category_vectors.shape[1]
    )

    category_index.add(
        category_vectors
    )

    CATEGORY_INDEXES[
        category_name
    ] = category_index

    print(
        f"{category_name}:",
        len(category_indices),
        "products"
    )


print()
print("-" * 60)
print("Taxonomy Validator Loaded")
print("-" * 60)


def calculate_category_support(
    combined_embedding,
    category_name
):
    """
    Calculate top-k catalog similarity
    for one category.
    """

    category_index = CATEGORY_INDEXES[
        category_name
    ]

    search_k = min(
        TOP_K,
        category_index.ntotal
    )

    scores, _ = category_index.search(
        combined_embedding,
        search_k
    )

    if len(scores[0]) == 0:
        return 0.0

    return float(
        np.mean(
            scores[0]
        )
    )


def validate_taxonomy(
    image_embedding,
    text_embedding,
    predicted_category
):
    """
    Compare the predicted category against
    all existing taxonomy categories.

    Returns:
        predicted support
        best alternative support
        support margin
        category support scores
    """

    if predicted_category not in LABELS:

        return {
            "predicted_category": predicted_category,
            "catalog_support": 0.0,
            "best_alternative_category": None,
            "best_alternative_support": 0.0,
            "support_margin": 0.0,
            "category_supports": {},
            "validation_status": "Invalid Category",
            "possible_taxonomy_gap": True
        }


    image_embedding = np.asarray(
        image_embedding,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    text_embedding = np.asarray(
        text_embedding,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    combined_embedding = (
        image_embedding
        + text_embedding
    ) / 2.0


    norm = np.linalg.norm(
        combined_embedding,
        axis=1,
        keepdims=True
    )

    combined_embedding = (
        combined_embedding
        / np.maximum(
            norm,
            1e-12
        )
    )

    combined_embedding = np.ascontiguousarray(
        combined_embedding,
        dtype=np.float32
    )


    category_supports = {}

    for category_name in LABELS:

        category_supports[
            category_name
        ] = round(
            calculate_category_support(
                combined_embedding,
                category_name
            ),
            4
        )


    predicted_support = category_supports[
        predicted_category
    ]


    alternatives = {
        category: score
        for category, score
        in category_supports.items()
        if category != predicted_category
    }


    best_alternative_category = max(
        alternatives,
        key=alternatives.get
    )

    best_alternative_support = alternatives[
        best_alternative_category
    ]


    support_margin = (
        predicted_support
        - best_alternative_support
    )


    return {
        "predicted_category": predicted_category,
        "catalog_support": predicted_support,
        "best_alternative_category": best_alternative_category,
        "best_alternative_support": round(
            best_alternative_support,
            4
        ),
        "support_margin": round(
            support_margin,
            4
        ),
        "category_supports": category_supports,
        "validation_status": "Needs Calibration",
        "possible_taxonomy_gap": False
    }


def determine_taxonomy_status(
    confidence,
    support_margin,
    mismatch
):
    """
    Determine whether the AI prediction can pass
    automatically or should go to human review.

    Important:
        Low confidence, low margin, or mismatch
        indicate uncertainty.

        They do NOT automatically mean that the
        product is outside the taxonomy.

        A taxonomy gap must be confirmed by a
        human reviewer.
    """

    confidence = float(
        confidence
    )

    support_margin = float(
        support_margin
    )

    mismatch = bool(
        mismatch
    )


    if (
        confidence >= CONFIDENCE_THRESHOLD
        and support_margin >= MARGIN_THRESHOLD
        and not mismatch
    ):

        return "PASS"


    return "REVIEW"