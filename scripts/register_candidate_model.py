import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.database.model_registry import register_model_version


register_model_version(
    model_name="multimodal_classifier",
    version="multimodal_classifier_candidate_20261007",
    model_type="PyTorch multimodal classifier",
    artifact_path="models/candidates/multimodal_classifier_candidate_20261007.pt",
    metrics={
        "validation_accuracy": 0.9817097415506958,
        "validation_correct": 4938,
        "validation_total": 5030,
        "test_accuracy": 0.9795269330153051,
        "test_correct": 4928,
        "test_total": 5031,
        "status": "candidate_rejected",
        "promotion_eligible": False
    }
)
