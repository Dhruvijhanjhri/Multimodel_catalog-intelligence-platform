from pathlib import Path

from evaluate_model import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "multimodal_classifier.pt"
)


def compare_models(candidate_path):

    active_results = evaluate_model(ACTIVE_MODEL)
    candidate_results = evaluate_model(candidate_path)

    active_validation = active_results["validation"]["accuracy"]
    candidate_validation = candidate_results["validation"]["accuracy"]

    active_test = active_results["test"]["accuracy"]
    candidate_test = candidate_results["test"]["accuracy"]

    validation_pass = (
        candidate_validation >= active_validation
    )

    test_pass = (
        candidate_test >= active_test
    )

    return {
        "active_model": {
            "path": str(ACTIVE_MODEL),
            "metrics": active_results,
        },
        "candidate_model": {
            "path": str(candidate_path),
            "metrics": candidate_results,
        },
        "comparison": {
            "validation_pass": validation_pass,
            "test_pass": test_pass,
            "candidate_acceptable": (
                validation_pass and test_pass
            ),
        },
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage: python scripts/compare_models.py "
            "<candidate_model_path>"
        )

        raise SystemExit(1)

    candidate_path = Path(sys.argv[1])

    if not candidate_path.exists():

        print(
            f"Candidate model not found: {candidate_path}"
        )

        raise SystemExit(1)

    results = compare_models(candidate_path)

    print()
    print("-" * 50)
    print("Model Comparison")
    print("-" * 50)

    print("Active Model:")
    print(
        results["active_model"]["metrics"]
    )

    print()
    print("Candidate Model:")
    print(
        results["candidate_model"]["metrics"]
    )

    print()
    print("Comparison:")
    print(
        results["comparison"]
    )
