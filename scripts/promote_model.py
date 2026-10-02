import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.database.model_registry import (
    get_active_model_version,
    promote_model_version,
)

MODEL_NAME = "multimodal_classifier"


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_model.py <candidate_version>")
        raise SystemExit(1)

    candidate_version = sys.argv[1]

    active_model = get_active_model_version(MODEL_NAME)

    print(f"Currently active: {active_model['version']}")
    print(f"Requested candidate: {candidate_version}")

    confirmation = input(
        f"Type PROMOTE {candidate_version} to continue: "
    )

    if confirmation != f"PROMOTE {candidate_version}":
        print("Promotion cancelled.")
        return

    promote_model_version(MODEL_NAME, candidate_version)


if __name__ == "__main__":
    main()