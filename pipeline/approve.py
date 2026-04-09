"""Approve and deploy a candidate model after human review.

Reads the pending_approval.json created by retrain.py, replaces the
champion model with the candidate, and updates the dataset registry.

Usage:
    python pipeline/approve.py
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
PIPELINE_DIR = BASE_DIR / "pipeline"

CHAMPION_MODEL_PATH = MODEL_DIR / "model.lgb"
PREVIOUS_MODEL_PATH = MODEL_DIR / "model_previous.lgb"
CANDIDATE_MODEL_PATH = MODEL_DIR / "model_candidate.lgb"
PENDING_APPROVAL_PATH = PIPELINE_DIR / "pending_approval.json"
DATASET_REGISTRY_PATH = BASE_DIR / "data" / "dataset_registry.json"


def main() -> None:
    # Check pending approval exists
    if not PENDING_APPROVAL_PATH.exists():
        print("ERROR: No pending approval found. Run the retraining pipeline first.")
        print(f"  Expected file: {PENDING_APPROVAL_PATH}")
        sys.exit(1)

    with open(PENDING_APPROVAL_PATH) as f:
        approval = json.load(f)

    if approval.get("status") != "pending":
        print(f"ERROR: Approval status is '{approval.get('status')}', not 'pending'.")
        print("  There is no pending model to approve.")
        sys.exit(1)

    candidate_path = Path(approval["candidate_model_path"])
    if not candidate_path.exists():
        print(f"ERROR: Candidate model not found at {candidate_path}")
        sys.exit(1)

    print("Approving model candidate ...")
    print()

    # Archive current champion (if it exists) as model_previous.lgb
    if CHAMPION_MODEL_PATH.exists():
        shutil.copy2(str(CHAMPION_MODEL_PATH), str(PREVIOUS_MODEL_PATH))
        print(f"  Archived champion model -> {PREVIOUS_MODEL_PATH}")

    # Promote candidate to champion
    shutil.copy2(str(candidate_path), str(CHAMPION_MODEL_PATH))
    print(f"  Promoted candidate model -> {CHAMPION_MODEL_PATH}")

    # Update pending_approval.json
    now = datetime.now(timezone.utc).isoformat()
    approval["status"] = "approved"
    approval["approval_timestamp"] = now
    with open(PENDING_APPROVAL_PATH, "w") as f:
        json.dump(approval, f, indent=2)
    print(f"  Updated approval status -> approved")

    # Update dataset_registry.json
    if DATASET_REGISTRY_PATH.exists():
        with open(DATASET_REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {}

    # Increment model version
    current_version = registry.get("model_version", "v0")
    try:
        version_num = int(current_version.lstrip("v")) + 1
    except (ValueError, AttributeError):
        version_num = 1
    new_version = f"v{version_num}"

    registry["model_version"] = new_version
    registry["model_path"] = str(CHAMPION_MODEL_PATH)
    registry["updated_at"] = now
    registry["previous_model_path"] = str(PREVIOUS_MODEL_PATH)
    registry["validation_metrics"] = approval.get("validation_metrics", {})

    with open(DATASET_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Updated dataset registry -> {new_version}")

    print()
    print("Model approved and deployed.")


if __name__ == "__main__":
    main()
