#!/usr/bin/env python3
"""Read-only post-refit integrity audit for all three frozen SOLETE LightGBM models.

Install as scripts/verify_all_three_final_refits.py and run from the repository:
    python scripts/verify_all_three_final_refits.py

Does NOT read SOLETE HDF5/terminal data; does NOT fit, predict, score, or write files.
The data-input SHA values are provenance assertions from the refit records, not
independently recomputed here from SOLETE source measurements.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import re
import subprocess
from pathlib import Path


FROZEN_LOCK_COMMIT = "c6a9162"
LOCK_REL = "results/development_only/revised_model_selection_lock_v1.json"
ARCHIVE_REL = "results/final_model_refits/revised_lightgbm_v1"
EXPECTED_ARTIFACTS = {
    15: {
        "family": "F1", "rows": 104817, "columns": 133,
        "sha256": "aeadd57863612e732c6b98972c2be80b60037dec0769331aab6e1e4c92d17bdf",
    },
    30: {
        "family": "F2", "rows": 104814, "columns": 138,
        "sha256": "e841a2aa4c2ef5a19ba3c8c9ec708eea5e878c1d2c29ea625234d259eecf6eef",
    },
    60: {
        "family": "F2", "rows": 104808, "columns": 138,
        "sha256": "aac4d7e3c1c84495f494f783c930ce0917d35e2441356220afffb4eb68c0f023",
    },
}
EXPECTED_PARAMETERS = {"learning_rate": 0.03, "n_estimators": 300, "num_leaves": 15}
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError(reason)


def run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=check)


def root_from_script() -> Path:
    for path in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent):
        if (path / ".git").exists() and (path / LOCK_REL).is_file():
            return path.resolve()
    raise FileNotFoundError("Run this script from the SOLETE repository checkout")


def main() -> None:
    root = root_from_script()
    require(run_git(root, "merge-base", "--is-ancestor", FROZEN_LOCK_COMMIT, "HEAD",
                    check=False).returncode == 0,
            f"Frozen selection commit {FROZEN_LOCK_COMMIT} is not a HEAD ancestor")
    lock_bytes = (root / LOCK_REL).read_bytes()
    require(lock_bytes == run_git(root, "show", f"{FROZEN_LOCK_COMMIT}:{LOCK_REL}").stdout,
            "The model-selection lock differs from its frozen committed version")
    lock = json.loads(lock_bytes)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    require(lock["evaluation_policy"]["terminal_access_authorized"] is False,
            "Unexpected terminal evaluation authorization")
    require(lock["archived_audit"]["archived_fold_scores"] == 1665,
            "Unexpected selection campaign size")

    for source_path, expected_blob_sha in lock["reproducibility"]["git_blob_shas"].items():
        require((root / source_path).is_file(), f"Missing frozen source: {source_path}")
        actual_blob_sha = run_git(root, "hash-object", source_path).stdout.decode().strip()
        require(actual_blob_sha == expected_blob_sha, f"Frozen source modified: {source_path}")

    pins = lock["reproducibility"]["dependencies_pinned_from_pyproject_toml"]
    for package, pinned_version in pins.items():
        require(metadata.version(package) == pinned_version,
                f"Dependency mismatch: {package} != {pinned_version}")

    # Loading native model files is for structural validation only; no predictions.
    import lightgbm as lgb

    selections = {int(x["horizon_minutes"]): x for x in lock["selected_candidates"]}
    require(set(selections) == set(EXPECTED_ARTIFACTS), "Unexpected locked horizon list")
    model_dir = root / ARCHIVE_REL
    require(model_dir.is_dir(), f"Missing final model archive folder: {model_dir}")
    schema_hashes: dict[int, str] = {}
    input_hashes: dict[int, str] = {}
    for horizon, expected in EXPECTED_ARTIFACTS.items():
        selection = selections[horizon]
        require(selection["model"] == "LightGBM" and selection["candidate_id"] == "LightGBM_C1",
                f"h{horizon}: model candidate mismatch")
        require(selection["feature_family"] == expected["family"],
                f"h{horizon}: feature family mismatch")
        require(selection["configuration"] == EXPECTED_PARAMETERS,
                f"h{horizon}: frozen hyperparameters mismatch")
        require(selection["refit"]["training_rows_expected"] == expected["rows"],
                f"h{horizon}: frozen refit row count mismatch")
        stem = f"lightgbm_c1_h{horizon}_{expected['family']}"
        model_file = model_dir / (stem + ".txt")
        record_file = model_dir / (stem + ".json")
        require(model_file.is_file() and record_file.is_file(),
                f"h{horizon}: one or both final model files are absent")
        record = json.loads(record_file.read_text(encoding="utf-8"))
        identity = record.get("training_identity")
        require(isinstance(identity, dict), f"h{horizon}: missing training identity")
        require(record.get("schema_version") == 1, f"h{horizon}: record schema mismatch")
        require(record.get("kind") == "SOLETE_REVISED_FINAL_DEVELOPMENT_LIGHTGBM_REFIT",
                f"h{horizon}: unexpected refit type")
        require(record.get("terminal_evaluated") is False and
                record.get("predictions_generated") is False and
                record.get("early_stopping_used") is False,
                f"h{horizon}: unexpected evaluation/early stopping state")
        require(record.get("dependencies") == pins, f"h{horizon}: dependency record mismatch")
        require(record.get("fitted_boosting_iterations") == 300,
                f"h{horizon}: wrong recorded boosting-round count")
        require(record.get("model_filename") == model_file.name,
                f"h{horizon}: model filename mismatch")
        require(record.get("model_size_bytes") == model_file.stat().st_size,
                f"h{horizon}: model byte size mismatch")
        actual_digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        require(actual_digest == expected["sha256"],
                f"h{horizon}: actual model SHA differs from the executed refit notebook")
        require(record.get("model_sha256") == actual_digest,
                f"h{horizon}: model SHA differs from its provenance record")

        fixed_identity = {
            "horizon_minutes": horizon,
            "model": "LightGBM",
            "candidate_id": "LightGBM_C1",
            "feature_family": expected["family"],
            "configuration": EXPECTED_PARAMETERS,
            "fixed_parameters": lock["refit_protocol"]["lightgbm_fixed_parameters_from_frozen_source"],
            "training_rows": expected["rows"],
            "feature_columns": expected["columns"],
            "lock_commit": FROZEN_LOCK_COMMIT,
            "lock_sha256": lock_sha,
            "last_label_valid_time_utc": "2019-05-31T22:55:00+00:00",
        }
        for key, value in fixed_identity.items():
            require(identity.get(key) == value, f"h{horizon}: training identity field {key} mismatch")
        for key in ("feature_schema_sha256", "training_input_sha256"):
            require(isinstance(identity.get(key), str) and bool(HEX_SHA.fullmatch(identity[key])),
                    f"h{horizon}: invalid {key}")
        schema_hashes[horizon] = identity["feature_schema_sha256"]
        input_hashes[horizon] = identity["training_input_sha256"]
        source_commit = record.get("fitted_on_checkout_head")
        require(isinstance(source_commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", source_commit)),
                f"h{horizon}: missing training checkout commit")
        require(run_git(root, "merge-base", "--is-ancestor", FROZEN_LOCK_COMMIT,
                        source_commit, check=False).returncode == 0,
                f"h{horizon}: training checkout does not descend from selection lock")

        booster = lgb.Booster(model_file=str(model_file))  # DOES NOT train or predict
        require(booster.current_iteration() == 300,
                f"h{horizon}: native model has unexpected iteration count")
        require(booster.num_feature() == expected["columns"],
                f"h{horizon}: native model feature count mismatch")
        require(len(booster.feature_name()) == expected["columns"],
                f"h{horizon}: native model feature name count mismatch")
        print(f"{horizon:>2} min: VERIFIED — {expected['family']} — "
              f"{expected['rows']:,} development rows — "
              f"{expected['columns']} features — "
              f"300 trees — sha256={actual_digest}")

    require(schema_hashes[30] == schema_hashes[60],
            "F2 feature-schema fingerprints disagree across horizons")
    require(len(set(input_hashes.values())) == 3,
            "Expected distinct training-input fingerprints by horizon")
    print("ALL THREE FINAL LIGHTGBM REFITS: VERIFIED — source, lock, model bytes, metadata, native structure")
    print("NO HDF5 READS | NO TRAINING | NO PREDICTIONS | NO SCORING | NO FILE WRITES")
    print("NOTE: The original refit notebooks recomputed each development-input SHA."
          " This stand-alone audit verifies the recorded hashes' consistency without rereading SOLETE.")


if __name__ == "__main__":
    main()
