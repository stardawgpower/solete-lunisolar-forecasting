from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "notebook15_final_results_synthesis"


def _true_count(series: pd.Series) -> int:
    return int(series.astype(str).str.lower().eq("true").sum())


def test_notebook15_artifacts_match_frozen_manifest() -> None:
    manifest_path = RESULT_DIR / "notebook15_artifact_sha256_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    assert len(manifest) == 17

    for row in manifest.itertuples(index=False):
        artifact_path = RESULT_DIR / row.artifact

        assert artifact_path.is_file(), f"Missing artifact: {row.artifact}"

        actual_size = artifact_path.stat().st_size
        assert actual_size == int(row.size_bytes), (
            f"Size mismatch for {row.artifact}: "
            f"{actual_size} != {row.size_bytes}"
        )

        actual_sha256 = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

        assert actual_sha256 == row.sha256, (
            f"SHA-256 mismatch for {row.artifact}"
        )


def test_notebook15_scientific_audit_matches_summary_tables() -> None:
    audit = json.loads(
        (
            RESULT_DIR / "notebook15_final_synthesis_audit.json"
        ).read_text()
    )
    science = audit["scientific_audit"]

    manifest = pd.read_csv(
        RESULT_DIR / "notebook15_artifact_sha256_manifest.csv"
    )
    primary = pd.read_csv(
        RESULT_DIR / "table_primary_architecture_summary.csv"
    )
    matched = pd.read_csv(
        RESULT_DIR / "table_matched_cross_dataset_primary.csv"
    )
    mechanism = pd.read_csv(
        RESULT_DIR / "table_mechanism_control_summary.csv"
    )

    assert audit["notebook"] == "15_final_results_synthesis.ipynb"
    assert audit["models_refit"] is False
    assert audit["experiments_rerun"] is False
    assert audit["source_notebooks"] == [
        "12_phase1_final_synthesis.ipynb",
        "13_phase2_f4_representation_analysis.ipynb",
        "14_pvod_external_validation.ipynb",
    ]

    assert audit["final_artifact_count"] == len(manifest)
    assert science["matched_primary_conditions"] == len(matched)

    assert science["exact_primary_classification_agreement"] == _true_count(
        matched["same_classification"]
    )
    assert science["same_resolved_direction"] == _true_count(
        matched["same_resolved_direction"]
    )

    for dataset_key, dataset in (
        ("solete", "SOLETE"),
        ("pvod", "PVOD"),
    ):
        subset = primary.loc[primary["dataset"].eq(dataset)]

        assert science[f"{dataset_key}_primary_F4_lower"] == int(
            subset["primary_F4_lower"].sum()
        )
        assert science[f"{dataset_key}_primary_F3_lower"] == int(
            subset["primary_F3_lower"].sum()
        )
        assert science[f"{dataset_key}_primary_crosses_zero"] == int(
            subset["primary_crosses_zero"].sum()
        )

    for dataset_key, dataset in (
        ("solete", "SOLETE"),
        ("pvod", "PVOD"),
    ):
        subset = mechanism.loc[mechanism["dataset"].eq(dataset)]

        assert science[f"{dataset_key}_alignment_F4_real_lower"] == int(
            subset["null_F4_real_lower"].sum()
        )
        assert science[f"{dataset_key}_alignment_control_lower"] == int(
            subset["null_control_lower"].sum()
        )
        assert science[f"{dataset_key}_alignment_crosses_zero"] == int(
            subset["null_crosses_zero"].sum()
        )

    pvod = mechanism.loc[mechanism["dataset"].eq("PVOD")]

    assert science["pvod_reference_coding_crosses_zero"] == int(
        pvod["reference_crosses_zero"].sum()
    )