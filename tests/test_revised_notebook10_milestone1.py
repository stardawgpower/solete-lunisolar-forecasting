"""Static regression gates for revised Notebook 10 Milestone 1.

The tests do not need SOLETE HDF5 or DE440s, and do not train models.
Dynamic numerical checks execute when the notebook runs on the local data.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "notebooks" / "09_modelling_dataset_and_protocol.ipynb"
REVISED = ROOT / "notebooks" / "10_revised_classical_models.ipynb"
FROZEN_CONSTRUCTION_HASHES = (
    "18d232021455524939be8e14fc67842819ea4b8701a41a279b8997a47768bbee",
    "6add41c3649cfd7300a1fa04e0a966f30f7fc78c2a4b4bbaa054056358ddc21b",
    "724b7cdf9cd01790c9a0c9371967ff1c264c0f0d9730d1ef8d9be30dd359456d",
    "303947573618f2bcc5c280e2dd6e2d392805dc194073d51a8e4c84c7b4f3a373",
    "58ab56827f81c210df0ec662208f217baa3043d22c1940b028b041407efa9f09",
)



def _read_archived_notebook(path: Path) -> str:
    """Read the original notebook, including in consolidated clones."""
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", f"056a346:notebooks/{path.name}"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_notebook(path: Path) -> dict:
    return json.loads(_read_archived_notebook(path))


def _code_cells(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_revised_notebook_construction_cells_match_notebook09_exactly():
    reference = _code_cells(_load_notebook(REFERENCE))
    revised = _code_cells(_load_notebook(REVISED))
    assert len(reference) >= 5
    assert len(revised) >= 9
    assert tuple(
        hashlib.sha256(code.encode("utf-8")).hexdigest()
        for code in reference[:5]
    ) == FROZEN_CONSTRUCTION_HASHES
    assert revised[1:6] == reference[:5]
    assert tuple(
        hashlib.sha256(code.encode("utf-8")).hexdigest()
        for code in revised[1:6]
    ) == FROZEN_CONSTRUCTION_HASHES


def test_revised_notebook_has_no_stale_results_dependencies():
    source = "\n".join(_code_cells(_load_notebook(REVISED)))
    for forbidden in (
        "notebook09_primary_experiment",
        "notebook10_secondary_ml_models",
        "primary_protocol",
        "split_labels[",
        "assign_solete_split(",
    ):
        assert forbidden not in source, f"Stale dependency: {forbidden}"
    for required in (
        "make_rolling_origin_masks",
        "make_terminal_refit_mask",
        "terminal_holdout_mask",
        "validate_rolling_origin_counts",
        "pd.testing.assert_frame_equal",
        "encode_panchang_fixed_vocabulary",
    ):
        assert required in source, f"Missing safety check: {required}"


def test_milestone1_original_cells_have_no_training_or_file_writes():
    forbidden_methods = {
        "fit", "partial_fit", "fit_transform", "predict", "predict_proba",
        "to_csv", "to_parquet", "to_pickle", "to_json", "to_hdf",
        "dump", "dumps", "write_text", "write_bytes", "save_model",
    }
    for source in _code_cells(_load_notebook(REVISED))[:9]:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_methods, (
                    f"Milestone 1 must be read-only: .{node.func.attr}()"
                )


def test_revised_notebook_compiles_and_has_no_execution_errors():
    notebook = _load_notebook(REVISED)
    code_index = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code_index += 1
            assert not any(
                output.get("output_type") == "error"
                for output in cell.get("outputs", [])
            )
            compile("".join(cell["source"]), f"cell-{code_index}", "exec")
    assert code_index >= 9


def test_milestone2_ridge_pilot_is_development_only():
    code = _code_cells(_load_notebook(REVISED))
    assert len(code) >= 10
    pilot = code[9]
    assert "evaluate_solete_cv_candidates(" in pilot
    assert "build_ridge_model(" in pilot
    assert 'PILOT_FAMILY = "F0"' in pilot
    assert 'PILOT_HORIZON = 15' in pilot
    assert 'pilot_y = pilot_base["target_P_Solar_kW"].mask(' in pilot
    assert "holdout_masks[PILOT_HORIZON]" in pilot
    for disallowed in (".to_csv(", ".to_parquet(", ".predict(", ".to_json("):
        assert disallowed not in pilot


def test_complete_ridge_cv_is_development_only_and_preserves_pilot():
    notebook = _load_notebook(REVISED)
    code = [s for s in _code_cells(notebook) if s.strip()]
    assert len(code) >= 11
    full = code[10]
    for required in (
        'RIDGE_HORIZONS = (15, 30, 60)',
        'RIDGE_FAMILIES = ("F0", "F1", "F2", "F3", "F4")',
        'ridge_dev_y = ridge_base["target_P_Solar_kW"].mask(ridge_holdout)',
        'ridge_pilot_fold_scores.copy(deep=True)',
        'evaluate_solete_cv_candidates(',
        'EXPECTED_CV_COUNTS_BY_HORIZON[ridge_horizon]',
        'ridge_cv_fold_scores',
        'ridge_cv_selected',
    ):
        assert required in full
    for forbidden in (
        '.fit(', '.predict(', '.to_csv(', '.to_parquet(',
        '.to_json(', '.write_text(', 'make_terminal_refit_mask(',
        'results/notebook',
    ):
        assert forbidden not in full
    protocol = notebook["metadata"]["solete_revised_protocol"]
    assert protocol["terminal_test_evaluated"] is False


def test_ridge_calendar_diagnostic_is_development_only():
    notebook = _load_notebook(REVISED)
    code = [s for s in _code_cells(notebook) if s.strip()]
    assert len(code) >= 12
    diagnostic = code[11]
    for required in (
        "DIAG_HORIZON = 60",
        "DIAG_ALPHA = 100.0",
        'diag_y = diag_base["target_P_Solar_kW"].mask(diag_holdout)',
        '"F0_plus_basic"',
        '"F0_plus_cyclic"',
        'cv_masks[DIAG_HORIZON][fold_number]',
        'diag_base.loc[validation, "issue_time"].min()',
        "build_ridge_model(alpha=DIAG_ALPHA)",
        "ridge_calendar_diagnostic_summary",
        "Ridge calendar-feature diagnostic: PASSED",
    ):
        assert required in diagnostic
    for forbidden in (
        ".to_csv(", ".to_parquet(", ".to_json(",
        ".write_text(", ".write_bytes(",
        "make_terminal_refit_mask(",
        "results/notebook",
    ):
        assert forbidden not in diagnostic
    protocol = notebook["metadata"]["solete_revised_protocol"]
    assert protocol["terminal_test_evaluated"] is False
