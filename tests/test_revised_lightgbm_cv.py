"""Regression guards for the full development-only LightGBM CV cell."""
from __future__ import annotations
import ast
import hashlib
import json
import subprocess
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "10_revised_classical_models.ipynb"
FROZEN_PILOT_HASH = "737666043670cb6718e6c97a9da81249b206e6543ccd77608867126e0b90777b"



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


def _notebook():
    return json.loads(_read_archived_notebook(NOTEBOOK))


def test_existing_executed_pilot_remains_unchanged():
    nb = _notebook()
    pilot = next(c for c in nb["cells"] if c.get("id") == "m3-tree-resource-pilot")
    src = "".join(pilot["source"])
    assert hashlib.sha256(src.encode()).hexdigest() == FROZEN_PILOT_HASH
    assert any("Tree-model resource pilot: PASSED (2 development-only fits)" in "".join(out.get("text", []))
               for out in pilot.get("outputs", []) if out.get("output_type") == "stream")


def test_complete_lightgbm_cv_remains_development_only():
    nb = _notebook()
    cells = [c for c in nb["cells"] if c.get("id") == "m4-lightgbm-full-cv"]
    assert len(cells) == 1
    cell = cells[0]
    assert cell["cell_type"] == "code"
    source = "".join(cell["source"])
    ast.parse(source)
    for required in (
        "LIGHTGBM_CV_HORIZONS = (15, 30, 60)",
        'LIGHTGBM_CV_FAMILIES = ("F0", "F1", "F2", "F3", "F4")',
        "LIGHTGBM_CONFIGS", "build_lightgbm_model",
        'lgb_y = lgb_base["target_P_Solar_kW"].mask(lgb_holdout)',
        'lgb_folds = make_rolling_origin_masks(lgb_base["valid_time"])',
        "evaluate_cv_candidates(", "folds=lgb_folds,",
        "configurations=lightgbm_grid,", "forbidden_mask=lgb_holdout,",
        "EXPECTED_CV_COUNTS_BY_HORIZON", "len(lightgbm_cv_fold_scores) == 3 * 5 * 8 * 3 == 360",
        "Complete LightGBM rolling-CV selection: PASSED",
    ):
        assert required in source
    for forbidden in (
        "make_terminal_refit_mask(", "terminal_holdout_mask(",
        ".to_csv(", ".to_parquet(", ".to_json(", ".to_pickle(",
        ".write_text(", ".write_bytes(", "results/", "pickle.dump(",
        "RANDOM_FOREST_CONFIGS", "build_random_forest_model",
    ):
        assert forbidden not in source
    # The regression guard must remain valid both before and after execution.
    if cell["execution_count"] is None:
        assert not cell["outputs"]
    else:
        assert any("Complete LightGBM rolling-CV selection: PASSED" in "".join(o.get("text", []))
                   for o in cell.get("outputs", []) if o.get("output_type") == "stream")
    assert nb["metadata"]["solete_revised_protocol"]["terminal_test_evaluated"] is False
    # The LightGBM CV cell itself writes no artifacts (guarded above), but
    # later Notebook 10 sections intentionally archive development-only results.
    # Therefore the notebook-wide no_result_writes flag must be False.
    assert nb["metadata"]["solete_revised_protocol"]["no_result_writes"] is False
