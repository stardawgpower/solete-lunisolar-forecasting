"""Static regression guard for the development-only tree-model pilot."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "10_revised_classical_models.ipynb"



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


def test_tree_resource_pilot_remains_development_only():
    notebook = json.loads(_read_archived_notebook(NOTEBOOK))
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    ]
    assert len(code) >= 13
    source = code[12]
    ast.parse(source)
    for expected in (
        'TREE_PILOT_HORIZON = 60',
        'TREE_PILOT_FAMILY = "F4"',
        'TREE_PILOT_FOLD = 1',
        'tree_y = tree_base["target_P_Solar_kW"].mask(tree_holdout)',
        'evaluate_cv_candidates(',
        'forbidden_mask=tree_holdout',
        'configurations=(dict(candidate),)',
        'build_random_forest_model',
        'build_lightgbm_model',
        'Tree-model resource pilot: PASSED',
    ):
        assert expected in source
    for forbidden in (
        '.to_csv(', '.to_parquet(', '.to_pickle(', '.to_json(',
        '.write_text(', '.write_bytes(', 'make_terminal_refit_mask(',
        'results/notebook',
    ):
        assert forbidden not in source
    assert (
        notebook["metadata"]["solete_revised_protocol"]["terminal_test_evaluated"]
        is False
    )
