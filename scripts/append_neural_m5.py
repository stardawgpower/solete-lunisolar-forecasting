"""Append Milestone 5 opt-in batch controls, preserving earlier notebook outputs.

Run from the SOLETE repository root: python scripts/append_neural_m5.py
"""
from __future__ import annotations
import ast
import json
from pathlib import Path

CELL_ID = "m5-neural-batch-safe-run"
INTRO_ID = "m5-neural-batch-intro"
OUTRO_ID = "m5-neural-batch-outro"
FROZEN_REQUIRED_IDS = (
    "m4-campaign-preflight", "m4-campaign-policy", "m4-campaign-fit",
    "m4-run-all-summary",
)


def _markdown(cell_id: str, content: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": content.splitlines(keepends=True)}


def _code(cell_id: str, content: str) -> dict:
    ast.parse(content)
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": content.splitlines(keepends=True)}


M5_CELL_SOURCE = '''# Milestone 5: opt-in bounded batch of SOURCE/DATA-VERIFIED development CV fits.
# IMPORTANT: changing the switch to True can launch expensive training.
M5_RUN_BATCH = False
M5_MAX_NEW_FITS = 1              # First live launch: ONE new evaluation only.
M5_MODELS = ('LSTM',)            # Later: ('LSTM', 'TCN') for all 360.

from pathlib import Path
from src.revised_neural_batch import run_checkpointed_batch
from src.revised_neural_cv import prepare_development_fold
from src.forecasting_protocol import make_rolling_origin_masks

assert globals().get('m4_preflight_passed', False), 'Run data setup and M4 preflight first'
assert M3_RUN_LSTM is False and M3_RUN_TCN is False
assert M4_RUN_ONE is False


def m5_payload_for(horizon, family, fold_number):
    base = base_modelling_tables[horizon]
    forbidden = holdout_masks[horizon]
    assert int(forbidden.sum()) == 26496
    masked = base['target_P_Solar_kW'].mask(forbidden)
    assert masked.loc[forbidden].isna().all()
    fold = make_rolling_origin_masks(base['valid_time'])[fold_number - 1]
    return prepare_development_fold(
        X=feature_family_tables[horizon][family],
        masked_target=masked,
        issue_time=base['issue_time'], valid_time=base['valid_time'],
        forbidden_mask=forbidden, fold=fold,
        horizon_minutes=horizon, family=family,
    )

m5_report = run_checkpointed_batch(
    root=PROJECT_ROOT,
    output_dir=Path(PROJECT_ROOT) / 'results/development_only/revised_neural_cv_v1',
    payload_factory=m5_payload_for,
    enabled=M5_RUN_BATCH,
    max_new_fits=M5_MAX_NEW_FITS,
    models=M5_MODELS,
)
if not M5_RUN_BATCH:
    assert m5_report['new_fits'] == 0
    print('NEURAL MILESTONE 5 BATCH: DISABLED — no training executed')
    print('Completed M4 archive records:', m5_report['after'], 'of 360')
'''

M5_INTRO = '''## Milestone 5 — Restart-safe full neural CV batch orchestration

Development only. The original Milestone 4 model implementation, 360 candidate/fold
identities, source/data fingerprinting and the locked terminal holdout are unchanged.
This section is an **optional scheduler**, not a new training or selection protocol.

**Default Run All is safe:** `M5_RUN_BATCH=False` and a cap of one new fit.
When explicitly enabling this cell, execute **only this cell** after Sections A–C
and Milestone 4 preflight have passed. Each fit may take minutes or more.

The two Milestone 3 checkpointed feasibility gates cannot be silently promoted
to model-selection results because their implementation fingerprints differ.
The first Milestone 5 fit will therefore recompute LSTM_C1/15min/F0/fold1
under the campaign's frozen implementation. This is intentional provenance.

Completed CV records are immutable and reused only after exact data/source
identity checks. Interrupted training has no checkpoint and must be rerun.
Start with `M5_MODELS=('LSTM',)` and `M5_MAX_NEW_FITS=1`. Once the first
campaign record passes audit, grow the cap gradually. For all 360 eventually,
use `M5_MODELS=('LSTM', 'TCN')`; avoid running concurrent notebook kernels.
Never evaluate terminal test targets during selection.
'''

M5_OUTRO = '''### Milestone 5 execution boundary

`MILESTONE 5 BATCH: PASSED` means the **bounded batch** completed, not necessarily
that the 360-evaluation selection study finished. A full LSTM/TCN 15×4×3 grid
is complete only when `read_campaign_progress` has exactly 360 valid records
and `complete_rankings` returns all 30 model/horizon/family blocks.

Before committing, verify checkpoint completeness, freeze selected configurations,
and confirm source/data integrity. Never interpret Milestone 3 single-fold gate
scores or incomplete candidate grids as final winners. No terminal refits/tests.
'''


def append_m5(nb: dict) -> bool:
    cells = nb.get("cells", [])
    ids = [c.get("id") for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate notebook cell identifiers")
    expected = (INTRO_ID, CELL_ID, OUTRO_ID)
    existing = [name for name in expected if name in ids]
    if existing:
        if len(existing) == len(expected) and ids[-3:] == list(expected) and "M5_RUN_BATCH = False" in "".join(cells[-2]["source"]):
            return False
        raise ValueError("Conflicting/incomplete Milestone 5 cells already exist")
    if any(name not in ids for name in FROZEN_REQUIRED_IDS):
        raise ValueError("Missing required frozen Milestone 4 cell identifiers")
    if ids.index("m4-campaign-fit") > ids.index("m4-run-all-summary"):
        raise ValueError("Unexpected Milestone 4 ordering")
    meta = nb.setdefault("metadata", {}).setdefault("solete_revised_neural_protocol", {})
    if meta.get("terminal_test_evaluated") is not False:
        raise ValueError("Refusing to append when terminal-test provenance is not false")
    nb["cells"].extend([
        _markdown(INTRO_ID, M5_INTRO),
        _code(CELL_ID, M5_CELL_SOURCE),
        _markdown(OUTRO_ID, M5_OUTRO),
    ])
    meta["phase"] = "milestone5-batch-scheduler-installed-not-executed"
    meta["milestone5_full_cv_complete"] = False
    meta["milestone5_batch_default_enabled"] = False
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "11_revised_neural_models.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = append_m5(notebook)
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("NEURAL MILESTONE 5 APPENDED: PASSED — all earlier outputs preserved")
    else:
        print("NEURAL MILESTONE 5 ALREADY INSTALLED: PASSED — no changes")
    print("M5_RUN_BATCH=False: no neural training will start on Run All")


if __name__ == "__main__":
    main()
