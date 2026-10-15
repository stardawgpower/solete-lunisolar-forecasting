"""Append milestone 3 to the EXISTING executed revised Notebook 11 safely.

Keeps every prior cell (including execution outputs) and changes no prior code.
No model training or terminal evaluation is performed by this installer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

M2_COMPLETION_ID = 'm2-revised-neural-completion-gate'
M3_IDS = (
    'm3-fullsize-gates-intro', 'm3-fullsize-gates-policy',
    'm3-fullsize-lstm-intro', 'm3-fullsize-lstm-cell',
    'm3-fullsize-tcn-intro', 'm3-fullsize-tcn-cell',
    'm3-fullsize-completion-gate',
)


def cell(kind: str, cell_id: str, contents: str) -> dict:
    obj = {'cell_type':kind,'id':cell_id,'metadata':{},'source':contents.splitlines(keepends=True)}
    if kind == 'code':
        obj.update({'execution_count':None,'outputs':[]})
    return obj


NEW_CELLS = [
    cell('markdown', M3_IDS[0], """## Milestone 3 — Controlled full-size nested CV gates (disabled by default)

**Not full neural-CV selection.** This milestone enables exactly **one** full-size
15-minute/F0/fold-1 candidate (C1) per architecture, after explicit opt-in.
A 10% chronological inner validation split with a 60-minute label gap selects
an epoch from at most 80 epochs. A **fresh** model is then refitted with a
fresh scaler on all 52,401 outer-training rows for that many epochs, and
scores the 17,856 outer-validation rows once. Neither stage accesses the
June–August 2019 terminal holdout.

**Safety:** LSTM and TCN gates are **both OFF by default**. Running all cells
without changing the switches below will NOT train full-size models.
Each completed gate creates one immutable JSON checkpoint. An interrupted
training run does **not** save a partial score; the fit must then be repeated.
These single-fold results are not architecture rankings.
"""),
    cell('code', M3_IDS[1], """# Preflight and explicit resource-control switches. No model fitting in this cell.
from pathlib import Path
from src.revised_neural_full_cv import (
    GATE_HORIZON, GATE_FAMILY, GATE_FOLD, GATE_TRAIN_ROWS,
    GATE_VALIDATION_ROWS, GATE_KIND, gate_source_fingerprint,
    validate_fullsize_payload, run_full_size_gate,
)

M3_RUN_LSTM = False  # Set True ONLY when ready for a potentially long full-size fit.
M3_RUN_TCN = False   # Keep False until LSTM gate runtime is reviewed.

m3_root = Path(PROJECT_ROOT)
m3_outdir = m3_root / 'results/development_only/revised_neural_fullsize_gate_v1'
m3_payload = neural_fold_smokes[(15, 'F0', 1)]
m3_train_idx, m3_val_idx, m3_inner = validate_fullsize_payload(m3_payload)
assert (len(m3_train_idx), len(m3_val_idx)) == (GATE_TRAIN_ROWS, GATE_VALIDATION_ROWS)
assert (GATE_HORIZON, GATE_FAMILY, GATE_FOLD) == (15, 'F0', 1)
assert not (m3_inner['inner_validation_mask'] & m3_payload['validation_mask']).any()
assert holdout_masks[15].sum() == 26496
m3_source_sha = gate_source_fingerprint(m3_root)
print('NEURAL MILESTONE 3 PREFLIGHT: PASSED')
print('Full-size fold:', len(m3_train_idx), 'train;', len(m3_val_idx), 'outer validation')
print('Inner-fit:', int(m3_inner['inner_train_mask'].sum()),
      'inner-stop:', int(m3_inner['inner_validation_mask'].sum()),
      'embargo:', m3_inner['embargo_rows'])
print('Both full-size model gates disabled by default:', not (M3_RUN_LSTM or M3_RUN_TCN))
print('Development-only checkpoint directory:', m3_outdir)
"""),
    cell('markdown', M3_IDS[2], """### Gate A — LSTM_C1, full-size 15-minute/F0/fold-1 CV

**Opt in explicitly:** change `M3_RUN_LSTM = False` to `True` in the
preflight cell, then rerun the preflight and **only this LSTM gate cell**.
Keep TCN disabled. This may take a considerable amount of time: inner epoch
selection and the fresh outer refit both train on tens of thousands of rows.
No full-grid search is enabled.
"""),
    cell('code', M3_IDS[3], """if M3_RUN_LSTM:
    m3_lstm_record, m3_lstm_new = run_full_size_gate(
        payload=m3_payload, root=m3_root, model_name='LSTM',
        source_sha256=m3_source_sha, output_dir=m3_outdir,
    )
    assert 1 <= m3_lstm_record['best_epoch'] <= 80
    assert m3_lstm_record['outer_training_rows'] == GATE_TRAIN_ROWS
    assert m3_lstm_record['outer_validation_rows'] == GATE_VALIDATION_ROWS
    print('LSTM FULL-SIZE GATE: PASSED', 'new fit' if m3_lstm_new else 'checkpoint reused')
    print('Best inner epoch:', m3_lstm_record['best_epoch'])
    print('Inner selection seconds:', round(m3_lstm_record['inner_selection_seconds'], 1))
    print('Outer refit seconds:', round(m3_lstm_record['outer_refit_seconds'], 1))
    print('Outer fold MAE (kW):', round(m3_lstm_record['validation_mae_kw'], 6))
else:
    print('LSTM FULL-SIZE GATE: DISABLED — no model fit performed')
"""),
    cell('markdown', M3_IDS[4], """### Gate B — TCN_C1, same full-size fold

**Do not enable yet.** Once the LSTM full-size gate finishes and its runtime
and checkpoint have been checked, switch `M3_RUN_TCN = True` in the preflight
cell and run this TCN cell. Do not use the single-fold scores to select an
architecture or open the terminal holdout.
"""),
    cell('code', M3_IDS[5], """if M3_RUN_TCN:
    m3_tcn_record, m3_tcn_new = run_full_size_gate(
        payload=m3_payload, root=m3_root, model_name='TCN',
        source_sha256=m3_source_sha, output_dir=m3_outdir,
    )
    assert 1 <= m3_tcn_record['best_epoch'] <= 80
    assert m3_tcn_record['outer_training_rows'] == GATE_TRAIN_ROWS
    assert m3_tcn_record['outer_validation_rows'] == GATE_VALIDATION_ROWS
    print('TCN FULL-SIZE GATE: PASSED', 'new fit' if m3_tcn_new else 'checkpoint reused')
    print('Best inner epoch:', m3_tcn_record['best_epoch'])
    print('Inner selection seconds:', round(m3_tcn_record['inner_selection_seconds'], 1))
    print('Outer refit seconds:', round(m3_tcn_record['outer_refit_seconds'], 1))
    print('Outer fold MAE (kW):', round(m3_tcn_record['validation_mae_kw'], 6))
else:
    print('TCN FULL-SIZE GATE: DISABLED — no model fit performed')
"""),
    cell('markdown', M3_IDS[6], """## Milestone 3 checkpoint gate — stop here

When the LSTM full-size gate completes, **stop and share its best epoch,
inner-selection time, outer-refit time, and checkpoint outcome** before
running the TCN. The two full-size gates are only resource/correctness checks,
not completed model-selection grids. All 360 complete neural-CV evaluations
and the locked terminal test remain disabled.

**Do not commit the newly generated checkpoints until review**; preserve this
executed notebook's earlier outputs and never overwrite the historical
`11_deep_learning_models.ipynb`.
"""),
]


def append_milestone3(nb: dict) -> bool:
    if nb.get('nbformat') != 4 or len(nb.get('cells', [])) < 22:
        raise ValueError('Unexpected notebook format or missing Milestone 2 cells')
    ids = [c.get('id') for c in nb['cells']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate notebook cell identifiers')
    if set(M3_IDS) & set(ids):
        if tuple(ids[-len(M3_IDS):]) == M3_IDS:
            return False  # idempotent; does not rewrite existing cells
        raise ValueError('Partial or unexpected Milestone 3 append')
    if ids[-1] != M2_COMPLETION_ID:
        raise ValueError('Notebook does not end at frozen Milestone 2 completion gate')
    if len(ids) != 22:
        raise ValueError('Unexpected Milestone 2 cell count')
    m = nb.get('metadata', {}).get('solete_revised_neural_protocol', {})
    if m.get('terminal_test_evaluated') is not False:
        raise ValueError('Terminal holdout metadata must remain false')
    if m.get('milestone2_pilot_executed') is not True:
        raise ValueError('Milestone 2 pilots must be committed/executed first')
    if m.get('development_cv_models_trained') is not False:
        raise ValueError('Development-CV training must not have started')
    if m.get('phase') != 'milestone2-inner-split-and-resource-pilot-complete':
        raise ValueError('Milestone 2 notebook phase mismatch')
    nb['cells'].extend(NEW_CELLS)
    m.update({'milestone3_fullsize_lstm_gate_executed': False,
              'milestone3_fullsize_tcn_gate_executed': False,
              'phase':'milestone3-fullsize-single-fold-gates-disabled-pending-execution',
              'terminal_test_evaluated':False})
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path,
                        default=Path('notebooks/11_revised_neural_models.ipynb'))
    args = parser.parse_args()
    path = args.path
    original = path.read_text(encoding='utf-8')
    nb = json.loads(original)
    if append_milestone3(nb):
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n',encoding='utf-8')
        print('MILESTONE 3 APPENDED: PASSED — prior notebook cells and outputs preserved')
    else:
        print('MILESTONE 3 ALREADY INSTALLED — notebook unchanged')
    print('Both full-size gates are OFF by default; no model was trained.')


if __name__ == '__main__':
    main()
