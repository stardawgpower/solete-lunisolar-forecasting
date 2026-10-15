"""Add Milestone 4 to the *current* executed Notebook 11, preserving prior cells.

Safe with notebook copies containing the user's post-gate scratch/audit cells.
No training occurs during installation. Idempotent on repeated execution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

IDS = (
    'm4-campaign-intro','m4-campaign-preflight',
    'm4-campaign-policy','m4-campaign-fit','m4-campaign-outro',
)
M3_IDS = (
    'm3-fullsize-gates-intro','m3-fullsize-gates-policy',
    'm3-fullsize-lstm-intro','m3-fullsize-lstm-cell',
    'm3-fullsize-tcn-intro','m3-fullsize-tcn-cell',
    'm3-fullsize-completion-gate',
)


def cell(kind, identifier, content):
    obj = dict(cell_type=kind, id=identifier, metadata={},
               source=content.splitlines(keepends=True))
    if kind == 'code':
        obj.update(execution_count=None, outputs=[])
    return obj


NEW = [
    cell('markdown',IDS[0],'''## Milestone 4 — Restart-safe neural development-CV campaign

This milestone enables a **single** candidate/outer-fold at a time (opt-in
only). Every completed run writes an immutable, source/data-fingerprinted
checkpoint, then subsequent calls reuse that exact record. The entire 360-fit
campaign is **disabled by default**; there is no auto-loop. Partial/interrupted
fits do not count as results. Do not use single-fold feasibility scores as
hyperparameter selections or evaluate the June–August 2019 terminal holdout.

**Provenance:** the two Milestone 3 gates are read-only audited separately;
they are NOT silently copied into the new model-selection archive because the
campaign uses a newly versioned training implementation.
'''),
    cell('code',IDS[1],'''# Read-only M3 provenance audit + development-only M4 policy preflight.
from pathlib import Path
from src.revised_neural_campaign import (
    audit_milestone3_gates, campaign_source_sha256,
    validate_campaign_payload, read_campaign_progress,
    complete_rankings, run_one_cv,
)
from src.revised_neural_cv import prepare_development_fold
from src.forecasting_protocol import make_rolling_origin_masks

m4_root = Path(PROJECT_ROOT)
m4_dir = m4_root / 'results/development_only/revised_neural_cv_v1'
m4_m3_dir = m4_root / 'results/development_only/revised_neural_fullsize_gate_v1'
m4_m3_payload = neural_fold_smokes[(15, 'F0', 1)]
m4_audit = audit_milestone3_gates(root=m4_root, payload=m4_m3_payload,
                                   directory=m4_m3_dir)
for name in ('LSTM', 'TCN'):
    assert m4_audit[name] is not None, f'Expected completed {name} gate'
    print(f'Milestone 3 {name}: checkpoint integrity VERIFIED; '
          f"MAE={m4_audit[name]['validation_mae_kw']:.6f} kW")
m4_sha = campaign_source_sha256(m4_root)
progress = read_campaign_progress(m4_dir)
print('M4 archived completed CV fits:', len(progress), 'of 360')
print('NEURAL MILESTONE 4 PREFLIGHT: PASSED — no training performed')
'''),
    cell('code',IDS[2],'''# Explicit one-fit cap. This cell NEVER starts training.
M4_RUN_ONE = False                 # Change to True only when ready.
M4_MODEL = 'LSTM'                  # 'LSTM' or 'TCN'
M4_HORIZON = 15                    # 15, 30, 60
M4_FAMILY = 'F0'                   # F0 through F4
M4_FOLD = 1                       # 1, 2, 3
M4_CONFIG_INDEX = 0               # 0, 1, 2, 3
print('One-fit campaign switch:', 'ENABLED' if M4_RUN_ONE else 'DISABLED')
'''),
    cell('code',IDS[3],'''# ONLY this cell may start one full nested-CV training evaluation.
# Leave M4_RUN_ONE=False during initial notebook execution.
if not M4_RUN_ONE:
    print('NEURAL MILESTONE 4 TRAINING: DISABLED — no fit executed')
else:
    base = base_modelling_tables[M4_HORIZON]
    forbidden = holdout_masks[M4_HORIZON]
    assert int(forbidden.sum()) == 26496
    target_masked = base['target_P_Solar_kW'].mask(forbidden)
    assert target_masked.loc[forbidden].isna().all()
    fold = make_rolling_origin_masks(base['valid_time'])[M4_FOLD - 1]
    m4_payload = prepare_development_fold(
        X=feature_family_tables[M4_HORIZON][M4_FAMILY],
        masked_target=target_masked,
        issue_time=base['issue_time'], valid_time=base['valid_time'],
        forbidden_mask=forbidden, fold=fold,
        horizon_minutes=M4_HORIZON, family=M4_FAMILY,
    )
    validate_campaign_payload(m4_payload)
    m4_record, m4_is_new = run_one_cv(
        payload=m4_payload, root=m4_root, model_name=M4_MODEL,
        config_index=M4_CONFIG_INDEX, source_sha256=m4_sha,
        output_dir=m4_dir,
    )
    print('NEURAL MILESTONE 4 SINGLE FIT: PASSED —',
          'new fit' if m4_is_new else 'matching checkpoint reused')
    print('Best epoch:', m4_record['best_epoch'])
    print('MAE kW:', m4_record['validation_mae_kw'])
    print('RMSE kW:', m4_record['validation_rmse_kw'])
    print('Training seconds:', round(m4_record['inner_selection_seconds']+
                                   m4_record['outer_refit_seconds'], 1))
    print('Archived CV fits:', len(read_campaign_progress(m4_dir)), 'of 360')
'''),
    cell('markdown',IDS[4],'''## Milestone 4 initial completion gate — stop here

The default run should print `NEURAL MILESTONE 4 PREFLIGHT: PASSED` and
`NEURAL MILESTONE 4 TRAINING: DISABLED`. Do not start a full-grid search yet.
The audited M3 feasibility gate records must stay separate from M4 CV score
checkpoints. Candidate ranking is not permitted until **all four candidates
have all three outer-fold scores** for a model/horizon/family block.

For a later explicitly approved trial, edit the Milestone 4 switch cell,
run it, then run **only** the Milestone 4 training cell. Each invocation
trains at most one candidate/fold and resumes exact completed checkpoints.
''')
]


def append_milestone4(nb):
    if nb.get('nbformat') != 4 or len(nb.get('cells', [])) < 29:
        raise ValueError('Missing executed revised Notebook 11 Milestone 3')
    ids=[c.get('id') for c in nb['cells']]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate notebook cell identifiers')
    if set(IDS) & set(ids):
        if all(k in ids for k in IDS):
            positions = [ids.index(k) for k in IDS]
            # A later manual audit cell after the M4 extension is also valid.
            if positions == list(range(positions[0], positions[0] + len(IDS))):
                return False
        raise ValueError('Partial/modified Milestone 4 extension detected')
    if not all(k in ids for k in M3_IDS):
        raise ValueError('Missing frozen Milestone 3 identifiers')
    places = [ids.index(k) for k in M3_IDS]
    # M1/M2 must occupy the original first 22 cells; Milestone 3 may contain
    # additional user-created checkpoint/audit cells between its frozen IDs.
    if places[0] != 22 or places != sorted(places):
        raise ValueError('Notebook Milestone 3 identifiers are not in order')
    meta=nb.get('metadata',{}).get('solete_revised_neural_protocol',{})
    if meta.get('terminal_test_evaluated') is not False:
        raise ValueError('Terminal-holdout flag must be false')
    if meta.get('milestone2_pilot_executed') is not True:
        raise ValueError('Milestone 2 pilot record required')
    if meta.get('phase','').split('-')[0] not in ('milestone3','milestone4'):
        raise ValueError('Unexpected revised-neural notebook phase')
    nb['cells'].extend(NEW)
    meta.update({'milestone4_campaign_enabled':False,
                 'phase':'milestone4-audit-and-one-fit-campaign-disabled',
                 'terminal_test_evaluated':False})
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--path',type=Path,
                        default=Path('notebooks/11_revised_neural_models.ipynb'))
    args=parser.parse_args()
    path=args.path
    contents=path.read_text(encoding='utf-8')
    nb=json.loads(contents)
    if append_milestone4(nb):
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print('MILESTONE 4 APPENDED: PASSED — prior cells and executed outputs preserved')
    else:
        print('MILESTONE 4 ALREADY INSTALLED — unchanged')
    print('Full campaign DISABLED; no training was executed')


if __name__=='__main__':
    main()
