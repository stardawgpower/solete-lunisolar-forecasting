"""Append versioned zero-variance-year scaling trial; leave all prior cells untouched.

From repository root: python scripts/append_neural_m6.py
"""
from __future__ import annotations
import ast
import json
from pathlib import Path

INTRO_ID = 'm6-year-scaling-v2-intro'
CELL_ID = 'm6-year-scaling-v2-batch'
OUTRO_ID = 'm6-year-scaling-v2-outro'
M5_ID = 'm5-neural-batch-safe-run'

INTRO = '''## Milestone 6 — F1+ calendar-year variance correction, versioned

The canonical **valid_utc_year** predictor is constant on the 2018 train rows
of the first F1 fold. The frozen scaler correctly raised an exception instead
of dividing by zero. The revised V2 scaler centers that specific feature on
TRAIN and uses std=1 only if TRAIN variance is zero. All other zero-variance
continuous predictors remain fatal; if the year varies, frozen V1 scaling
is delegated to unchanged. No future/outer-validation targets fit the scaler.

**Provenance:** This leaves `src/neural_models.py`, all original Milestone 3
checkpoints, the V1 neural campaign implementation, and the 12 complete F0
checkpoints unchanged. F0 uses original V1 source fingerprints; F1 through F4
use new V2 training/scaling code with a separate versioned checkpoint archive.
The combined scheduler checks both archive fingerprints and reports progress
out of the same 360 frozen evaluations, without overwriting V1 results.

Run All remains safe because `M6_RUN_BATCH=False`. Start with one new V2
F1 evaluation, verify its score and source/data identity, then consider the
remaining 11 F1 fits. Never use the terminal holdout or silently promote
Milestone 3 feasibility scores. Do not run simultaneous batch schedulers.
'''

CELL = '''# Milestone 6: bounded, explicit opt-in for versioned V2 year-safe F1+ scaling.
# Safe Run All: this switch MUST remain False in committed notebook.
M6_RUN_BATCH = False
M6_MAX_NEW_FITS = 1
M6_MODELS = ('LSTM',)

from pathlib import Path
from src.revised_neural_campaign import read_campaign_progress, campaign_source_sha256
from src.revised_neural_batch_v2 import run_mixed_batch, read_mixed_progress, complete_mixed_rankings

assert globals().get('m4_preflight_passed', False), 'Run canonical preflight first'
assert M3_RUN_LSTM is False and M3_RUN_TCN is False
assert M4_RUN_ONE is False and M5_RUN_BATCH is False
assert callable(m5_payload_for), 'Execute the disabled M5 setup cell first'

m6_root = Path(PROJECT_ROOT)
m6_v1 = m6_root / 'results/development_only/revised_neural_cv_v1'
m6_v2 = m6_root / 'results/development_only/revised_neural_cv_v2_year_std1'
assert len(read_campaign_progress(m6_v1)) >= 12, 'Missing committed V1 F0 checkpoints'

m6_report = run_mixed_batch(
    root=m6_root, v1_dir=m6_v1, v2_dir=m6_v2,
    payload_factory=m5_payload_for,
    max_new_fits=M6_MAX_NEW_FITS, models=M6_MODELS,
    enabled=M6_RUN_BATCH,
)
print('MILESTONE 6 combined archived:', m6_report['after'], '/360')
if not M6_RUN_BATCH:
    assert m6_report['new_fits'] == 0
    print('MILESTONE 6 BATCH: DISABLED — no model training')
'''

OUTRO = '''### Milestone 6 verification boundary

A successful first V2 candidate/fold checkpoint is NOT a finished feature-family
comparison. Rank F1 only after 4 candidates × 3 folds. Inspect the results and
tests before running more. Save the notebook with `M6_RUN_BATCH=False`.

Final model/horizon/family comparisons must disclose the frozen V1/F0 and
versioned V2/F1–F4 train-only scaling policies, including why the V2 correction
was required. The terminal June–August 2019 holdout remains excluded.
'''


def _cell(cell_type: str, cell_id: str, source: str) -> dict:
    cell = {'cell_type':cell_type,'id':cell_id,'metadata':{},
            'source':source.splitlines(keepends=True)}
    if cell_type == 'code':
        ast.parse(source)
        cell.update({'execution_count':None,'outputs':[]})
    return cell


def append_m6(nb: dict) -> bool:
    cells = nb.get('cells', [])
    ids = [c.get('id') for c in cells]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate notebook cell identifiers')
    required = ('m4-campaign-preflight','m4-run-all-summary', M5_ID)
    if any(x not in ids for x in required):
        raise ValueError('Milestone 6 requires completed M4/M5 notebook cells')
    if 'M5_RUN_BATCH = False' not in ''.join(cells[ids.index(M5_ID)]['source']):
        raise ValueError('Save Milestone 5 training switch DISABLED before installing Milestone 6')
    tags = (INTRO_ID, CELL_ID, OUTRO_ID)
    already = [x for x in tags if x in ids]
    if already:
        if len(already)==3 and ids[-3:]==list(tags):
            return False
        raise ValueError('Conflicting or partially installed Milestone 6 cells')
    meta = nb.setdefault('metadata',{}).setdefault('solete_revised_neural_protocol',{})
    if meta.get('terminal_test_evaluated') is not False:
        raise ValueError('Locked terminal holdout provenance missing')
    cells.extend((_cell('markdown',INTRO_ID,INTRO),
                  _cell('code',CELL_ID,CELL),
                  _cell('markdown',OUTRO_ID,OUTRO)))
    meta['phase']='milestone6-v2-constant-year-scaler-opt-in-installed'
    meta['milestone6_batch_default_enabled']=False
    return True


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    path=root/'notebooks/11_revised_neural_models.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    changed=append_m6(nb)
    if changed:
        path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
        print('NEURAL MILESTONE 6 APPENDED: PASSED — existing notebook cells and outputs retained')
    else:
        print('NEURAL MILESTONE 6 ALREADY INSTALLED: PASSED')
    print('M6_RUN_BATCH=False: safe Run All; old F0 checkpoints unchanged')


if __name__=='__main__':
    main()
