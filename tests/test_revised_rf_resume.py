"""Static guards and synthetic resume tests (never fit actual RF models)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.revised_rf_checkpoint import run_block

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / '10_revised_classical_models.ipynb'
FROZEN_LGB = '71aa4140023fa35666964ac2f027b1a2d0722f87882bee5ee09cd48054920ba4'


def test_revised_notebook_is_additive_and_terminal_untouched():
    nb = json.loads(NB.read_text(encoding='utf-8'))
    lgb = next(c for c in nb['cells'] if c.get('id') == 'm4-lightgbm-full-cv')
    assert hashlib.sha256(''.join(lgb['source']).encode()).hexdigest() == FROZEN_LGB
    assert any('Complete LightGBM rolling-CV selection: PASSED' in ''.join(o.get('text', []))
               for o in lgb['outputs'] if o.get('output_type') == 'stream')
    for cell_id in ('m5-lgb-audit', 'm5-rf-resume'):
        cells = [c for c in nb['cells'] if c.get('id') == cell_id]
        assert len(cells) == 1
        assert cells[0]['cell_type'] == 'code'
        ast.parse(''.join(cells[0]['source']))
    rf_source = ''.join(next(c for c in nb['cells'] if c.get('id') == 'm5-rf-resume')['source'])
    for required in ('RF_MAX_NEW_FITS = 12', 'run_block(',
                     'rf_y = rf_base["target_P_Solar_kW"].mask(rf_holdout)',
                     'EXPECTED_CV_COUNTS_BY_HORIZON[rf_horizon]',
                     'evaluate_cv_candidates', 'Complete Random Forest rolling-CV selection: PASSED'):
        assert required in rf_source
    for forbidden in ('make_terminal_refit_mask(', '.predict(', 'terminal_evaluation',
                      'build_lightgbm_model', 'refit_mask'):
        assert forbidden not in rf_source
    assert nb['metadata']['solete_revised_protocol']['terminal_test_evaluated'] is False


def test_atomic_resume_identity_and_fold_accounting(tmp_path):
    # 64 development timestamps, then exactly 26,496 unavailable holdout labels.
    n = 64 + 26496
    time = pd.date_range('2018-06-01', periods=n, freq='5min', tz='UTC')
    index = pd.RangeIndex(n)
    valid = pd.Series(time, index=index)
    issue = valid - pd.Timedelta(minutes=15)
    mask = pd.Series(np.arange(n)>=64,index=index)
    X = pd.DataFrame({'x': np.arange(n, dtype=float)},index=index)
    y = pd.Series(np.arange(n,dtype=float),index=index).mask(mask)
    folds = []
    counts = []
    for f, (train_stop, val_start) in enumerate(((5,12),(19,27),(33,42)),1):
        train = pd.Series(np.arange(n)<train_stop,index=index)
        validation = pd.Series(np.isin(np.arange(n),[val_start,val_start+1]),index=index)
        folds.append({'fold':f,'train_mask':train,'validation_mask':validation})
        counts.append((train_stop,2))
    configs = tuple({'max_features':.5, 'min_samples_leaf':1, 'max_depth':i}
                    for i in range(8))
    seen = []
    def fake_evaluator(*, folds, configurations, forbidden_mask, y, **kwargs):
        assert y.loc[forbidden_mask].isna().all()
        assert len(folds)==len(configurations)==1
        fold=folds[0]
        conf=configurations[0]
        seen.append((conf['max_depth'],fold['fold']))
        rows={'config_index':0,'fold':fold['fold'],
              'train_rows':int(fold['train_mask'].sum()),
              'validation_rows':int(fold['validation_mask'].sum()),
              'validation_mae':float(conf['max_depth']+.01*fold['fold']),
              'validation_rmse':float(conf['max_depth']+.5)}
        return pd.DataFrame([rows]),pd.DataFrame([{'rank':1}])
    args=dict(X=X,y=y, issue_time=issue,valid_time=valid,forbidden_mask=mask,
              folds=folds,configurations=configs,model_factory=object(),
              evaluator=fake_evaluator,horizon_minutes=15,feature_family='F0',
              expected_counts=counts, checkpoint_path=tmp_path/'h15_F0.json',
              source_sha256='0'*64)
    rows, ranking, new = run_block(**args,max_new_fits=2)
    assert (len(rows),ranking,new)==(2,None,2)
    rows, ranking, new = run_block(**args,max_new_fits=3)
    assert (len(rows),ranking,new)==(5,None,3)
    rows, ranking, new = run_block(**args,max_new_fits=24)
    assert (len(rows),len(ranking),new)==(24,8,19)
    assert (ranking.iloc[0]['config_index'],ranking.iloc[0]['rank'])==(0,1)
    assert len(seen)==24
    rows,ranking,new = run_block(**args,max_new_fits=2)
    assert new==0 and len(seen)==24
    with pytest.raises(ValueError,match='identity mismatch'):
        run_block(**dict(args,configurations=tuple(reversed(configs))),max_new_fits=2)
    changed=X.copy();changed.iloc[0,0]=123456
    with pytest.raises(ValueError,match='identity mismatch'):
        run_block(**dict(args,X=changed),max_new_fits=2)
    saved=json.loads((tmp_path/'h15_F0.json').read_text())
    saved['fits'][0]['validation_mae']=float('nan')
    (tmp_path/'h15_F0.json').write_text(json.dumps(saved))
    with pytest.raises(AssertionError):
        run_block(**args,max_new_fits=2)
