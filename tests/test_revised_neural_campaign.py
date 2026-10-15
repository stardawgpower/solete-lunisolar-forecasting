"""Milestone 4 guards: no real training and no terminal dataset access."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.revised_neural_campaign as m4


def toy_payload():
    ntr, nval=180, 120
    issue_train=pd.date_range('2018-11-30 07:00',periods=ntr,freq='5min',tz='UTC')-pd.Timedelta(minutes=15)
    issue_val=pd.date_range('2018-12-01 00:00',periods=nval,freq='5min',tz='UTC')-pd.Timedelta(minutes=15)
    issue=issue_train.append(issue_val)
    valid=issue+pd.Timedelta(minutes=15)
    n=len(issue)
    return {'sequence':np.ones((n,13,9),dtype=np.float32),
            'auxiliary':np.ones((n,1),dtype=np.float32),
            'target_kw':np.arange(n,dtype=np.float32),
            'train_mask':np.arange(n)<ntr,
            'validation_mask':np.arange(n)>=ntr,
            'issue_time':pd.Series(issue,index=pd.RangeIndex(n)),
            'valid_time':pd.Series(valid,index=pd.RangeIndex(n)),
            'horizon_minutes':15,'family':'F0','fold':1,
            'train_rows':ntr,'validation_rows':nval,
            'continuous_positions':np.array([0],dtype=np.int64),
            'binary_positions':np.array([],dtype=np.int64)}


@pytest.fixture
def toy(monkeypatch):
    original=m4.EXPECTED_CV_COUNTS_BY_HORIZON
    copied=dict(original)
    copied[15]=((180,120),)*3
    monkeypatch.setattr(m4,'EXPECTED_CV_COUNTS_BY_HORIZON',copied)
    return toy_payload()


def test_frozen_grid_is_complete():
    assert [c['candidate_id'] for c in m4.grid('LSTM')]==[f'LSTM_C{i}' for i in range(1,5)]
    assert [c['candidate_id'] for c in m4.grid('TCN')]==[f'TCN_C{i}' for i in range(1,5)]
    with pytest.raises(ValueError):m4.grid('GRU')
    assert len({m4.checkpoint_name(m,h,f,c,i) for m in m4.MODELS for h in m4.HORIZONS for f in m4.FAMILIES for c in range(4) for i in (1,2,3)})==360


def test_legal_fold_split_passes(toy):
    train,val,inner=m4.validate_campaign_payload(toy)
    assert (len(train),len(val),inner['embargo_rows'])==(180,120,12)
    assert not (inner['inner_train_mask'] & toy['validation_mask']).any()
    assert not (inner['inner_validation_mask'] & toy['validation_mask']).any()


def test_illegal_horizon_feature_family_and_fold(toy):
    for key,value in [('horizon_minutes',45),('family','F5'),('fold',4)]:
        bad=dict(toy,**{key:value})
        with pytest.raises(ValueError):m4.validate_campaign_payload(bad)


def test_bad_counts_and_gap_fail(toy):
    bad=dict(toy,train_rows=170)
    with pytest.raises(ValueError,match='metadata'):m4.validate_campaign_payload(bad)
    bad=dict(toy,train_mask=toy['train_mask'].copy())
    bad['train_mask'][-1]=True
    with pytest.raises(ValueError,match='overlap'):m4.validate_campaign_payload(bad)
    bad=dict(toy,issue_time=toy['issue_time'].copy())
    bad['issue_time'].iloc[180]=bad['valid_time'].iloc[179]
    with pytest.raises(ValueError):m4.validate_campaign_payload(bad)


def test_nonfinite_data_invalid(toy):
    bad=dict(toy,target_kw=toy['target_kw'].copy())
    bad['target_kw'][-1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):m4.validate_campaign_payload(bad)


def test_bad_auxiliary_contract_rejected(toy):
    bad=dict(toy,auxiliary=toy['auxiliary'][:,:0])
    with pytest.raises(ValueError,match='dimensions'):m4.validate_campaign_payload(bad)
    bad=dict(toy,binary_positions=np.array([0]),continuous_positions=np.array([],dtype=np.int64))
    with pytest.raises(ValueError,match='binary'):m4.validate_campaign_payload(bad)


def _fake_modules(monkeypatch):
    tf=types.ModuleType('tensorflow')
    tf.__version__='synthetic-tf'
    monkeypatch.setitem(sys.modules,'tensorflow',tf)
    archive=types.ModuleType('src.revised_rf_checkpoint')
    def atomic_text(p, content, *,refuse_different=False):
        p=Path(p)
        if p.exists() and refuse_different and p.read_text()!=content:
            raise ValueError('refuse overwrite')
        p.write_text(content,encoding='utf-8')
    archive.atomic_text=atomic_text
    monkeypatch.setitem(sys.modules,'src.revised_rf_checkpoint',archive)


def record():
    return {'inner_training_rows':150,'inner_validation_rows':18,'inner_embargo_rows':12,
        'outer_training_rows':180,'outer_validation_rows':120,'inner_epochs_completed':16,
        'best_epoch':8,'best_inner_validation_loss_scaled':.14,'trainable_parameters':6497,
        'inner_selection_seconds':40.,'outer_refit_seconds':20.,'outer_prediction_seconds':.2,
        'validation_mae_kw':.13,'validation_rmse_kw':.36}


def test_one_fit_atomic_resume_and_data_mismatch(tmp_path,toy,monkeypatch):
    _fake_modules(monkeypatch)
    monkeypatch.setattr(m4,'campaign_source_sha256',lambda _: 'frozen')
    calls=[]
    def train_one(**kwargs):
        calls.append(1)
        assert len(kwargs['train_idx'])==180
        assert len(kwargs['val_idx'])==120
        return record()
    monkeypatch.setattr(m4,'_train_one',train_one)
    arg=dict(payload=toy,root=tmp_path,model_name='LSTM',config_index=0,
             source_sha256='frozen',output_dir=tmp_path/'checkpoints')
    rec,new=m4.run_one_cv(**arg)
    assert new and rec==record() and calls==[1]
    saved=arg['output_dir']/m4.checkpoint_name('LSTM',15,'F0',0,1)
    assert saved.exists()
    state=json.loads(saved.read_text())
    assert state['kind']==m4.KIND
    assert len(m4.read_campaign_progress(arg['output_dir']))==1
    assert m4.complete_rankings(arg['output_dir']).empty
    rec,new=m4.run_one_cv(**arg)
    assert not new and calls==[1]
    tampered=dict(toy,target_kw=toy['target_kw'].copy())
    tampered['target_kw'][-1]+=1
    with pytest.raises(ValueError,match='identity mismatch'):
        m4.run_one_cv(**dict(arg,payload=tampered))
    with pytest.raises(ValueError,match='fingerprint'):
        m4.run_one_cv(**dict(arg,source_sha256='wrong'))
    assert not saved.with_suffix('.lock').exists()


def test_stale_lock_prevents_run(tmp_path,toy,monkeypatch):
    _fake_modules(monkeypatch)
    monkeypatch.setattr(m4,'campaign_source_sha256',lambda _: 'frozen')
    path=tmp_path/m4.checkpoint_name('TCN',15,'F0',0,1)
    path.with_suffix('.lock').touch()
    with pytest.raises(RuntimeError,match='locked'):
        m4.run_one_cv(payload=toy,root=tmp_path,model_name='TCN',config_index=0,
                      source_sha256='frozen',output_dir=tmp_path)


def test_broken_checkpoint_rejected(tmp_path,toy,monkeypatch):
    _fake_modules(monkeypatch)
    monkeypatch.setattr(m4,'campaign_source_sha256',lambda _: 'frozen')
    monkeypatch.setattr(m4,'_train_one',lambda **kwargs: record())
    arg=dict(payload=toy,root=tmp_path,model_name='TCN',config_index=1,
             source_sha256='frozen',output_dir=tmp_path)
    m4.run_one_cv(**arg)
    path=tmp_path/m4.checkpoint_name('TCN',15,'F0',1,1)
    state=json.loads(path.read_text())
    state['record']['best_epoch']=81
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='epoch'):
        m4.run_one_cv(**arg)


def test_ranking_waits_for_all_candidate_folds(tmp_path):
    assert m4.complete_rankings(tmp_path).empty
    for candidate in range(4):
        for fold in range(1,4):
            identity={'model':'LSTM','horizon_minutes':15,'feature_family':'F0',
                      'config_index':candidate,'outer_fold':fold,
                      'configuration':m4.grid('LSTM')[candidate],
                      'outer_train_rows':180,'outer_validation_rows':120}
            obj=dict(schema_version=1,kind=m4.KIND,identity=identity,record=record())
            path=tmp_path/m4.checkpoint_name('LSTM',15,'F0',candidate,fold)
            path.write_text(json.dumps(obj))
            if candidate<3 or fold<3:
                assert m4.complete_rankings(tmp_path).empty
    ranked=m4.complete_rankings(tmp_path)
    assert len(ranked)==4
    assert ranked['config_index'].tolist()==[0,1,2,3]


def test_append_keeps_executed_content_and_idempotent():
    from scripts.append_neural_m4 import append_milestone4, IDS, M3_IDS
    cells=[dict(cell_type='code',id=f'cell-{i}',source=['x\n'],
                execution_count=1,outputs=[dict(output_type='stream',text=['output\n'])])
           for i in range(22)]
    cells.extend(dict(id=key,cell_type='markdown',source=['safe\n']) for key in M3_IDS)
    cells.append(dict(id='extra-audit',cell_type='code',source=['y\n'],
                      execution_count=21,outputs=[]))
    nb=dict(nbformat=4,cells=cells,metadata={
        'solete_revised_neural_protocol':dict(terminal_test_evaluated=False,
           milestone2_pilot_executed=True,phase='milestone3-fullsize-single-fold-gates-disabled-pending-execution')})
    original=json.dumps(nb['cells'])
    assert append_milestone4(nb)
    assert json.dumps(nb['cells'][:-len(IDS)])==original
    assert not append_milestone4(nb)
    assert [c['id'] for c in nb['cells'][-len(IDS):]]==list(IDS)
    assert nb['metadata']['solete_revised_neural_protocol']['terminal_test_evaluated'] is False
    assert 'M4_RUN_ONE = False' in ''.join(nb['cells'][-3]['source'])


def test_training_restarts_fresh_model_and_never_scales_outer_labels(toy,monkeypatch):
    # Verify actual training flow with fake Keras; no TensorFlow installation.
    inner=m4.make_inner_split(toy)
    toy['target_kw'][toy['validation_mask']]=1e8  # hostile outer labels
    events=[]
    class Model:
        def __init__(self):
            events.append(('new-model',len(events)))
        def count_params(self):return 6497
        def fit(self,inputs,y,**kwargs):
            assert np.max(y) < 1e8
            is_inner='validation_data' in kwargs
            events.append(('fit',len(y),is_inner))
            if is_inner:
                assert kwargs['epochs']==80
                assert len(kwargs['validation_data'][1])==18
                return types.SimpleNamespace(history={'val_loss':[.5,.3,.4],'loss':[.8,.6,.7]})
            assert len(y)==180 and kwargs['epochs']==2
            return types.SimpleNamespace(history={'loss':[.7,.6]})
        def predict(self,x,**kwargs):
            events.append(('predict',len(x[0])))
            return np.ones((len(x[0]),1),dtype=np.float32)
    def fake_scaler(seq,aux,target,train_mask,**kwargs):
        assert len(target) == len(train_mask)
        assert np.max(target)<1e8
        events.append(('scaler',len(target),int(train_mask.sum())))
        return {'one':'yes'}
    nn=types.ModuleType('src.neural_models')
    nn.fit_train_only_scalers=fake_scaler
    nn.apply_train_only_scalers=lambda seq,aux,target,stats:{'sequence':seq,'auxiliary':aux,'target_scaled':target}
    nn.inverse_target_scaling=lambda prediction,stats:prediction
    monkeypatch.setitem(sys.modules,'src.neural_models',nn)
    monkeypatch.setattr(m4,'_make_model',lambda *args:Model())
    tf=types.ModuleType('tensorflow')
    tf.keras=types.SimpleNamespace(
        backend=types.SimpleNamespace(clear_session=lambda:None),
        callbacks=types.SimpleNamespace(EarlyStopping=lambda **kwargs:kwargs))
    monkeypatch.setitem(sys.modules,'tensorflow',tf)
    rec=m4._train_one(payload=toy,model_name='LSTM',configuration=m4.grid('LSTM')[0],
                      train_idx=np.arange(180),val_idx=np.arange(180,300),inner=inner)
    assert rec['best_epoch']==2 and rec['inner_epochs_completed']==3
    assert rec['validation_mae_kw']>1000
    assert len([e for e in events if e[0]=='new-model'])==2
    assert [e[1:] for e in events if e[0]=='fit']==[(150,True),(180,False)]
    assert [e[1:] for e in events if e[0]=='scaler']==[(168,150),(180,180)]
    assert [e[1:] for e in events if e[0]=='predict']==[(120,)]


def test_installer_accepts_interleaved_m3_checkpoint_audit():
    """A manually added TCN audit between M3 TCN and M3 gate must survive."""
    from scripts.append_neural_m4 import append_milestone4, IDS, M3_IDS

    cells = [dict(cell_type='code', id=f'base-{i}', source=['pass\n'],
                  execution_count=i + 1, outputs=[])
             for i in range(22)]
    for key in M3_IDS[:-1]:
        cells.append(dict(cell_type='markdown', id=key, source=['frozen\n']))
    cells.append(dict(cell_type='code', id='audit-tcn-checkpoint',
                      source=['print("TCN CHECKPOINT: VERIFIED")\n'],
                      execution_count=19,
                      outputs=[dict(output_type='stream', name='stdout',
                                    text=['TCN CHECKPOINT: VERIFIED\n'])]))
    cells.append(dict(cell_type='markdown', id=M3_IDS[-1], source=['gate\n']))
    cells.append(dict(cell_type='code', id='later-audit',
                      source=['print("LSTM CHECKPOINT: VERIFIED")\n'],
                      execution_count=20, outputs=[]))
    nb = dict(nbformat=4, cells=cells, metadata={
        'solete_revised_neural_protocol': dict(
            terminal_test_evaluated=False, milestone2_pilot_executed=True,
            phase='milestone3-fullsize-single-fold-gates-disabled-pending-execution')})
    original_cells = json.dumps(cells, sort_keys=True)
    assert append_milestone4(nb) is True
    assert json.dumps(nb['cells'][:-len(IDS)], sort_keys=True) == original_cells
    assert [c['id'] for c in nb['cells'][-len(IDS):]] == list(IDS)
    assert append_milestone4(nb) is False
    assert nb['metadata']['solete_revised_neural_protocol']['milestone4_campaign_enabled'] is False

    # A later audit after M4 should not cause a duplicate installation.
    nb['cells'].append(dict(id='post-m4-audit', cell_type='markdown', source=['note\n']))
    assert append_milestone4(nb) is False
