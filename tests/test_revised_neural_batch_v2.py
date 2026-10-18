"""Synthetic mixed V1/V2 archive invariants; no real fitting."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def batch(monkeypatch):
    v1batch=types.ModuleType('src.revised_neural_batch')
    v1=types.ModuleType('src.revised_neural_campaign')
    v2=types.ModuleType('src.revised_neural_campaign_v2')
    ranking=types.ModuleType('src.revised_neural_training')
    keys=tuple([('LSTM',15,'F0',c,f) for c in range(4) for f in (1,2,3)]+[
        ('LSTM',15,'F1',c,f) for c in range(4) for f in (1,2,3)])
    v1batch.campaign_keys=lambda models:keys
    kinds={'v1':'v1-kind','v2':'v2-kind'}
    record_schema={}
    def reader(folder,kind):
        rows=[]
        for path in sorted(Path(folder).glob('*.json')):
            state=json.loads(path.read_text())
            if state['kind']!=kinds[kind]:raise ValueError('Unexpected checkpoint kind')
            r=state['identity']
            rows.append({'model':r['model'],'horizon_minutes':r['horizon_minutes'],
                'feature_family':r['feature_family'],'config_index':r['config_index'],
                'fold':r['outer_fold'],'candidate_id':r['candidate_id'],
                'validation_mae_kw':0.1,'validation_rmse_kw':0.2,'best_epoch':1})
        return pd.DataFrame(rows,columns=['model','horizon_minutes','feature_family',
            'config_index','fold','candidate_id','validation_mae_kw','validation_rmse_kw','best_epoch'])
    def audit_prior(folder,*,source_sha256):
        data=reader(folder,'v1')
        for path in Path(folder).glob('*.json'):
            if json.loads(path.read_text())['identity']['source_sha256']!=source_sha256:
                raise ValueError('V1 source fingerprint')
        return len(data)
    v1batch._audit_prior_source=audit_prior
    v1.campaign_source_sha256=lambda root:'sha-original'
    v2.campaign_source_sha256=lambda root:'sha-corrected'
    v1.read_campaign_progress=lambda folder:reader(folder,'v1')
    v2.read_campaign_progress=lambda folder:reader(folder,'v2')
    v2.SCALING_POLICY='calendar-year-safe-v2'
    v2.KIND=kinds['v2'];v2.SCHEMA_VERSION=2
    def checkpoint_name(model,h,f,c,k):
        return f'{model.lower()}_c{c+1}_h{h}_{f}_fold{k}.json'
    v1.checkpoint_name=checkpoint_name
    v2.checkpoint_name=checkpoint_name
    history=[]
    def run(kind):
        def execute(*,payload,root,model_name,config_index,source_sha256,output_dir):
            family=payload['family'];fold=payload['fold'];horizon=payload['horizon']
            assert (kind=='v1')==(family=='F0')
            expected='sha-original' if kind=='v1' else 'sha-corrected'
            assert source_sha256==expected and payload['masked_terminal']
            p=Path(output_dir)/checkpoint_name(model_name,horizon,family,config_index,fold)
            if p.is_file():
                state=json.loads(p.read_text())
                if state['identity']['source_sha256']!=expected:
                    raise ValueError('Checkpoint fingerprint mismatch')
                return state['record'], False
            ident={'source_sha256':expected,'feature_family':family,
                   'candidate_id':f'{model_name}_C{config_index+1}',
                   'model':model_name,'config_index':config_index,
                   'outer_fold':fold,'horizon_minutes':horizon}
            if kind=='v2':ident['scaling_policy']=v2.SCALING_POLICY
            state={'kind':kinds[kind],'schema_version':2 if kind=='v2' else 1,
                   'identity':ident,'record':{'validation_mae_kw':0.12}}
            p.write_text(json.dumps(state))
            history.append((kind,family,config_index,fold))
            return state['record'], True
        return execute
    v1.run_one_cv=run('v1')
    v2.run_one_cv=run('v2')
    ranking.rank_neural_cv_scores=lambda block: block
    monkeypatch.setitem(sys.modules,'src.revised_neural_batch',v1batch)
    monkeypatch.setitem(sys.modules,'src.revised_neural_campaign',v1)
    monkeypatch.setitem(sys.modules,'src.revised_neural_campaign_v2',v2)
    monkeypatch.setitem(sys.modules,'src.revised_neural_training',ranking)
    spec=importlib.util.spec_from_file_location('batchv2_fixture',ROOT/'src/revised_neural_batch_v2.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod._history=history
    mod._names=checkpoint_name
    return mod


def payload(h,f,k):
    return {'horizon':h,'family':f,'fold':k,'masked_terminal':True}


def populate_v1(batch,path):
    path.mkdir(parents=True,exist_ok=True)
    for c in range(4):
        for f in (1,2,3):
            p=path/batch._names('LSTM',15,'F0',c,f)
            p.write_text(json.dumps({'kind':'v1-kind','schema_version':1,
                'record':{'validation_mae_kw':0.1},
                'identity':{'source_sha256':'sha-original','feature_family':'F0',
                'model':'LSTM','candidate_id':f'LSTM_C{c+1}','config_index':c,
                'outer_fold':f,'horizon_minutes':15}}))


def test_resume_12_original_f0_and_one_new_v2(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2'
    populate_v1(batch,one)
    args={'root':tmp_path,'v1_dir':one,'v2_dir':two,'payload_factory':payload,
          'max_new_fits':1,'models':('LSTM',),'enabled':True,'report':lambda _:None}
    r=batch.run_mixed_batch(**args)
    assert (r['before'],r['after'],r['new_fits'],r['reused_fits'])==(12,13,1,12)
    assert batch._history==[('v2','F1',0,1)]
    assert not (one/'.neural_batch.lock').exists()
    assert len(list(one.glob('*.json')))==12
    assert len(list(two.glob('*.json')))==1
    r=batch.run_mixed_batch(**dict(args,max_new_fits=2))
    assert (r['before'],r['after'],r['new_fits'],r['reused_fits'])==(13,15,2,13)
    assert batch._history==[('v2','F1',0,1),('v2','F1',0,2),('v2','F1',0,3)]


def test_disabled_reads_existing_but_does_not_create(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2';populate_v1(batch,one)
    r=batch.run_mixed_batch(root=tmp_path,v1_dir=one,v2_dir=two,
        payload_factory=lambda *a: (_ for _ in ()).throw(AssertionError('payload used')))
    assert (r['after'],r['new_fits'])==(12,0)
    assert not two.exists()


def test_f0_archive_corruption_refuses_all_new_training(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2';populate_v1(batch,one)
    p=next(one.glob('*.json'))
    s=json.loads(p.read_text());s['identity']['source_sha256']='tampered';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='fingerprint'):
        batch.run_mixed_batch(root=tmp_path,v1_dir=one,v2_dir=two,
            payload_factory=payload,enabled=True)
    assert batch._history==[]


def test_v2_archive_corruption_refuses_new_training(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2';populate_v1(batch,one);two.mkdir()
    p=two/batch._names('LSTM',15,'F1',0,1)
    p.write_text(json.dumps({'kind':'v2-kind','schema_version':2,
        'identity':{'source_sha256':'tampered','scaling_policy':'calendar-year-safe-v2',
            'model':'LSTM','horizon_minutes':15,'feature_family':'F1','candidate_id':'LSTM_C1',
            'config_index':0,'outer_fold':1},'record':{'validation_mae_kw':0.1}}))
    with pytest.raises(ValueError,match='fingerprint'):
        batch.run_mixed_batch(root=tmp_path,v1_dir=one,v2_dir=two,
            payload_factory=payload,enabled=True)
    assert batch._history==[]


def test_combined_records_reject_v1_non_f0(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2';populate_v1(batch,one)
    p=next(one.glob('*.json'))
    s=json.loads(p.read_text());s['identity']['feature_family']='F1';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='unexpectedly contains F1'):
        batch.read_mixed_progress(one,two)


def test_locked_batch_refuses_parallel_run(batch,tmp_path):
    one=tmp_path/'v1';two=tmp_path/'v2';populate_v1(batch,one)
    (one/'.neural_batch.lock').touch()
    with pytest.raises(RuntimeError,match='already running'):
        batch.run_mixed_batch(root=tmp_path,v1_dir=one,v2_dir=two,
            payload_factory=payload,enabled=True)
    assert not batch._history


@pytest.mark.parametrize('cap',[0,361,True,1.5])
def test_invalid_caps_fail_closed(batch,tmp_path,cap):
    with pytest.raises(ValueError,match='Invalid enabled'):
        batch.run_mixed_batch(root=tmp_path,v1_dir=tmp_path/'a',v2_dir=tmp_path/'b',
            payload_factory=payload,max_new_fits=cap)
