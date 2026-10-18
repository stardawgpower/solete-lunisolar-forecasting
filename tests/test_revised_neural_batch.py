"""Synthetic scheduler and notebook-installer guards; never train TensorFlow."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture
def m5(monkeypatch):
    campaign = types.ModuleType('src.revised_neural_campaign')
    campaign.FAMILIES = ('F0','F1','F2','F3','F4')
    campaign.HORIZONS = (15,30,60)
    campaign.MODELS = ('LSTM','TCN')
    campaign.KIND = 'complete-candidate-single-fold-development-cv'
    campaign.SCHEMA_VERSION = 1
    campaign.grid = lambda model: ({'candidate_id':model+'_C1'},)*4
    campaign.checkpoint_name = lambda m,h,f,c,k: f'{m.lower()}_c{c+1}_h{h}_{f}_fold{k}.json'
    campaign.campaign_source_sha256 = lambda root: 'source-sha'
    campaign.read_campaign_progress = lambda path: list(Path(path).glob('*.json'))
    fitted=[]
    def run_one_cv(*, payload, root, model_name, config_index, source_sha256, output_dir):
        assert source_sha256 == 'source-sha'
        assert payload['holdout_hidden'] is True
        path=Path(output_dir)/campaign.checkpoint_name(model_name,payload['h'],payload['f'],config_index,payload['fold'])
        if path.exists():
            state=json.loads(path.read_text())
            assert state['identity']['source_sha256']==source_sha256
            return state['record'], False
        rec={'validation_mae_kw':0.1}
        state={'schema_version':1,'kind':campaign.KIND,
               'identity':{'source_sha256':source_sha256}, 'record':rec}
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(state))
        fitted.append((model_name,payload['h'],payload['f'],config_index,payload['fold']))
        return rec, True
    campaign.run_one_cv = run_one_cv
    monkeypatch.setitem(sys.modules,'src.revised_neural_campaign',campaign)
    spec=importlib.util.spec_from_file_location('m5_test_module',PKG/'src/revised_neural_batch.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._fitted=fitted
    return module


def payload(h,f,fold):
    return {'h':h, 'f':f, 'fold':fold, 'holdout_hidden':True}


def test_canonical_grid_is_complete_and_in_order(m5):
    keys=m5.campaign_keys()
    assert len(keys)==360
    assert len(set(keys))==360
    assert keys[0]==('LSTM',15,'F0',0,1)
    assert keys[1]==('LSTM',15,'F0',0,2)
    assert keys[179]==('LSTM',60,'F4',3,3)
    assert keys[180]==('TCN',15,'F0',0,1)
    assert len(m5.campaign_keys(('TCN',)))==180


@pytest.mark.parametrize('models',[(),('X',),('LSTM','LSTM'),('TCN','X')])
def test_reject_invalid_models(m5,models):
    with pytest.raises(ValueError,match='models'):
        m5.campaign_keys(models)


@pytest.mark.parametrize('cap',[0,361,1.0,True,'1'])
def test_fail_closed_on_invalid_fit_limits(m5,tmp_path,cap):
    with pytest.raises(ValueError,match='max_new_fits'):
        m5.run_checkpointed_batch(root=tmp_path,output_dir=tmp_path/'cv',payload_factory=payload,max_new_fits=cap)


def test_default_disabled_does_not_call_payload_or_fit(m5,tmp_path):
    report=m5.run_checkpointed_batch(root=tmp_path,output_dir=tmp_path/'cv',
        payload_factory=lambda *args: (_ for _ in ()).throw(AssertionError('payload called')))
    assert report['enabled'] is False and report['new_fits']==0
    assert m5._fitted==[]
    assert not (tmp_path/'cv').exists()


def test_one_fit_only_and_reuse_exact_checkpoint(m5,tmp_path):
    location=tmp_path/'cv'
    args=dict(root=tmp_path,output_dir=location,payload_factory=payload,
              models=('LSTM',),enabled=True,max_new_fits=1,report=lambda *_:None)
    first=m5.run_checkpointed_batch(**args)
    assert first['new_fits']==1 and first['after']==1
    assert m5._fitted==[('LSTM',15,'F0',0,1)]
    second=m5.run_checkpointed_batch(**args)
    assert second['new_fits']==1 and second['reused_fits']==1 and second['after']==2
    assert m5._fitted[-1]==('LSTM',15,'F0',0,2)
    assert not (location/'.neural_batch.lock').exists()


def test_cap_two_allows_exactly_two_new(m5,tmp_path):
    report=m5.run_checkpointed_batch(root=tmp_path,output_dir=tmp_path/'cv',
        payload_factory=payload,models=('TCN',),enabled=True,max_new_fits=2,
        report=lambda *_:None)
    assert report['new_fits']==2 and report['after']==2
    assert m5._fitted==[('TCN',15,'F0',0,1),('TCN',15,'F0',0,2)]


def test_stale_source_is_rejected_without_new_fit(m5,tmp_path):
    folder=tmp_path/'cv';folder.mkdir()
    f=folder/'lstm_c1_h15_F0_fold1.json'
    f.write_text(json.dumps({'schema_version':1,'kind':m5.KIND,'identity':{'source_sha256':'stale'},'record':{}}))
    with pytest.raises(ValueError,match='fingerprint'):
        m5.run_checkpointed_batch(root=tmp_path,output_dir=folder,payload_factory=payload,
                                  enabled=True,report=lambda *_:None)
    assert not m5._fitted


def test_lock_refuses_second_batch(m5,tmp_path):
    folder=tmp_path/'cv';folder.mkdir()
    (folder/'.neural_batch.lock').touch()
    with pytest.raises(RuntimeError,match='already running'):
        m5.run_checkpointed_batch(root=tmp_path,output_dir=folder,payload_factory=payload,
                                  enabled=True,report=lambda *_:None)
    assert not m5._fitted


def test_failure_releases_lock(m5,tmp_path):
    folder=tmp_path/'cv'
    def failure(*args):
        raise RuntimeError('Synthetic failure before training')
    with pytest.raises(RuntimeError,match='Synthetic failure'):
        m5.run_checkpointed_batch(root=tmp_path,output_dir=folder,payload_factory=failure,
                                  enabled=True,report=lambda *_:None)
    assert not (folder/'.neural_batch.lock').exists()
    assert not m5._fitted


def _installer():
    spec=importlib.util.spec_from_file_location('m5_installer',PKG/'scripts/append_neural_m5.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_notebook_append_preserves_prior_outputs_and_idempotent():
    installer=_installer()
    before=[{'id':x,'cell_type':'code','source':['print(1)\n'],
             'outputs':[{'output_type':'stream','name':'stdout','text':['SAFE\n']}],
             'execution_count':1, 'metadata':{}}
            for x in installer.FROZEN_REQUIRED_IDS]
    old=json.loads(json.dumps(before))
    nb={'cells':before, 'metadata':{'solete_revised_neural_protocol':{'terminal_test_evaluated':False}}}
    assert installer.append_m5(nb)
    assert nb['cells'][:len(old)]==old
    assert len(nb['cells'])==len(old)+3
    assert not installer.append_m5(nb)
    assert nb['cells'][:len(old)]==old
    assert 'M5_RUN_BATCH = False' in ''.join(nb['cells'][-2]['source'])


def test_notebook_requires_preflight_and_locked_terminal(m5):
    installer=_installer()
    with pytest.raises(ValueError,match='Missing required'):
        installer.append_m5({'cells':[], 'metadata':{'solete_revised_neural_protocol':{'terminal_test_evaluated':False}}})
    nb={'cells':[{'id':i,'cell_type':'code','source':['pass\n']} for i in installer.FROZEN_REQUIRED_IDS],
        'metadata':{'solete_revised_neural_protocol':{'terminal_test_evaluated':True}}}
    with pytest.raises(ValueError,match='terminal-test'):
        installer.append_m5(nb)
