"""Milestone 6: synthetic safety tests; no TensorFlow training or SOLETE files."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name,PKG/path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scaler(monkeypatch):
    module=types.ModuleType('src.neural_models')
    module.SCALING_DDOF=0
    calls=[]
    def arrays(s,a,t):
        s,a,t=map(np.asarray,(s,a,t))
        if s.ndim!=3 or a.ndim!=2 or t.shape!=(len(s),) or len(a)!=len(s):
            raise ValueError('Bad raw input dimensions')
        if any(not np.isfinite(x).all() for x in (s,a,t)):
            raise ValueError('Nonfinite neural array')
        return s,a,t
    def positions(n,c,b):
        c,b=map(np.asarray,(c,b))
        if c.ndim!=1 or b.ndim!=1 or set(c)|set(b)!=set(range(n)) or set(c)&set(b):
            raise ValueError('Bad auxiliary feature positions')
        return c,b
    def old(s,a,t,m,*,continuous_positions,binary_positions):
        calls.append(True)
        if (np.asarray(a)[np.asarray(m)][:,continuous_positions].std(axis=0) <= 0).any():
            raise ValueError('Zero-variance')
        return {'frozen_scaler_used':True}
    module._validate_raw_neural_arrays=arrays
    module._validate_positions=positions
    module.fit_train_only_scalers=old
    monkeypatch.setitem(sys.modules,'src.neural_models',module)
    mod=load('src/revised_neural_scaling_v2.py','test_m6_scaler')
    mod._legacy_calls=calls
    return mod


def toy(n=120,year_val=2018.0):
    rng=np.random.default_rng(32)
    seq=rng.normal(size=(n,13,9)).astype('float32')
    aux=rng.normal(size=(n,16)).astype('float32')
    aux[:,1]=year_val
    aux[:,15]=np.arange(n)%2
    aux[80:,1]=2019.0
    target=np.linspace(0,2,n,dtype='float32')
    mask=np.arange(n)<80
    c=np.arange(15,dtype=int)
    b=np.asarray([15],dtype=int)
    cols=('P_Solar_kw_issue_minus_24h','valid_utc_year')+tuple('valid_extra_'+str(i) for i in range(2,16))
    return seq,aux,target,mask,c,b,cols


def call(scaler,data):
    seq,aux,y,mask,c,b,cols=data
    return scaler.fit_train_only_scalers_v2(seq,aux,y,mask,
        continuous_positions=c,binary_positions=b,auxiliary_columns=cols)


def test_year_constant_train_std_one_and_train_only(scaler):
    data=toy()
    state=call(scaler,data)
    assert state['n_train']==80
    assert state['auxiliary_mean'][1]==2018.0
    assert state['auxiliary_std'][1]==1.0
    assert state['auxiliary_std'][0]>0
    assert state['ddof']==0
    assert state['target_mean']==pytest.approx(np.asarray(data[2][:80],dtype=float).mean())
    assert not scaler._legacy_calls
    normalized=(data[1][:,1]-state['auxiliary_mean'][1])/state['auxiliary_std'][1]
    assert np.all(normalized[:80]==0.0)
    assert np.all(normalized[80:]==1.0)


def test_outer_validation_values_do_not_fit_scaler(scaler):
    data=list(toy())
    first=call(scaler,data)
    changed=list(data);changed[1]=data[1].copy();changed[2]=data[2].copy()
    changed[1][80:,1]=9999
    changed[2][80:]+=1000
    second=call(scaler,changed)
    for key in ('sequence_mean','sequence_std','auxiliary_mean','auxiliary_std'):
        np.testing.assert_array_equal(first[key],second[key])
    assert first['target_mean']==second['target_mean']
    assert first['target_std']==second['target_std']


def test_varying_year_delegates_to_frozen_scaler(scaler):
    data=list(toy());data[1]=data[1].copy();data[1][40:80,1]=2019.
    value=call(scaler,data)
    assert value=={'frozen_scaler_used':True}
    assert len(scaler._legacy_calls)==1


def test_only_year_may_be_constant(scaler):
    data=list(toy());data[1]=data[1].copy();data[1][:,3]=4.0
    with pytest.raises(ValueError,match='Unexpected constant continuous'):
        call(scaler,data)


def test_bad_binary_indicator_fails(scaler):
    data=list(toy());data[1]=data[1].copy();data[1][5,15]=2.0
    with pytest.raises(ValueError,match='Binary'):
        call(scaler,data)


@pytest.mark.parametrize('kind', ['renamed','shifted','two-year-columns'])
def test_year_schema_fails_closed(scaler,kind):
    data=list(toy());cols=list(data[6])
    if kind=='renamed': cols[1]='valid_year'
    if kind=='shifted': cols[1],cols[2]=cols[2],cols[1]
    if kind=='two-year-columns': cols[3]='valid_utc_year'
    data[6]=cols
    with pytest.raises(ValueError,match='year-column'):
        call(scaler,data)


def test_variable_sequence_invalid(scaler):
    data=list(toy());data[0]=np.ones_like(data[0])
    with pytest.raises(ValueError,match='sequence channel'):
        call(scaler,data)


def test_variable_target_invalid(scaler):
    data=list(toy());data[2]=np.ones_like(data[2])
    with pytest.raises(ValueError,match='positive variance'):
        call(scaler,data)


def test_no_data_or_checkpoint_write_from_import(tmp_path):
    before=set(tmp_path.iterdir())
    load('src/revised_neural_scaling_v2.py','test_m6_import_no_side_effect')
    assert set(tmp_path.iterdir())==before


def test_v2_candidate_source_is_separate_from_frozen_v1():
    source=(PKG/'src/revised_neural_campaign_v2.py').read_text()
    assert "'src/revised_neural_campaign.py'" in source
    assert "'src/revised_neural_campaign_v2.py'" in source
    assert "'src/revised_neural_scaling_v2.py'" in source
    assert "FAMILIES = ('F1', 'F2', 'F3', 'F4')" in source
    assert "SCHEMA_VERSION = 2" in source
    assert "'scaling_policy':SCALING_POLICY" in source
    assert 'fit_train_only_scalers_v2 as fit_train_only_scalers' in source


def test_mixed_runner_routes_only_f0_to_v1():
    source=(PKG/'src/revised_neural_batch_v2.py').read_text()
    assert "v1 = family == 'F0'" in source
    assert "directory = v1_dir if v1 else v2_dir" in source
    assert "run_v1 if v1 else run_v2" in source
    assert "'.neural_batch.lock'" in source


def test_installer_preserves_existing_outputs_and_is_idempotent():
    m=load('scripts/append_neural_m6.py','test_m6_installer')
    ids=['m4-campaign-preflight','m4-run-all-summary',m.M5_ID]
    cells=[{'id':name,'cell_type':'code','source':['M5_RUN_BATCH = False\n'],
            'execution_count':1,'outputs':[{'output_type':'stream','name':'stdout','text':['SAVED\n']}],
            'metadata':{}} for name in ids]
    old=json.loads(json.dumps(cells))
    nb={'cells':cells,'metadata':{'solete_revised_neural_protocol':{'terminal_test_evaluated':False}}}
    assert m.append_m6(nb)
    assert nb['cells'][:3]==old
    assert len(nb['cells'])==6
    assert m.append_m6(nb) is False
    assert nb['cells'][:3]==old


def test_installer_refuses_enabled_m5_or_terminal_access():
    m=load('scripts/append_neural_m6.py','test_m6_installer2')
    ids=['m4-campaign-preflight','m4-run-all-summary',m.M5_ID]
    cells=[{'id':name,'cell_type':'code','source':['M5_RUN_BATCH = True\n']} for name in ids]
    nb={'cells':cells,'metadata':{'solete_revised_neural_protocol':{'terminal_test_evaluated':False}}}
    with pytest.raises(ValueError,match='DISABLED'):
        m.append_m6(nb)
    cells[-1]['source']=['M5_RUN_BATCH = False\n']
    nb['metadata']['solete_revised_neural_protocol']['terminal_test_evaluated']=True
    with pytest.raises(ValueError,match='terminal'):
        m.append_m6(nb)
