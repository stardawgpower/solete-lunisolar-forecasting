"""Milestone 3 full-size gate guards with synthetic fits; no TensorFlow needed."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.revised_neural_full_cv as gate
from src.revised_neural_training import make_inner_split


def toy():
    n = 300
    timestamps = pd.date_range('2018-06-01', periods=n + 12, freq='5min', tz='UTC')
    index = timestamps[:180].append(timestamps[192:])
    return {
        'sequence': (np.arange(n, dtype='float32')[:, None, None] / 1000
                     + np.ones((n, 13, 9), dtype='float32')),
        'auxiliary': (np.arange(n, dtype='float32')[:, None] / 100 + 1),
        'target_kw': np.arange(n, dtype='float32') / 300 + 0.5,
        'train_mask': np.arange(n) < 180,
        'validation_mask': np.arange(n) >= 180,
        'issue_time': pd.Series(index, index=pd.RangeIndex(n)),
        'valid_time': pd.Series(index + pd.Timedelta(minutes=15), index=pd.RangeIndex(n)),
        'horizon_minutes': 15, 'family': 'F0', 'fold': 1,
        'train_rows': 180, 'validation_rows': 120,
        'continuous_positions': np.array([0], dtype=np.int64),
        'binary_positions': np.array([], dtype=np.int64),
    }


@pytest.fixture
def small_gate(monkeypatch):
    monkeypatch.setattr(gate, 'GATE_TRAIN_ROWS', 180)
    monkeypatch.setattr(gate, 'GATE_VALIDATION_ROWS', 120)
    return toy()


@pytest.mark.parametrize('history,want', [([.4,.3,.2], 3), ([.2,.2,.5], 1), ([.5], 1)])
def test_choose_best_epoch(history, want):
    assert gate.choose_best_epoch(history)[0] == want


@pytest.mark.parametrize('history', [[], [float('nan')], [-.1], [float('inf')], [1.] * 81, [[1.,2.]]])
def test_epoch_rejects_invalid_history(history):
    with pytest.raises(ValueError):
        gate.choose_best_epoch(history)


def test_real_frozen_counts_and_scope_are_fixed():
    assert (gate.GATE_HORIZON, gate.GATE_FAMILY, gate.GATE_FOLD) == (15, 'F0', 1)
    assert (gate.GATE_TRAIN_ROWS, gate.GATE_VALIDATION_ROWS) == (52401, 17856)
    assert gate.MAX_SELECTION_EPOCHS == 80
    assert gate.GATE_MODELS == ('LSTM', 'TCN')
    assert 'not-ranked' in gate.GATE_KIND


def test_gate_requires_complete_fold_and_inner_gap(small_gate):
    train, val, inner = gate.validate_fullsize_payload(small_gate)
    assert (len(train), len(val)) == (180,120)
    assert not np.any((inner['inner_train_mask'] | inner['inner_validation_mask']) & small_gate['validation_mask'])
    assert inner['embargo_rows'] == 12
    assert small_gate['valid_time'].iloc[np.flatnonzero(inner['inner_train_mask'])[-1]] < small_gate['issue_time'].iloc[np.flatnonzero(inner['inner_validation_mask'])[0]]


def test_gate_rejects_shortcut_and_nonf0(small_gate):
    bad = dict(small_gate, train_rows=179)
    with pytest.raises(ValueError,match='metadata'):
        gate.validate_fullsize_payload(bad)
    bad = dict(small_gate, family='F4')
    with pytest.raises(ValueError,match='restricted'):
        gate.validate_fullsize_payload(bad)
    bad = dict(small_gate, auxiliary=np.ones((300,2),np.float32))
    with pytest.raises(ValueError,match='one auxiliary'):
        gate.validate_fullsize_payload(bad)
    bad = dict(small_gate, fold=2)
    with pytest.raises(ValueError,match='restricted'):
        gate.validate_fullsize_payload(bad)


def test_gate_rejects_overlap_nonfinite_wrong_times(small_gate):
    bad = dict(small_gate, train_mask=small_gate['train_mask'].copy())
    bad['train_mask'][-1] = True
    with pytest.raises(ValueError,match='overlap'):
        gate.validate_fullsize_payload(bad)
    bad = dict(small_gate, target_kw=small_gate['target_kw'].copy())
    bad['target_kw'][90] = np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        gate.validate_fullsize_payload(bad)
    bad = dict(small_gate, valid_time=small_gate['valid_time'].copy())
    bad['valid_time'].iloc[0] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError,match='alignment'):
        gate.validate_fullsize_payload(bad)


def test_payload_fingerprint_catches_outer_label_and_timestamp_changes(small_gate):
    initial = gate._hash_payload(small_gate)
    changed = dict(small_gate, target_kw=small_gate['target_kw'].copy())
    changed['target_kw'][-1] += 10
    assert gate._hash_payload(changed) != initial
    changed = dict(small_gate, issue_time=small_gate['issue_time'].copy())
    changed['issue_time'].iloc[3] += pd.Timedelta(minutes=5)
    assert gate._hash_payload(changed) != initial


def test_gate_source_fingerprint_catches_modified_file(tmp_path):
    files = ('forecasting_protocol.py','neural_models.py','revised_neural_cv.py',
             'revised_neural_training.py','revised_neural_full_cv.py')
    (tmp_path / 'src').mkdir()
    for name in files:
        (tmp_path / 'src' / name).write_text('a')
    first = gate.gate_source_fingerprint(tmp_path)
    (tmp_path / 'src' / 'neural_models.py').write_text('b')
    assert first != gate.gate_source_fingerprint(tmp_path)


def _fake_modules(monkeypatch):
    """Distinct fresh models, audited train rows, epoch policy, and fake predictions."""
    audit = []
    class Model:
        def __init__(self, name): self.name=name
        def count_params(self): return 1234
        def fit(self, inputs, y, **kw):
            # Expect inner stage only sees indices <180; refit sees full 180.
            audit.append(('fit', self.name, len(y), kw['epochs'], bool('validation_data' in kw)))
            if 'validation_data' in kw:
                assert len(y) < 180
                assert kw['epochs'] == 80 and len(kw['callbacks']) == 1
                assert len(kw['validation_data'][1]) > 0
                return types.SimpleNamespace(history={'val_loss':[.8,.4,.5], 'loss':[.9,.6,.55]})
            assert len(y) == 180 and kw['epochs'] == 2
            return types.SimpleNamespace(history={'loss':[.5,.4]})
        def predict(self, inputs, **kw):
            audit.append(('predict', self.name, len(inputs[0])))
            return np.full((len(inputs[0]), 1), 0.7, dtype=np.float32)
    def fit_scalers(seq,aux,target,train_mask,**kw):
        assert len(seq) == len(target) == len(train_mask)
        assert np.max(target) < 1000  # outer labels poisoned in other test; not passed
        if len(target)==180:
            assert np.all(train_mask)
        else:
            assert train_mask.sum() < len(target)
        audit.append(('scale-fit',len(target),int(train_mask.sum())))
        return {'a':1}
    def apply_scalers(seq,aux,target,stats):
        return {'sequence':seq, 'auxiliary':aux, 'target_scaled':target}
    fake_nn=types.ModuleType('src.neural_models')
    fake_nn.fit_train_only_scalers=fit_scalers
    fake_nn.apply_train_only_scalers=apply_scalers
    fake_nn.inverse_target_scaling=lambda predictions,stats: predictions
    fake_nn.reset_neural_state=lambda seed: audit.append(('reset',seed))
    fake_nn.build_recurrent_fusion_model=lambda *a,**k: Model('LSTM')
    fake_nn.build_tcn_fusion_model=lambda *a,**k: Model('TCN')
    tf=types.ModuleType('tensorflow')
    tf.__version__='mock-keras-1'
    tf.keras=types.SimpleNamespace(
        backend=types.SimpleNamespace(clear_session=lambda: None),
        callbacks=types.SimpleNamespace(EarlyStopping=lambda **kw: kw),
    )
    monkeypatch.setitem(sys.modules,'src.neural_models',fake_nn)
    monkeypatch.setitem(sys.modules,'tensorflow',tf)
    fake_archive=types.ModuleType('src.revised_rf_checkpoint')
    def atomic_text(path,content,*,refuse_different=False):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and refuse_different and path.read_text()!=content:
            raise ValueError('Refuse overwrite')
        path.write_text(content,encoding='utf-8')
    fake_archive.atomic_text=atomic_text
    monkeypatch.setitem(sys.modules,'src.revised_rf_checkpoint',fake_archive)
    return audit


@pytest.mark.parametrize('model', ['LSTM','TCN'])
def test_full_training_gate_and_reuse_without_tensorflow(tmp_path,small_gate,monkeypatch,model):
    audit=_fake_modules(monkeypatch)
    monkeypatch.setattr(gate,'gate_source_fingerprint',lambda root:'frozen-sha')
    # Outer-validation targets contain a poison value; no inner scale/fit should see it.
    small_gate['target_kw'][small_gate['validation_mask']] = 2_000.
    kwargs=dict(payload=small_gate,root=tmp_path,model_name=model,
                source_sha256='frozen-sha',output_dir=tmp_path/'gate')
    record,trained=gate.run_full_size_gate(**kwargs)
    assert trained
    assert record['best_epoch']==2 and record['inner_epochs_completed']==3
    assert record['inner_training_rows']+record['inner_validation_rows']+record['inner_embargo_rows']==180
    assert record['validation_mae_kw'] > 1000
    assert len([e for e in audit if e[0]=='fit'])==2
    assert [e for e in audit if e[0]=='fit'][0][-1] is True
    assert [e for e in audit if e[0]=='fit'][1][-1] is False
    assert len([e for e in audit if e[0]=='predict'])==1
    path=kwargs['output_dir']/f'{model.lower()}_c1_h15_F0_fold1.json'
    saved=json.loads(path.read_text())
    assert saved['kind']==gate.GATE_KIND
    assert not {'predictions','test_metrics','terminal_labels'} & set(saved['record'])
    original_audit=list(audit)
    reused,trained=gate.run_full_size_gate(**kwargs)
    assert not trained and reused==record and audit==original_audit
    mismatch=dict(small_gate,target_kw=small_gate['target_kw'].copy())
    mismatch['target_kw'][-1]=3_000.
    with pytest.raises(ValueError,match='identity mismatch'):
        gate.run_full_size_gate(**dict(kwargs,payload=mismatch))
    assert not path.with_suffix('.lock').exists()


def test_checkpoint_rejects_corruption_and_stale_lock(tmp_path,small_gate,monkeypatch):
    _fake_modules(monkeypatch)
    monkeypatch.setattr(gate,'gate_source_fingerprint',lambda root:'same')
    kw=dict(payload=small_gate,root=tmp_path,model_name='LSTM',
            source_sha256='same',output_dir=tmp_path/'gate')
    path=kw['output_dir']/'lstm_c1_h15_F0_fold1.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.lock').write_text('stale')
    with pytest.raises(RuntimeError,match='locked'):
        gate.run_full_size_gate(**kw)
    path.with_suffix('.lock').unlink()
    gate.run_full_size_gate(**kw)
    state=json.loads(path.read_text())
    state['record']['outer_training_rows'] = 3
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='wrong frozen fold counts'):
        gate.run_full_size_gate(**kw)
    assert not path.with_suffix('.lock').exists()


def test_refuses_bad_model_or_bad_source(tmp_path,small_gate,monkeypatch):
    monkeypatch.setattr(gate,'gate_source_fingerprint',lambda root:'the-right-source')
    with pytest.raises(ValueError,match='Only LSTM'):
        gate.run_full_size_gate(payload=small_gate,root=tmp_path,model_name='GRU',
                                source_sha256='the-right-source',output_dir=tmp_path)
    with pytest.raises(ValueError,match='fingerprint'):
        gate.run_full_size_gate(payload=small_gate,root=tmp_path,model_name='LSTM',
                                source_sha256='incorrect',output_dir=tmp_path)


def test_append_is_additive_idempotent_and_training_off():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from scripts.append_neural_m3 import append_milestone3, NEW_CELLS, M3_IDS
    m={'terminal_test_evaluated':False,'milestone2_pilot_executed':True,
       'development_cv_models_trained':False,
       'phase':'milestone2-inner-split-and-resource-pilot-complete'}
    cells=[{'id':f'safe-{i}','cell_type':'code','source':['a\n'],
            'execution_count':2,'outputs':[{'output_type':'stream','text':['passed\n']}]} for i in range(21)]
    cells.append({'id':'m2-revised-neural-completion-gate','cell_type':'markdown','source':['stop']})
    nb={'nbformat':4,'cells':cells,'metadata':{'solete_revised_neural_protocol':m}}
    prior=json.dumps(nb['cells'])
    original_ids = {x['id'] for x in cells}
    assert append_milestone3(nb)
    assert json.dumps(nb['cells'][:22])==prior
    assert tuple(c['id'] for c in nb['cells'][-7:])==M3_IDS
    assert not append_milestone3(nb)
    assert nb['metadata']['solete_revised_neural_protocol']['terminal_test_evaluated'] is False
    for c in NEW_CELLS:
        assert c['id'] not in original_ids
    policy=''.join(nb['cells'][23]['source'])
    assert 'M3_RUN_LSTM = False' in policy
    assert 'M3_RUN_TCN = False' in policy
    assert nb['cells'][25]['execution_count'] is None
