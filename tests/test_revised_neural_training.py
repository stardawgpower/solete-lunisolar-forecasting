"""Milestone 2 development-only protocol tests. No TensorFlow training."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.revised_neural_training import (
    LSTM_CANDIDATES, TCN_CANDIDATES, INNER_LABEL_GAP_MINUTES,
    make_inner_split, pilot_indices, rank_neural_cv_scores,
    _checkpoint, _array_signature,
)


def data():
    n = 300
    full = pd.date_range("2018-06-01", periods=n + 12, freq="5min", tz="UTC")
    ts = full[:180].append(full[192:])  # 60-minute omitted outer-gap rows
    idx = pd.Index(range(n))
    # This package holds only 270 development rows; there is no terminal data.
    target = (1 + np.arange(n) * 0.003).astype("float32")
    return {
        "sequence": np.ones((n, 13, 9), dtype="float32") + (np.arange(n)/1000)[:,None,None],
        "auxiliary": np.ones((n, 1), dtype="float32") + np.arange(n, dtype="float32")[:,None]/100,
        "target_kw": target,
        "train_mask": np.arange(n) < 180,
        "validation_mask": np.arange(n) >= 180,
        "issue_time": pd.Series(ts, index=idx),
        "valid_time": pd.Series(ts + pd.Timedelta(minutes=15), index=idx),
        "horizon_minutes": 15,
        "family": "F0",
        "fold": 1,
        "continuous_positions": np.asarray([0], dtype="int64"),
        "binary_positions": np.asarray([], dtype="int64"),
    }


def test_frozen_candidate_grids_and_no_reused_winners():
    assert [c["candidate_id"] for c in LSTM_CANDIDATES] == [f"LSTM_C{i}" for i in range(1, 5)]
    assert [c["candidate_id"] for c in TCN_CANDIDATES] == [f"TCN_C{i}" for i in range(1, 5)]
    assert len({c["candidate_id"] for c in LSTM_CANDIDATES + TCN_CANDIDATES}) == 8
    assert tuple(TCN_CANDIDATES[0]["dilations"]) == (1, 2, 4)


def test_inner_split_is_disjoint_and_embargoed():
    p = data()
    m = make_inner_split(p)
    fit, val = np.flatnonzero(m["inner_train_mask"]), np.flatnonzero(m["inner_validation_mask"])
    assert fit.size >= 10 and val.size >= 5
    assert fit[-1] < val[0] < np.flatnonzero(p["validation_mask"])[0]
    assert not np.any((m["inner_train_mask"] | m["inner_validation_mask"]) & p["validation_mask"])
    assert (p["valid_time"].iloc[val[0]] - p["valid_time"].iloc[fit[-1]]) >= pd.Timedelta(minutes=INNER_LABEL_GAP_MINUTES + 5)
    assert p["valid_time"].iloc[fit[-1]] < p["issue_time"].iloc[val[0]]
    assert m["embargo_rows"] > 0


def test_different_horizons_get_same_inner_gap():
    p = data()
    p["horizon_minutes"] = 60
    p["valid_time"] = p["issue_time"] + pd.Timedelta(minutes=60)
    m = make_inner_split(p)
    fit = np.flatnonzero(m["inner_train_mask"])
    val = np.flatnonzero(m["inner_validation_mask"])
    assert p["valid_time"].iloc[fit[-1]] < p["issue_time"].iloc[val[0]]


def test_pilot_never_reaches_outer_validation():
    p = data()
    i, j = pilot_indices(p, train_limit=48, validation_limit=17)
    assert len(i) == 48 and len(j) == 17
    assert j[-1] < np.flatnonzero(p["validation_mask"])[0]
    p["target_kw"][p["validation_mask"]] = 1e9
    i2, j2 = pilot_indices(p, train_limit=48, validation_limit=17)
    np.testing.assert_array_equal(i, i2)
    np.testing.assert_array_equal(j, j2)


@pytest.mark.parametrize("bad", [0, -1, .8, float("nan")])
def test_bad_inner_fraction_fails(bad):
    with pytest.raises(ValueError):
        make_inner_split(data(), fraction=bad)


def test_overlap_and_wrong_time_alignment_fail():
    p = data()
    p["validation_mask"][179] = True
    with pytest.raises(ValueError, match="overlap"):
        make_inner_split(p)
    p = data()
    p["valid_time"].iloc[0] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="alignment"):
        make_inner_split(p)


def test_pilot_caps_must_be_integer():
    with pytest.raises(ValueError, match="limits"):
        pilot_indices(data(), train_limit=1)
    with pytest.raises(ValueError, match="limits"):
        pilot_indices(data(), validation_limit=16.0)


def full_scores():
    records = []
    # Unweighted mean fold MAE: candidate 0 -> mean 4, candidate 1 -> mean 2.
    for candidate in (0, 1):
        for fold in (1, 2, 3):
            records.append(dict(model="LSTM", horizon_minutes=15, feature_family="F0",
                                config_index=candidate, candidate_id=f"LSTM_C{candidate+1}", fold=fold,
                                validation_mae_kw=[1., 1., 10.][fold-1] if candidate == 0 else 2.,
                                validation_rmse_kw=3., best_epoch=5))
    return pd.DataFrame(records)


def test_rank_requires_three_folds_and_mean_kw():
    s = full_scores()
    ranked = rank_neural_cv_scores(s)
    assert ranked["config_index"].tolist() == [1, 0]
    assert ranked["rank"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="three distinct folds"):
        rank_neural_cv_scores(s.iloc[:-1].copy())
    with pytest.raises(ValueError, match="duplicate"):
        rank_neural_cv_scores(pd.concat([s, s.iloc[[0]]]))


def test_rank_exact_ties_broken_by_index_not_rmse():
    s = full_scores()
    s["validation_mae_kw"] = 3.
    s.loc[s["config_index"].eq(0), "validation_rmse_kw"] = 9.
    s.loc[s["config_index"].eq(1), "validation_rmse_kw"] = 1.
    assert rank_neural_cv_scores(s)["config_index"].tolist() == [0, 1]


def test_checkpoint_resume_fails_closed(tmp_path, monkeypatch):
    # Isolated fake atomic implementation exercises writer integration without TensorFlow.
    mod = types.ModuleType("src.revised_rf_checkpoint")
    def atomic_text(p, s, **kwargs):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(s, encoding="utf-8")
    mod.atomic_text = atomic_text
    monkeypatch.setitem(sys.modules, "src.revised_rf_checkpoint", mod)
    path = tmp_path / "pilot.json"
    identity = {"data_sha256": "x", "configuration": {"dilations": [1, 2, 4]}}
    record = {"elapsed_seconds": .4, "trainable_parameters": 321,
              "train_rows": 64, "inner_validation_rows": 16, "epochs": 1}
    assert _checkpoint(path, identity) is None
    assert _checkpoint(path, identity, record) == record
    assert _checkpoint(path, identity) == record
    assert json.loads(path.read_text())["kind"] == "resource-pilot-not-cv"
    with pytest.raises(ValueError, match="identity mismatch"):
        _checkpoint(path, {"data_sha256": "different"})
    obj = json.loads(path.read_text())
    obj["record"]["epochs"] = 2
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Invalid pilot"):
        _checkpoint(path, identity)


def test_array_signature_detects_changed_data():
    a = np.array([1.,2.],dtype=np.float32)
    b = a.copy(); b[0] = 3.
    assert _array_signature([a]) != _array_signature([b])


def test_pilot_run_and_resume_with_mock_keras(tmp_path, monkeypatch):
    """Exercise both resource models, sealed subset, and JSON resume, no TensorFlow."""
    import src.revised_neural_training as training
    p = data()
    # Huge OUTER-validation target perturbation must never enter the fitted subset.
    p["target_kw"][p["validation_mask"]] = 1e9
    fitted = []
    class FakeModel:
        def __init__(self, name): self.name = name
        def count_params(self): return 1234
        def fit(self, inputs, target, **kwargs):
            assert len(target) > 0 and len(target) <= 2048
            assert np.max(target) < 1e9
            assert len(kwargs["validation_data"][1]) <= 512
            assert np.max(kwargs["validation_data"][1]) < 1e9
            assert kwargs["epochs"] == 1
            fitted.append(self.name)
            return types.SimpleNamespace(history={"loss": [1.]})
    faux_nn = types.ModuleType("src.neural_models")
    faux_nn.fit_train_only_scalers = lambda sequence, auxiliary, target, mask, **kw: {"train_mask": mask}
    faux_nn.apply_train_only_scalers = lambda sequence, auxiliary, target, stats: {
        "sequence": sequence, "auxiliary": auxiliary, "target_scaled": target,
    }
    faux_nn.build_recurrent_fusion_model = lambda *a, **kw: FakeModel("LSTM")
    faux_nn.build_tcn_fusion_model = lambda *a, **kw: FakeModel("TCN")
    faux_nn.reset_neural_state = lambda *a, **kw: None
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.__version__ = "fake-tf"
    fake_tf.keras = types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda: None))
    fake_ckpt = types.ModuleType("src.revised_rf_checkpoint")
    def atomic_text(path, contents, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != contents:
            raise ValueError("Cannot overwrite")
        path.write_text(contents)
    fake_ckpt.atomic_text = atomic_text
    monkeypatch.setitem(sys.modules, "src.neural_models", faux_nn)
    monkeypatch.setitem(sys.modules, "src.revised_rf_checkpoint", fake_ckpt)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(training, "source_fingerprint", lambda root: "frozen-sha")
    for name in ("LSTM", "TCN"):
        record, did_fit = training.run_resource_pilot(
            payload=p, root=tmp_path, model_name=name,
            source_sha256="frozen-sha", output_dir=tmp_path / "pilots")
        assert did_fit and record["epochs"] == 1
        assert record["trainable_parameters"] == 1234
        assert record["elapsed_seconds"] > 0
    assert fitted == ["LSTM", "TCN"]
    for name in ("LSTM", "TCN"):
        _, did_fit = training.run_resource_pilot(
            payload=p, root=tmp_path, model_name=name,
            source_sha256="frozen-sha", output_dir=tmp_path / "pilots")
        assert not did_fit
    assert fitted == ["LSTM", "TCN"]
    with pytest.raises(ValueError, match="identity mismatch"):
        altered = data()
        altered["target_kw"][10] = 999.
        training.run_resource_pilot(
            payload=altered, root=tmp_path, model_name="TCN",
            source_sha256="frozen-sha", output_dir=tmp_path / "pilots")
