"""Read-only tests for GRU/BiLSTM campaign extension."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src import revised_neural_campaign as v1
from src import revised_neural_campaign_v2 as v2
from src import revised_recurrent_extension as ext
from src.revised_neural_training import LSTM_CANDIDATES


def test_exact_complete_grid():
    keys = ext.expected_keys()
    assert len(keys) == 360 and len(set(keys)) == 360
    assert len(ext.expected_keys(("GRU",))) == 180
    assert len(ext.expected_keys(("BiLSTM",))) == 180
    assert keys[0] == ("GRU", 15, "F0", 0, 1)
    assert keys[-1] == ("BiLSTM", 60, "F4", 3, 3)


def test_recurrent_configs_match_original_lstm_shapes():
    for model in ext.MODELS:
        configs = ext.candidates(model)
        for index, config in enumerate(configs):
            assert config["candidate_id"] == f"{model}_C{index+1}"
            assert {k: v for k, v in config.items() if k != "candidate_id"} == {
                k: v for k, v in LSTM_CANDIDATES[index].items() if k != "candidate_id"
            }


def test_only_new_models_permitted():
    with pytest.raises(ValueError):
        ext.candidates("LSTM")
    with pytest.raises(ValueError):
        ext.checkpoint_name("TCN", 15, "F0", 0, 1)
    with pytest.raises(ValueError):
        ext.expected_keys(("GRU", "GRU"))
    with pytest.raises(ValueError):
        ext.checkpoint_name("GRU", 15, "F0", 4, 1)
    with pytest.raises(ValueError):
        ext.checkpoint_name("GRU", 15, "F0", 0, 4)


def test_frozen_v1_v2_globals_are_unchanged():
    # The historical runtime functions are reused through private namespaces,
    # without mutating originals and invalidating 360 existing checkpoints.
    assert v1._train_one.__globals__["_make_model"] is v1._make_model
    assert v2._train_one.__globals__["_make_model"] is v2._make_model
    assert ext._F0_TRAIN.__globals__["_make_model"] is ext._build_model
    assert ext._V2_TRAIN.__globals__["_make_model"] is ext._build_model
    assert ext._F0_ONE.__globals__["_train_one"] is ext._F0_TRAIN
    assert ext._V2_ONE.__globals__["_train_one"] is ext._V2_TRAIN
    assert ext._F0_ONE.__code__ is v1.run_one_cv.__code__
    assert ext._V2_ONE.__code__ is v2.run_one_cv.__code__
    assert ext._F0_TRAIN.__code__ is v1._train_one.__code__
    assert ext._V2_TRAIN.__code__ is v2._train_one.__code__


def test_f0_and_f1_f4_use_correct_frozen_protocol():
    assert ext._F0_ONE.__globals__["validate_campaign_payload"] is v1.validate_campaign_payload
    assert ext._V2_ONE.__globals__["validate_campaign_payload"] is v2.validate_campaign_payload
    assert ext._F0_TRAIN.__globals__["_make_model"] is ext._build_model
    assert ext._V2_TRAIN.__globals__["_make_model"] is ext._build_model
    assert ext._F0_READ.__globals__["checkpoint_name"] is ext.checkpoint_name
    assert ext._V2_READ.__globals__["checkpoint_name"] is ext.checkpoint_name


def test_empty_extension_is_valid_and_read_only(tmp_path):
    report = ext.read_progress(tmp_path, verify_source=False)
    assert report.empty
    assert not (tmp_path / "results").exists()


def test_batch_disabled_cannot_fit_or_write(tmp_path, monkeypatch):
    cols = ("model", "horizon_minutes", "feature_family", "config_index", "fold")
    empty = pd.DataFrame(columns=cols)
    monkeypatch.setattr(ext, "read_mixed_progress", lambda *args: pd.DataFrame(index=range(360)))
    monkeypatch.setattr(ext, "read_progress", lambda root: empty)
    monkeypatch.setattr(ext, "_verify_existing_data", lambda root, table: None)
    payload = Mock(side_effect=AssertionError("No payload construction when disabled"))
    out = ext.run_batch(root=tmp_path, payload_factory=payload, enabled=False)
    payload.assert_not_called()
    assert out["after"] == 0 and out["new_fits"] == 0
    assert not (tmp_path / "results").exists()


def test_bad_switch_or_model_fails_closed(tmp_path):
    payload = lambda *args: None
    with pytest.raises(ValueError):
        ext.run_batch(root=tmp_path, payload_factory=payload, enabled=1)
    with pytest.raises(ValueError):
        ext.run_batch(root=tmp_path, payload_factory=payload, models=("LSTM",))
    with pytest.raises(ValueError):
        ext.run_batch(root=tmp_path, payload_factory=payload, max_new_fits=0)
