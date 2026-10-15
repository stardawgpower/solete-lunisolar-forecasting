"""Synthetic contract checks for revised neural input packing; no model fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.revised_neural_cv import (
    CHANNEL_NAMES, F0_COLUMNS, AUXILIARY_DIMENSIONS, FAMILY_DIMENSIONS,
    materialize_neural_inputs, prepare_development_fold,
)


def toy(family="F0"):
    n = 90
    index = pd.date_range("2018-06-01", periods=n, freq="5min", tz="UTC")
    values = {}
    for c, name in enumerate(CHANNEL_NAMES):
        for lag in range(13):
            values[f"{name}_lag_{lag}"] = (np.arange(n) * 0.02 + c * 100 + lag).astype(float)
    values[F0_COLUMNS[-1]] = np.arange(n, dtype=float) * .1 + 1
    suffix = ["valid_utc_is_weekend"] + [f"valid_calendar_{j}" for j in range(14)]
    suffix += ["valid_solar_elevation_deg", "valid_solar_cos_zenith",
               "valid_solar_azimuth_sin", "valid_solar_azimuth_cos",
               "valid_solar_daylight"]
    suffix += [f"valid_lunar_{j}" for j in range(9)]
    suffix += [f"valid_tithi_{j}" for j in range(97)]
    features = list(F0_COLUMNS) + suffix[:FAMILY_DIMENSIONS[family]-118]
    for col in features[118:]:
        values[col] = (np.arange(n, dtype=float) / 100
                       if col not in {"valid_solar_daylight", "valid_utc_is_weekend"} and not col.startswith("valid_tithi_")
                       else (np.arange(n) % 2).astype(float))
    X = pd.DataFrame(values, index=index).loc[:,features]
    valid_time = pd.Series(index + pd.Timedelta(minutes=15), index=index)
    issue_time = pd.Series(index, index=index)
    forbidden_mask = pd.Series(np.arange(n) >= 60, index=index)
    target = pd.Series(np.arange(n, dtype=float) * .3, index=index).mask(forbidden_mask)
    fold = {"fold":1, "train_mask":pd.Series(np.arange(n)<20, index=index),
            "validation_mask":pd.Series((np.arange(n)>=33) & (np.arange(n)<49), index=index)}
    args=dict(X=X, masked_target=target, issue_time=issue_time,
              valid_time=valid_time, forbidden_mask=forbidden_mask,
              fold=fold,horizon_minutes=15, family=family)
    return args


@pytest.mark.parametrize("family", list(FAMILY_DIMENSIONS))
def test_sequence_and_aux_shapes_and_chronology(family):
    args=toy(family)
    p=materialize_neural_inputs(args["X"],family)
    assert p["sequence"].shape==(90,13,9)
    assert p["auxiliary"].shape==(90,AUXILIARY_DIMENSIONS[family])
    np.testing.assert_allclose(p["sequence"][:,0,0],args["X"]["P_Solar_kw_lag_12"],rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(p["sequence"][:,-1,0],args["X"]["P_Solar_kw_lag_0"],rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(p["auxiliary"][:,0],args["X"][F0_COLUMNS[-1]],rtol=1e-6,atol=1e-5)
    if family=="F4":
        assert len(p["binary_positions"])==99
    if family == "F0":
        assert len(p["binary_positions"])==0
    if family == "F1":
        assert len(p["binary_positions"])==1


def test_fold_excludes_holdout_and_splits_train_first():
    p=prepare_development_fold(**toy("F4"))
    assert (p["train_rows"],p["validation_rows"])==(20,16)
    assert p["sequence"].shape==(36,13,9)
    assert p["train_mask"].sum()==20
    assert p["validation_mask"].sum()==16
    assert not (p["train_mask"] & p["validation_mask"]).any()
    assert p["issue_time"].max() < pd.Timestamp("2018-06-01", tz="UTC") + pd.Timedelta(minutes=5*60)


def test_refuses_unmasked_terminal_target():
    d=toy(); d["masked_target"].iloc[-1]=17.
    with pytest.raises(ValueError,match="Terminal target must be masked"):
        prepare_development_fold(**d)


def test_refuses_holdout_mask_overlap():
    d=toy(); d["fold"]["train_mask"].iloc[61]=True
    with pytest.raises(ValueError,match="holdout access"):
        prepare_development_fold(**d)


def test_refuses_label_availability_violation():
    d=toy(); d["fold"]["train_mask"].iloc[32]=True
    with pytest.raises(ValueError,match="not available"):
        prepare_development_fold(**d)


def test_refuses_wrong_valid_time_alignment():
    d=toy();d["valid_time"].iloc[0]+=pd.Timedelta(minutes=5)
    with pytest.raises(ValueError,match="alignment"):
        prepare_development_fold(**d)


def test_refuses_mismatched_columns():
    d=toy(); X=d["X"].rename(columns={"P_Solar_kw_lag_12":"P_Solar_kw_lag_13"})
    with pytest.raises(ValueError,match="canonical Notebook 10"):
        materialize_neural_inputs(X,"F0")
