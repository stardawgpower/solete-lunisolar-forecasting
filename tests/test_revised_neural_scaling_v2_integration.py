"""Live frozen-scaler compatibility smoke (no TensorFlow fitting, no SOLETE IO)."""
from __future__ import annotations
import numpy as np
import pytest


def test_corrected_year_scaler_is_accepted_by_frozen_apply_scaler():
    neural = pytest.importorskip('src.neural_models', reason='Requires user SOLETE repository')
    from src.revised_neural_scaling_v2 import fit_train_only_scalers_v2
    n=128
    rng=np.random.default_rng(102)
    seq=rng.normal(size=(n,13,9)).astype(np.float32)
    aux=rng.normal(size=(n,16)).astype(np.float32)
    aux[:96,1]=2018.
    aux[96:,1]=2019.
    aux[:,15]=(np.arange(n)%2).astype(np.float32)
    target=np.arange(n,dtype=np.float32)/30.0
    mask=np.arange(n)<96
    columns=('P_Solar_kw_issue_minus_24h','valid_utc_year')+tuple(
        f'valid_cont_{j}' for j in range(2,15))+('valid_utc_is_weekend',)
    scalers=fit_train_only_scalers_v2(
        seq,aux,target,mask,continuous_positions=np.arange(15,dtype=np.int64),
        binary_positions=np.asarray([15],dtype=np.int64),auxiliary_columns=columns,
    )
    scaled=neural.apply_train_only_scalers(seq,aux,target,scalers)
    year=scaled['auxiliary'][:,1]
    np.testing.assert_allclose(year[:96], 0, atol=1e-6)
    np.testing.assert_allclose(year[96:], 1, atol=1e-6)
    assert scalers['n_train']==96
    assert scalers['auxiliary_mean'][1]==2018.0
    assert scalers['auxiliary_std'][1]==1.0
