"""Synthetic regression tests for the revised, availability-safe CV runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.revised_classical_cv import evaluate_cv_candidates


class ConstantEstimator:
    def __init__(self, value, log):
        self.value = value
        self.log = log

    def fit(self, X, y):
        self.log.append(("fit", self.value, tuple(X.index)))
        return self

    def predict(self, X):
        self.log.append(("predict", self.value, tuple(X.index)))
        return np.full(len(X), self.value)


def toy_inputs():
    index = pd.date_range("2018-06-01", periods=100, freq="5min", tz="UTC")
    X = pd.DataFrame({"known_at_issue": np.arange(100)}, index=index)
    y = pd.Series(np.zeros(100), index=index, dtype=float)
    issue = pd.Series(index, index=index)
    valid = issue + pd.Timedelta(minutes=15)
    folds = []
    for number, train_end, val_start, val_end in (
        (1, 19, 22, 24),
        (2, 41, 44, 64),
        (3, 71, 74, 86),
    ):
        folds.append(
            {
                "fold": number,
                "train_mask": pd.Series(np.arange(100) < train_end, index=index),
                "validation_mask": pd.Series(
                    (np.arange(100) >= val_start) & (np.arange(100) < val_end),
                    index=index,
                ),
            }
        )
    forbidden = pd.Series(np.arange(100) >= 90, index=index)
    return X, y, issue, valid, folds, forbidden


def invoke(X, y, issue, valid, folds, forbidden, log, configurations=None):
    return evaluate_cv_candidates(
        X=X,
        y=y,
        issue_time=issue,
        valid_time=valid,
        horizon_minutes=15,
        folds=folds,
        forbidden_mask=forbidden,
        configurations=configurations or ({"value": 0}, {"value": 10}),
        model_factory=lambda config: ConstantEstimator(config["value"], log),
    )


def test_three_fresh_models_per_candidate_and_no_holdout():
    data = toy_inputs()
    log = []
    per_fold, selected = invoke(*data, log)
    assert len(per_fold) == 6
    assert len(selected) == 2
    assert per_fold["fold"].tolist() == [1, 2, 3, 1, 2, 3]
    assert len([x for x in log if x[0] == "fit"]) == 6
    assert len([x for x in log if x[0] == "predict"]) == 6
    assert all(all(t < data[0].index[90] for t in record[2]) for record in log)
    assert selected.iloc[0]["config_index"] == 0


def test_mean_of_fold_maes_not_pooled_validation_mae():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    y.iloc[22:24] = 0.0
    y.iloc[44:64] = 10.0
    y.iloc[74:86] = 5.0
    scores, ranking = invoke(X, y, issue, valid, folds, forbidden, [])
    # Candidate 0 has fold MAEs [0,10,5], candidate 10 [10,0,5].
    assert scores.loc[scores.config_index.eq(0), "validation_mae"].tolist() == [
        0,
        10,
        5,
    ]
    assert scores.loc[scores.config_index.eq(1), "validation_mae"].tolist() == [
        10,
        0,
        5,
    ]
    assert ranking["cv_mae"].tolist() == [5, 5]
    assert ranking.iloc[0]["config_index"] == 0  # deterministic tie, not pooled rows


def test_rejects_availability_leakage_before_fitting():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    bad = [dict(fold) for fold in folds]
    bad[0]["train_mask"] = pd.Series(np.arange(100) < 20, index=X.index)
    log = []
    with pytest.raises(ValueError, match="leak"):
        invoke(X, y, issue, valid, bad, forbidden, log)
    assert not log


def test_rejects_holdout_access_before_fitting():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    bad = [dict(fold) for fold in folds]
    bad[2]["validation_mask"] = pd.Series(
        (np.arange(100) >= 91) & (np.arange(100) < 95), index=X.index
    )
    log = []
    with pytest.raises(ValueError, match="holdout"):
        invoke(X, y, issue, valid, bad, forbidden, log)
    assert not log


def test_rejects_misalignment_and_wrong_horizon():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    with pytest.raises(ValueError, match="aligned"):
        invoke(X, y.iloc[::-1], issue, valid, folds, forbidden, [])
    with pytest.raises(ValueError, match="horizon"):
        invoke(X, y, issue, valid + pd.Timedelta(minutes=5), folds, forbidden, [])


def test_rejects_duplicate_candidates():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    with pytest.raises(ValueError, match="unique"):
        invoke(X, y, issue, valid, folds, forbidden, [], ({"value": 0}, {"value": 0}))


def test_rejects_nonfinite_features_before_fitting():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    X.loc[X.index[0], "known_at_issue"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        invoke(X, y, issue, valid, folds, forbidden, [])


def test_does_not_inspect_terminal_target_values():
    X, y, issue, valid, folds, forbidden = toy_inputs()
    # These timestamps are outside every CV fold. Their target is not
    # needed for selection, even to check finiteness.
    y.iloc[90:] = np.nan
    scores, ranking = invoke(X, y, issue, valid, folds, forbidden, [])
    assert len(scores) == 6
    assert len(ranking) == 2
