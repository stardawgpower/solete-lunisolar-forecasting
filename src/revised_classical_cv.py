"""Availability-safe expanding-window validation for revised SOLETE classical models.

This module only evaluates development folds. It never performs terminal refitting,
never reads stored model results, and never writes artifacts. Model factories must
return a NEW unfitted estimator (including any fitted preprocessing pipeline).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def _aligned_boolean_mask(mask: pd.Series, index: pd.Index, name: str) -> np.ndarray:
    if not isinstance(mask, pd.Series) or not mask.index.equals(index):
        raise ValueError(f"{name} must be a Series aligned exactly to X.index.")
    if mask.isna().any() or not pd.api.types.is_bool_dtype(mask.dtype):
        raise ValueError(f"{name} must contain only nonmissing booleans.")
    return mask.to_numpy(dtype=bool)


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    issue_time: pd.Series,
    valid_time: pd.Series,
    horizon_minutes: int,
    folds: Sequence[Mapping],
    forbidden_mask: pd.Series | None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a nonempty DataFrame.")
    index = X.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("X.index must be unique and chronologically ordered.")
    if not X.columns.is_unique:
        raise ValueError("X.columns must be unique.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Every model feature must be numeric.")
    if not isinstance(y, pd.Series) or not y.index.equals(index):
        raise ValueError("y must be a Series aligned exactly to X.index.")
    if not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError("Targets must be numeric.")
    for name, times in (("issue_time", issue_time), ("valid_time", valid_time)):
        if not isinstance(times, pd.Series) or not times.index.equals(index):
            raise ValueError(f"{name} must align exactly to X.index.")
        if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC":
            raise ValueError(f"{name} must use UTC-aware datetimes.")
        if times.isna().any():
            raise ValueError(f"{name} contains NaT.")
    if not isinstance(horizon_minutes, int) or horizon_minutes not in (15, 30, 60):
        raise ValueError("Expected the frozen 15-, 30-, or 60-minute horizon.")
    if not issue_time.is_monotonic_increasing or not valid_time.is_monotonic_increasing:
        raise ValueError("Forecast issue/valid times must be sorted.")
    if not (valid_time == issue_time + pd.Timedelta(minutes=horizon_minutes)).all():
        raise ValueError(
            "valid_time must equal issue_time plus the exact forecast horizon."
        )
    if not folds:
        raise ValueError("At least one validation fold is required.")

    blocked = (
        np.zeros(len(X), dtype=bool)
        if forbidden_mask is None
        else _aligned_boolean_mask(forbidden_mask, index, "forbidden_mask")
    )
    validated = []
    seen_folds = set()
    for fold in folds:
        number = int(fold["fold"])
        if number in seen_folds:
            raise ValueError("Fold numbers must be unique.")
        seen_folds.add(number)
        train = _aligned_boolean_mask(fold["train_mask"], index, "train_mask")
        validation = _aligned_boolean_mask(
            fold["validation_mask"], index, "validation_mask"
        )
        if not train.any() or not validation.any():
            raise ValueError(f"Fold {number}: train and validation must be nonempty.")
        if (train & validation).any():
            raise ValueError(f"Fold {number}: train/validation overlap.")
        if ((train | validation) & blocked).any():
            raise ValueError(f"Fold {number}: forbidden holdout rows are in CV.")
        # Validate only development rows used in this fold. Do NOT inspect
        # held-out target values during hyperparameter selection.
        used = np.flatnonzero(train | validation)
        if not np.isfinite(X.iloc[used].to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {number}: model features must be finite.")
        if not np.isfinite(y.iloc[used].to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {number}: targets must be finite.")
        # The model is frozen at the FIRST validation ISSUE. Training labels
        # must already be available at that time, not merely before its label.
        if (
            valid_time.iloc[np.flatnonzero(train)[-1]]
            >= issue_time.iloc[np.flatnonzero(validation)[0]]
        ):
            raise ValueError(
                f"Fold {number}: training labels leak into validation issues."
            )
        validated.append((number, train, validation))
    return validated


def evaluate_cv_candidates(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    issue_time: pd.Series,
    valid_time: pd.Series,
    horizon_minutes: int,
    folds: Sequence[Mapping],
    configurations: Sequence[Mapping],
    model_factory: Callable[[Mapping], object],
    forbidden_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_fold_scores, ranked_configurations) without using a test set.

    Selection is by the UNWEIGHTED mean of fold MAEs (not pooled row MAE).
    Ties are resolved by original configuration order; RMSE is not a tiebreak.
    Preconditions for every fold are checked *before* any model is fitted.
    """
    checked_folds = _validate_inputs(
        X, y, issue_time, valid_time, horizon_minutes, folds, forbidden_mask
    )
    if not configurations:
        raise ValueError("The configuration grid must not be empty.")
    keys = [
        json.dumps(dict(c), sort_keys=True, allow_nan=False) for c in configurations
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate configurations must be unique.")
    records = []
    for config_index, configuration in enumerate(configurations):
        for fold_number, train, validation in checked_folds:
            estimator = model_factory(dict(configuration))
            X_train = X.iloc[np.flatnonzero(train)]
            y_train = y.iloc[np.flatnonzero(train)]
            X_validation = X.iloc[np.flatnonzero(validation)]
            y_validation = y.iloc[np.flatnonzero(validation)].to_numpy(dtype=float)
            estimator.fit(X_train, y_train)
            predictions = np.asarray(estimator.predict(X_validation), dtype=float)
            if (
                predictions.shape != y_validation.shape
                or not np.isfinite(predictions).all()
            ):
                raise RuntimeError("Estimator returned invalid validation predictions.")
            residuals = y_validation - predictions
            records.append(
                {
                    "config_index": config_index,
                    "configuration": keys[config_index],
                    "fold": fold_number,
                    "train_rows": len(X_train),
                    "validation_rows": len(y_validation),
                    "validation_mae": float(np.mean(np.abs(residuals))),
                    "validation_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                }
            )

    scores = pd.DataFrame.from_records(records)
    grouped = (
        scores.groupby("config_index", sort=True)
        .agg(
            cv_mae=("validation_mae", "mean"),
            cv_rmse=("validation_rmse", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
    )
    grouped["configuration"] = grouped["config_index"].map(dict(enumerate(keys)))
    grouped = grouped.sort_values(
        ["cv_mae", "config_index"], ascending=[True, True], kind="stable"
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return scores, grouped


def evaluate_solete_cv_candidates(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    issue_time: pd.Series,
    valid_time: pd.Series,
    horizon_minutes: int,
    configurations: Sequence[Mapping],
    model_factory: Callable[[Mapping], object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict SOLETE wrapper: exact frozen counts and holdout exclusion."""
    from src.forecasting_protocol import (
        make_rolling_origin_masks,
        terminal_holdout_mask,
        validate_rolling_origin_counts,
    )

    validate_rolling_origin_counts(valid_time, horizon_minutes)
    folds = make_rolling_origin_masks(valid_time)
    if len(folds) != 3:
        raise RuntimeError("SOLETE requires exactly three rolling-origin folds.")
    return evaluate_cv_candidates(
        X=X,
        y=y,
        issue_time=issue_time,
        valid_time=valid_time,
        horizon_minutes=horizon_minutes,
        folds=folds,
        configurations=configurations,
        model_factory=model_factory,
        forbidden_mask=terminal_holdout_mask(valid_time),
    )
