"""Development-only neural input packing for revised SOLETE rolling-origin CV.

No TensorFlow dependency; does not fit models or scalers and never consumes
terminal labels or terminal features. The established neural scaling functions
may be applied to its development-fold-only arrays after this audit passes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNEL_NAMES = (
    "P_Solar_kw", "GHI_kw1m2", "POA_Irr_kw1m2", "temperature_degC",
    "humidity_clean", "pressure_clean_mbar", "wind_speed_m1s",
    "wind_direction_sin", "wind_direction_cos",
)
SEQUENCE_LENGTH = 13
PV_DAILY_LAG_COLUMN = "P_Solar_kw_issue_minus_24h"
FAMILY_DIMENSIONS = {"F0": 118, "F1": 133, "F2": 138, "F3": 147, "F4": 244}
AUXILIARY_DIMENSIONS = {"F0": 1, "F1": 16, "F2": 21, "F3": 30, "F4": 127}
PANCHANG_PREFIXES = (
    "valid_tithi_", "valid_paksha_", "valid_karana_",
    "valid_nakshatra_", "valid_yoga_",
)
HISTORICAL_COLUMNS = tuple(
    f"{name}_lag_{lag}" for name in CHANNEL_NAMES for lag in range(13)
)
F0_COLUMNS = HISTORICAL_COLUMNS + (PV_DAILY_LAG_COLUMN,)


def _checked_mask(mask: pd.Series, index: pd.Index, name: str) -> np.ndarray:
    if not isinstance(mask, pd.Series) or not mask.index.equals(index):
        raise ValueError(f"{name} must be an index-aligned pandas Series")
    if mask.isna().any() or not pd.api.types.is_bool_dtype(mask.dtype):
        raise ValueError(f"{name} must contain only nonmissing booleans")
    return mask.to_numpy(dtype=bool)


def materialize_neural_inputs(X: pd.DataFrame, family: str) -> dict:
    """Repackage exactly the canonical flat F0–F4 predictors, without labels.

    The sequence is ordered oldest-to-newest (t-60 to issue time t).
    Future-known deterministic features are auxiliary, NOT sequence channels.
    """
    if family not in FAMILY_DIMENSIONS:
        raise ValueError(f"Unknown SOLETE feature family: {family}")
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a nonempty DataFrame")
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("X index must be unique and chronological")
    if X.shape[1] != FAMILY_DIMENSIONS[family]:
        raise ValueError(f"{family} requires {FAMILY_DIMENSIONS[family]} columns")
    if tuple(X.columns[:118]) != F0_COLUMNS or not X.columns.is_unique:
        raise ValueError("F0 column names and ordering differ from canonical Notebook 10")
    aux_cols = (PV_DAILY_LAG_COLUMN,) + tuple(X.columns[118:])
    if len(aux_cols) != AUXILIARY_DIMENSIONS[family]:
        raise ValueError("Incorrect neural auxiliary dimension")
    if any(not col.startswith("valid_") for col in aux_cols[1:]):
        raise ValueError("Additional auxiliary columns must be valid-time deterministic features")
    numeric = X.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Neural input contains nonfinite features")

    # Source F0 has channel-major lag order; neural branch is time-major.
    # No forward-looking observed input: lags 12 ... 0 only.
    sequence = np.stack(
        [X.loc[:, [f"{name}_lag_{lag}" for name in CHANNEL_NAMES]]
         .to_numpy(dtype=np.float32)
         for lag in range(12, -1, -1)],
        axis=1,
    )
    auxiliary = X.loc[:, list(aux_cols)].to_numpy(dtype=np.float32)
    binary_positions = np.asarray(
        [i for i, col in enumerate(aux_cols)
         if col in {"valid_utc_is_weekend", "valid_solar_daylight"}
         or col.startswith(PANCHANG_PREFIXES)],
        dtype=np.int64,
    )
    continuous_positions = np.asarray(
        [i for i in range(len(aux_cols)) if i not in set(binary_positions)],
        dtype=np.int64,
    )
    if binary_positions.size and not np.isin(auxiliary[:, binary_positions], (0., 1.)).all():
        raise ValueError("Frozen binary deterministic features are not literal 0/1")
    if sequence.shape != (len(X), 13, 9) or auxiliary.shape != (len(X), len(aux_cols)):
        raise RuntimeError("Neural tensor shape mismatch")
    return {
        "sequence": np.ascontiguousarray(sequence),
        "auxiliary": np.ascontiguousarray(auxiliary),
        "auxiliary_columns": aux_cols,
        "continuous_positions": continuous_positions,
        "binary_positions": binary_positions,
    }


def prepare_development_fold(
    *, X: pd.DataFrame, masked_target: pd.Series,
    issue_time: pd.Series, valid_time: pd.Series,
    forbidden_mask: pd.Series, fold: dict,
    horizon_minutes: int, family: str,
) -> dict:
    """Materialize ONLY training+validation arrays after strict gap/holdout guards.

    Caller must first mask terminal target labels. This function never reads
    terminal predictor rows or terminal target values and never fits a scaler.
    """
    if horizon_minutes not in (15, 30, 60):
        raise ValueError("Unsupported horizon")
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a nonempty DataFrame")
    index = X.index
    for name, series in (("masked_target", masked_target),
                         ("issue_time", issue_time), ("valid_time", valid_time)):
        if not isinstance(series, pd.Series) or not series.index.equals(index):
            raise ValueError(f"{name} must align with X")
    if not isinstance(issue_time.dtype, pd.DatetimeTZDtype) or str(issue_time.dt.tz) != "UTC":
        raise ValueError("issue_time must be timezone-aware UTC")
    if not isinstance(valid_time.dtype, pd.DatetimeTZDtype) or str(valid_time.dt.tz) != "UTC":
        raise ValueError("valid_time must be timezone-aware UTC")
    if issue_time.isna().any() or valid_time.isna().any():
        raise ValueError("Null forecast timestamps")
    if not issue_time.is_monotonic_increasing or not valid_time.is_monotonic_increasing:
        raise ValueError("Forecast timestamps must be chronological")
    if not (valid_time == issue_time + pd.Timedelta(minutes=horizon_minutes)).all():
        raise ValueError("Incorrect issue/valid-time alignment")
    blocked = _checked_mask(forbidden_mask, index, "forbidden_mask")
    train = _checked_mask(fold["train_mask"], index, "train_mask")
    val = _checked_mask(fold["validation_mask"], index, "validation_mask")
    if not blocked.any() or not train.any() or not val.any():
        raise ValueError("Terminal, train, and validation masks must be nonempty")
    if (train & val).any() or ((train | val) & blocked).any():
        raise ValueError("Train/validation overlap or forbidden holdout access")
    if not masked_target.iloc[np.flatnonzero(blocked)].isna().all():
        raise ValueError("Terminal target must be masked before fold preparation")
    train_idx = np.flatnonzero(train)
    val_idx = np.flatnonzero(val)
    if train_idx[-1] >= val_idx[0]:
        raise ValueError("Training must precede validation chronologically")
    if valid_time.iloc[train_idx[-1]] >= issue_time.iloc[val_idx[0]]:
        raise ValueError("Training label not available by first validation issue")
    first_terminal_idx = np.flatnonzero(blocked)[0]
    if valid_time.iloc[val_idx[-1]] >= issue_time.iloc[first_terminal_idx]:
        raise ValueError("Validation label not available by first terminal issue")
    used = train | val
    if masked_target.iloc[np.flatnonzero(used)].isna().any():
        raise ValueError("Missing development targets")
    # Only after guardrails pass can model-ready arrays be constructed.
    selected_X = X.iloc[np.flatnonzero(used)]
    payload = materialize_neural_inputs(selected_X, family=family)
    target = masked_target.iloc[np.flatnonzero(used)].to_numpy(dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite development labels")
    payload.update({
        "target_kw": np.ascontiguousarray(target),
        "train_mask": np.ascontiguousarray(train[used]),
        "validation_mask": np.ascontiguousarray(val[used]),
        "issue_time": issue_time.iloc[np.flatnonzero(used)].copy(),
        "valid_time": valid_time.iloc[np.flatnonzero(used)].copy(),
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "fold": int(fold["fold"]),
        "horizon_minutes": horizon_minutes,
        "family": family,
    })
    return payload
