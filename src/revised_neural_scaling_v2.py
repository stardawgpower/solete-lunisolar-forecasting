"""Development-only V2 scaling policy for a constant `valid_utc_year` column.

The historical `src.neural_models` scaler is frozen and remains unchanged.
When training-only year has zero variance, center it on TRAIN and use scale=1;
all other statistics are computed solely from TRAIN with the frozen ddof.
Unknown zero-variance features still fail closed. This module never trains.
"""
from __future__ import annotations

import numpy as np

POLICY = 'constant-valid-utc-year-training-std-one-v2'


def fit_train_only_scalers_v2(sequence, auxiliary, target, train_mask, *,
                              continuous_positions, binary_positions,
                              auxiliary_columns) -> dict:
    from src.neural_models import (
        fit_train_only_scalers as legacy_fit,
        _validate_raw_neural_arrays,
        _validate_positions,
        SCALING_DDOF,
    )
    sequence, auxiliary, target = _validate_raw_neural_arrays(sequence, auxiliary, target)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if mask.shape != (sequence.shape[0],) or not mask.any():
        raise ValueError('Invalid train-only scaling mask')
    continuous_positions, binary_positions = _validate_positions(
        auxiliary.shape[1], continuous_positions, binary_positions,
    )
    if (not isinstance(auxiliary_columns, (tuple, list))
            or len(auxiliary_columns) != auxiliary.shape[1]
            or auxiliary_columns.count('valid_utc_year') != 1
            or auxiliary_columns[1] != 'valid_utc_year'):
        raise ValueError('Unexpected F1-F4 year-column contract')
    year_position = 1
    if year_position not in continuous_positions or year_position in binary_positions:
        raise ValueError('Calendar year must remain a continuous auxiliary feature')
    numeric_aux = auxiliary.astype(np.float64, copy=False)
    if binary_positions.size and not np.isin(numeric_aux[:, binary_positions], (0.0, 1.0)).all():
        raise ValueError('Binary auxiliary values must be zero or one')
    continuous_train = numeric_aux[mask][:, continuous_positions]
    stds = continuous_train.std(axis=0, ddof=SCALING_DDOF)
    if not np.isfinite(stds).all():
        raise ValueError('Invalid continuous auxiliary standard deviation')
    zero = continuous_positions[np.flatnonzero(stds == 0.0)]
    if len(zero) == 0:
        # V2 must have EXACT legacy behaviour on non-degenerate training data.
        return legacy_fit(sequence, auxiliary, target, mask,
                          continuous_positions=continuous_positions,
                          binary_positions=binary_positions)
    if not np.array_equal(zero, np.asarray([year_position])):
        raise ValueError('Unexpected constant continuous feature(s); only valid_utc_year is supported')
    year_train = numeric_aux[mask, year_position]
    if not np.isfinite(year_train).all() or not np.all(np.isin(year_train, (2018.0, 2019.0))):
        raise ValueError('Unexpected training calendar-year values')

    train_sequence = sequence[mask].astype(np.float64, copy=False)
    sequence_mean = train_sequence.mean(axis=(0, 1))
    sequence_std = train_sequence.std(axis=(0, 1), ddof=SCALING_DDOF)
    if not np.isfinite(sequence_std).all() or not np.all(sequence_std > 0.0):
        raise ValueError('Zero-variance or invalid sequence channel found')

    auxiliary_mean = continuous_train.mean(axis=0)
    auxiliary_std = stds.copy()
    year_cont_index = int(np.flatnonzero(continuous_positions == year_position)[0])
    auxiliary_std[year_cont_index] = 1.0
    train_target = target[mask].astype(np.float64, copy=False)
    target_mean = float(train_target.mean())
    target_std = float(train_target.std(ddof=SCALING_DDOF))
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError('Training target must have positive variance')
    return {
        'sequence_mean': sequence_mean.copy(), 'sequence_std': sequence_std.copy(),
        'auxiliary_mean': auxiliary_mean.copy(), 'auxiliary_std': auxiliary_std.copy(),
        'target_mean': target_mean, 'target_std': target_std,
        'continuous_positions': continuous_positions.copy(),
        'binary_positions': binary_positions.copy(), 'ddof': SCALING_DDOF,
        'n_train': int(mask.sum()),
    }
