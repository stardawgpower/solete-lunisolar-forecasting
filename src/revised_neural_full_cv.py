"""Milestone 3: single full-size nested SOLETE CV gate per LSTM/TCN.

This module imports no TensorFlow, fits no models and writes no files on import.
Only 15-minute/F0/outer fold 1/candidate C1 is enabled. Inner validation
chooses an epoch, a FRESH model is then refitted on all outer TRAIN rows,
and only that refitted model predicts OUTER validation rows. The terminal
holdout never enters a payload. No architecture ranking/final test occurs.

Records are atomic per *completed fold*, not per epoch: interrupting an
unfinished fit may require repeating it. A lock refuses concurrent runs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.revised_neural_training import (
    EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE,
    INNER_LABEL_GAP_MINUTES, INNER_VALIDATION_FRACTION,
    LSTM_CANDIDATES, MAX_SELECTION_EPOCHS, TCN_CANDIDATES,
    _masks_and_times, make_inner_split,
)

GATE_SCHEMA_VERSION = 1
GATE_HORIZON = 15
GATE_FAMILY = 'F0'
GATE_FOLD = 1
GATE_TRAIN_ROWS = 52401
GATE_VALIDATION_ROWS = 17856
GATE_CONFIG_INDEX = 0
GATE_BATCH_SIZE = 256
GATE_SEED = 42
GATE_KIND = 'full-size-single-fold-neural-cv-gate-not-ranked'
GATE_MODELS = ('LSTM', 'TCN')


def choose_best_epoch(validation_losses: list[float]) -> tuple[int, float]:
    """Pick the earliest achieved minimum; no outer-validation data used."""
    values = np.asarray(validation_losses, dtype=np.float64)
    if values.ndim != 1 or not 1 <= len(values) <= MAX_SELECTION_EPOCHS:
        raise ValueError('Invalid inner-validation history length')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative inner-validation loss')
    return int(np.argmin(values)) + 1, float(values.min())


def validate_fullsize_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Demand the exact already-masked canonical fold-1 payload and its gap."""
    if (payload.get('horizon_minutes'), payload.get('family'), payload.get('fold')) != (
        GATE_HORIZON, GATE_FAMILY, GATE_FOLD
    ):
        raise ValueError('Full-size gate is restricted to 15-minute F0 fold 1')
    train, validation, issue, valid = _masks_and_times(payload)
    if (int(train.sum()), int(validation.sum())) != (GATE_TRAIN_ROWS, GATE_VALIDATION_ROWS):
        raise ValueError('Full-size gate must use all frozen fold-1 rows')
    if payload.get('train_rows') != GATE_TRAIN_ROWS or payload.get('validation_rows') != GATE_VALIDATION_ROWS:
        raise ValueError('Fold population metadata mismatch')
    if np.asarray(payload['auxiliary']).shape[1] != 1:
        raise ValueError('F0 must have exactly one auxiliary feature')
    if not np.array_equal(np.asarray(payload['continuous_positions']), [0]):
        raise ValueError('F0 continuous auxiliary positions changed')
    if np.asarray(payload['binary_positions']).size:
        raise ValueError('F0 must have no binary auxiliary features')
    for key in ('sequence', 'auxiliary', 'target_kw'):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError(f'Nonfinite {key}')
    if not valid.is_monotonic_increasing:
        raise ValueError('Valid times must be ordered')
    inner = make_inner_split(payload)
    if inner['embargo_rows'] <= 0:
        raise ValueError('Missing inner availability embargo')
    if np.any((inner['inner_train_mask'] | inner['inner_validation_mask']) & validation):
        raise ValueError('Inner split accesses outer validation')
    inner_indices = np.flatnonzero(inner['inner_train_mask'])
    stop_indices = np.flatnonzero(inner['inner_validation_mask'])
    if valid.iloc[inner_indices[-1]] >= issue.iloc[stop_indices[0]]:
        raise ValueError('Inner fit labels unavailable by early stopping start')
    return np.flatnonzero(train), np.flatnonzero(validation), inner


def _hash_payload(payload: dict) -> str:
    """Hash exact in-scope values, ordered times, masks, and auxiliary schema."""
    h = hashlib.sha256()
    for key in ('sequence', 'auxiliary', 'target_kw', 'train_mask',
                'validation_mask', 'continuous_positions', 'binary_positions'):
        arr = np.ascontiguousarray(payload[key])
        h.update(key.encode() + b'\0')
        h.update(str((arr.shape, arr.dtype.str)).encode('ascii') + b'\0')
        h.update(arr.tobytes())
    for key in ('issue_time', 'valid_time'):
        series = payload[key]
        assert isinstance(series, pd.Series)
        h.update(key.encode() + b'\0')
        h.update(pd.util.hash_pandas_object(series, index=True).to_numpy(dtype='uint64').tobytes())
    return h.hexdigest()


def gate_source_fingerprint(root: Path) -> str:
    paths = (
        'src/forecasting_protocol.py', 'src/neural_models.py',
        'src/revised_neural_cv.py', 'src/revised_neural_training.py',
        'src/revised_neural_full_cv.py',
    )
    h = hashlib.sha256()
    for name in paths:
        h.update(name.encode('utf-8') + b'\0')
        h.update((Path(root) / name).read_bytes())
    return h.hexdigest()


def _validate_record(record: dict, identity: dict) -> None:
    fields = {'inner_training_rows', 'inner_validation_rows', 'inner_embargo_rows',
              'outer_training_rows', 'outer_validation_rows', 'inner_epochs_completed',
              'best_epoch', 'best_inner_validation_loss_scaled', 'trainable_parameters',
              'inner_selection_seconds', 'outer_refit_seconds',
              'outer_prediction_seconds', 'validation_mae_kw', 'validation_rmse_kw'}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError('Malformed completed gate record')
    for key in fields:
        if isinstance(record[key], bool) or not isinstance(record[key], (int, float)):
            raise ValueError(f'Invalid record field {key}')
        if not np.isfinite(record[key]):
            raise ValueError(f'Nonfinite record field {key}')
    ints = {'inner_training_rows', 'inner_validation_rows', 'inner_embargo_rows',
            'outer_training_rows', 'outer_validation_rows', 'inner_epochs_completed',
            'best_epoch', 'trainable_parameters'}
    if any(type(record[key]) is not int or record[key] <= 0 for key in ints):
        raise ValueError('Invalid record row/epoch/parameter counts')
    if (record['outer_training_rows'], record['outer_validation_rows']) != (GATE_TRAIN_ROWS, GATE_VALIDATION_ROWS):
        raise ValueError('Gate record has wrong frozen fold counts')
    if not 1 <= record['best_epoch'] <= record['inner_epochs_completed'] <= MAX_SELECTION_EPOCHS:
        raise ValueError('Invalid epoch accounting')
    if any(record[key] < 0 for key in (
        'best_inner_validation_loss_scaled', 'validation_mae_kw', 'validation_rmse_kw'
    )):
        raise ValueError('Invalid loss or metrics')
    if any(record[key] <= 0 for key in (
        'inner_selection_seconds', 'outer_refit_seconds', 'outer_prediction_seconds'
    )):
        raise ValueError('Invalid runtime')
    if record['inner_training_rows'] + record['inner_validation_rows'] + record['inner_embargo_rows'] != record['outer_training_rows']:
        raise ValueError('Inner population accounting mismatch')
    if identity['model'] not in GATE_MODELS:
        raise ValueError('Invalid checkpoint model')


def _read_checkpoint(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('schema_version') != GATE_SCHEMA_VERSION or state.get('kind') != GATE_KIND or state.get('identity') != identity:
        raise ValueError(f'Full-size gate checkpoint identity mismatch: {path}')
    record = state.get('record')
    _validate_record(record, identity)
    return record


def _build_model(model_name: str, config: dict, seed: int):
    from src.neural_models import (
        build_recurrent_fusion_model, build_tcn_fusion_model, reset_neural_state
    )
    reset_neural_state(seed)
    if model_name == 'LSTM':
        return build_recurrent_fusion_model('LSTM', 1, config, seed=seed)
    return build_tcn_fusion_model(config, 1)


def _fit_predict_one(*, payload: dict, model_name: str, config: dict,
                     train_idx: np.ndarray, val_idx: np.ndarray,
                     inner: dict) -> dict:
    """Training code isolated from checkpoint handling for synthetic testing."""
    import tensorflow as tf
    from src.neural_models import (
        fit_train_only_scalers, apply_train_only_scalers, inverse_target_scaling
    )
    sequence = payload['sequence']
    aux = payload['auxiliary']
    targets = payload['target_kw']
    inner_fit = np.flatnonzero(inner['inner_train_mask'])
    inner_stop = np.flatnonzero(inner['inner_validation_mask'])

    # INNER stage: neither outer-validation features nor labels enter any scaler,
    # model fit, early-stopping callback, or epoch selection.
    inner_rows = np.concatenate((inner_fit, inner_stop))
    local_train = np.arange(len(inner_rows)) < len(inner_fit)
    inner_scaler = fit_train_only_scalers(
        sequence[inner_rows], aux[inner_rows], targets[inner_rows], local_train,
        continuous_positions=payload['continuous_positions'],
        binary_positions=payload['binary_positions'],
    )
    inner_scaled = apply_train_only_scalers(
        sequence[inner_rows], aux[inner_rows], targets[inner_rows], inner_scaler
    )
    first = slice(0, len(inner_fit))
    second = slice(len(inner_fit), len(inner_rows))
    model = _build_model(model_name, config, GATE_SEED)
    count = int(model.count_params())
    started = perf_counter()
    try:
        history = model.fit(
            [inner_scaled['sequence'][first], inner_scaled['auxiliary'][first]],
            inner_scaled['target_scaled'][first],
            validation_data=(
                [inner_scaled['sequence'][second], inner_scaled['auxiliary'][second]],
                inner_scaled['target_scaled'][second],
            ),
            epochs=MAX_SELECTION_EPOCHS, batch_size=GATE_BATCH_SIZE,
            shuffle=True, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', min_delta=EARLY_STOPPING_MIN_DELTA,
                patience=EARLY_STOPPING_PATIENCE, restore_best_weights=False,
            )],
        )
        selection_seconds = perf_counter() - started
        losses = history.history.get('val_loss')
        epoch, loss = choose_best_epoch(losses)
        completed = len(losses)
    finally:
        tf.keras.backend.clear_session()
        del model
        del inner_scaled, inner_scaler

    # OUTER refit: recompute scaling on ALL outer-training rows, not the inner
    # training subset. Do not look at outer-validation LABELS until prediction.
    outer_mask = np.ones(len(train_idx), dtype=bool)
    outer_scaler = fit_train_only_scalers(
        sequence[train_idx], aux[train_idx], targets[train_idx], outer_mask,
        continuous_positions=payload['continuous_positions'],
        binary_positions=payload['binary_positions'],
    )
    outer_scaled = apply_train_only_scalers(
        sequence[train_idx], aux[train_idx], targets[train_idx], outer_scaler
    )
    # Dummy target for prediction-side transformation; the true outer target is
    # accessed only AFTER the fresh model has completed its refit and predicted.
    predict_scaled = apply_train_only_scalers(
        sequence[val_idx], aux[val_idx], np.zeros(len(val_idx), dtype=np.float32),
        outer_scaler,
    )
    model = _build_model(model_name, config, GATE_SEED)
    if int(model.count_params()) != count:
        raise RuntimeError('Inner and refit architectures differ')
    started = perf_counter()
    try:
        fit_history = model.fit(
            [outer_scaled['sequence'], outer_scaled['auxiliary']],
            outer_scaled['target_scaled'],
            epochs=epoch, batch_size=GATE_BATCH_SIZE, shuffle=True, verbose=2,
        )
        refit_seconds = perf_counter() - started
        if len(fit_history.history.get('loss', [])) != epoch:
            raise RuntimeError('Outer refit did not complete selected epoch count')
        started = perf_counter()
        estimated_scaled = np.asarray(model.predict(
            [predict_scaled['sequence'], predict_scaled['auxiliary']],
            batch_size=GATE_BATCH_SIZE, verbose=0,
        )).reshape(-1)
        prediction_seconds = perf_counter() - started
        if estimated_scaled.shape != (len(val_idx),) or not np.isfinite(estimated_scaled).all():
            raise RuntimeError('Invalid outer-fold predictions')
        estimates_kw = np.asarray(inverse_target_scaling(estimated_scaled, outer_scaler)).reshape(-1)
        true_kw = np.asarray(targets[val_idx], dtype=np.float64)
        delta = true_kw - estimates_kw
        if not np.isfinite(delta).all():
            raise RuntimeError('Nonfinite validation residuals')
        mae = float(np.abs(delta).mean(dtype=np.float64))
        rmse = float(np.sqrt(np.square(delta).mean(dtype=np.float64)))
    finally:
        tf.keras.backend.clear_session()
        del model

    return {
        'inner_training_rows': int(len(inner_fit)),
        'inner_validation_rows': int(len(inner_stop)),
        'inner_embargo_rows': int(inner['embargo_rows']),
        'outer_training_rows': int(len(train_idx)),
        'outer_validation_rows': int(len(val_idx)),
        'inner_epochs_completed': int(completed),
        'best_epoch': int(epoch),
        'best_inner_validation_loss_scaled': float(loss),
        'trainable_parameters': count,
        'inner_selection_seconds': float(selection_seconds),
        'outer_refit_seconds': float(refit_seconds),
        'outer_prediction_seconds': float(prediction_seconds),
        'validation_mae_kw': mae,
        'validation_rmse_kw': rmse,
    }


def run_full_size_gate(*, payload: dict, root: Path, model_name: str,
                       source_sha256: str, output_dir: Path) -> tuple[dict, bool]:
    """Compute/reuse one full-size fold; False on already-matching checkpoint.

    Strict scope: exactly two calls, one for each C1 architecture. If a run
    stops mid-epoch, no partial score is saved and the next run repeats it.
    """
    if model_name not in GATE_MODELS:
        raise ValueError('Only LSTM and TCN full-size resource gates are permitted')
    train_idx, val_idx, inner = validate_fullsize_payload(payload)
    expected_sha = gate_source_fingerprint(root)
    if not isinstance(source_sha256, str) or source_sha256 != expected_sha:
        raise ValueError('Full-size source fingerprint mismatch')
    import tensorflow as tf
    config = dict((LSTM_CANDIDATES if model_name == 'LSTM' else TCN_CANDIDATES)[GATE_CONFIG_INDEX])
    # json round trip turns tuple dilations into stable JSON lists.
    config = json.loads(json.dumps(config, sort_keys=True))
    identity = {
        'model': model_name, 'configuration': config, 'candidate_index': GATE_CONFIG_INDEX,
        'horizon_minutes': GATE_HORIZON, 'feature_family': GATE_FAMILY,
        'outer_fold': GATE_FOLD, 'outer_train_rows': GATE_TRAIN_ROWS,
        'outer_validation_rows': GATE_VALIDATION_ROWS,
        'inner_fraction': INNER_VALIDATION_FRACTION, 'inner_gap_minutes': INNER_LABEL_GAP_MINUTES,
        'max_epochs': MAX_SELECTION_EPOCHS, 'patience': EARLY_STOPPING_PATIENCE,
        'min_delta': EARLY_STOPPING_MIN_DELTA, 'batch_size': GATE_BATCH_SIZE,
        'seed': GATE_SEED, 'source_sha256': source_sha256,
        'data_sha256': _hash_payload(payload),
        'tensorflow_version': tf.__version__, 'numpy_version': np.__version__,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{model_name.lower()}_c1_h15_F0_fold1.json'
    lock = path.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Checkpoint is locked by another or interrupted run: {lock}') from exc
    os.close(fd)
    try:
        prior = _read_checkpoint(path, identity)
        if prior is not None:
            return prior, False
        record = _fit_predict_one(payload=payload, model_name=model_name, config=config,
                                  train_idx=train_idx, val_idx=val_idx, inner=inner)
        _validate_record(record, identity)
        from src.revised_rf_checkpoint import atomic_text
        state = {'schema_version': GATE_SCHEMA_VERSION, 'kind': GATE_KIND,
                 'identity': identity, 'record': record}
        atomic_text(path, json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + '\n',
                    refuse_different=True)
        return record, True
    finally:
        lock.unlink(missing_ok=True)
