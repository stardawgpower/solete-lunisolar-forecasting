"""Milestone 4: one-at-a-time, development-only nested neural CV.

This module never trains, imports TensorFlow, or writes on import. Every
completed (architecture, horizon, feature family, candidate, fold) is a
separate atomic JSON checkpoint. A failed or interrupted fit is NOT a score.
Existing Milestone-3 feasibility gates remain separate from campaign scores.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.forecasting_protocol import CV_FOLDS, EXPECTED_CV_COUNTS_BY_HORIZON, TEST_START
from src.revised_neural_cv import AUXILIARY_DIMENSIONS
from src.revised_neural_training import (
    INNER_VALIDATION_FRACTION, INNER_LABEL_GAP_MINUTES,
    MAX_SELECTION_EPOCHS, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE,
    LSTM_CANDIDATES, TCN_CANDIDATES, _masks_and_times,
    make_inner_split, rank_neural_cv_scores,
)
from src.revised_neural_full_cv import choose_best_epoch, _hash_payload

SCHEMA_VERSION = 1
KIND = 'complete-candidate-single-fold-development-cv'
BATCH_SIZE = 256
SEED = 42
MODELS = ('LSTM', 'TCN')
FAMILIES = ('F0', 'F1', 'F2', 'F3', 'F4')
HORIZONS = (15, 30, 60)


def grid(model_name: str) -> tuple[dict, ...]:
    if model_name == 'LSTM':
        return LSTM_CANDIDATES
    if model_name == 'TCN':
        return TCN_CANDIDATES
    raise ValueError('Only LSTM and TCN revised candidates are supported')


def validate_campaign_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact SOLETE population and fold boundaries; holdout absent by design."""
    horizon, family, fold = (payload.get('horizon_minutes'), payload.get('family'), payload.get('fold'))
    if type(horizon) is not int or horizon not in HORIZONS or family not in FAMILIES or type(fold) is not int or fold not in (1, 2, 3):
        raise ValueError('Unsupported canonical horizon/family/fold')
    train, validation, issue, valid = _masks_and_times(payload)
    expected = EXPECTED_CV_COUNTS_BY_HORIZON[horizon][fold - 1]
    if (int(train.sum()), int(validation.sum())) != tuple(expected):
        raise ValueError('Incorrect frozen training/validation row counts')
    if (payload.get('train_rows'), payload.get('validation_rows')) != tuple(expected):
        raise ValueError('Incorrect payload row metadata')
    seq = np.asarray(payload['sequence'])
    aux = np.asarray(payload['auxiliary'])
    target = np.asarray(payload['target_kw'])
    if seq.shape != (sum(expected), 13, 9) or aux.shape != (sum(expected), AUXILIARY_DIMENSIONS[family]):
        raise ValueError('Invalid canonical sequence/auxiliary dimensions')
    if not all(np.isfinite(arr).all() for arr in (seq, aux, target)):
        raise ValueError('Nonfinite development-only inputs or targets')
    continuous = np.asarray(payload['continuous_positions'])
    binary = np.asarray(payload['binary_positions'])
    if continuous.ndim != 1 or binary.ndim != 1 or continuous.dtype.kind not in 'iu' or binary.dtype.kind not in 'iu':
        raise ValueError('Invalid auxiliary position arrays')
    if not np.array_equal(np.sort(np.concatenate([continuous, binary])), np.arange(aux.shape[1])):
        raise ValueError('Auxiliary positions must exactly partition columns')
    if len(binary) != (0 if family == 'F0' else (1 if family == 'F1' else (2 if family in ('F2', 'F3') else 99))):
        raise ValueError('Unexpected binary auxiliary feature count')
    if binary.size and not np.isin(aux[:, binary], (0., 1.)).all():
        raise ValueError('Nonbinary auxiliary indicator')
    spec = CV_FOLDS[fold - 1]
    train_valid = valid.loc[train]
    val_valid = valid.loc[validation]
    if train_valid.iloc[0] < spec['train_start'] or val_valid.iloc[0] != spec['validation_start']:
        raise ValueError('Fold has incorrect chronological boundaries')
    boundary = spec['validation_end'] if fold < 3 else TEST_START - pd.Timedelta(minutes=60)
    if val_valid.iloc[-1] >= boundary:
        raise ValueError('Validation labels cross the protected fold boundary')
    if valid.loc[train].iloc[-1] >= issue.loc[validation].iloc[0]:
        raise ValueError('Unobserved training label reaches outer validation issue')
    # Prevent a fake re-indexed payload that simply has the same row counts.
    for mask in (train, validation):
        dt = valid.loc[mask].diff().dropna()
        if not dt.eq(pd.Timedelta(minutes=5)).all():
            raise ValueError('Noncanonical five-minute timestamp grid')
    inner = make_inner_split(payload)
    if inner['embargo_rows'] != 12 or int(inner['inner_train_mask'].sum()) < 32:
        raise ValueError('Incorrect availability-safe inner embargo')
    return np.flatnonzero(train), np.flatnonzero(validation), inner


def campaign_source_sha256(root: Path) -> str:
    names = (
        'src/forecasting_protocol.py', 'src/neural_models.py',
        'src/revised_neural_cv.py', 'src/revised_neural_training.py',
        'src/revised_neural_full_cv.py', 'src/revised_neural_campaign.py',
    )
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode() + b'\0')
        h.update((Path(root) / name).read_bytes())
    return h.hexdigest()


def _make_model(model_name: str, config: dict, aux_dim: int):
    from src.neural_models import (
        build_recurrent_fusion_model, build_tcn_fusion_model, reset_neural_state,
    )
    reset_neural_state(SEED)
    if model_name == 'LSTM':
        return build_recurrent_fusion_model('LSTM', aux_dim, config, seed=SEED)
    return build_tcn_fusion_model(config, aux_dim)


def _train_one(*, payload: dict, model_name: str, configuration: dict,
               train_idx: np.ndarray, val_idx: np.ndarray, inner: dict) -> dict:
    """Inner epoch selection, NEW outer-only scaler/model/refit, then outer score."""
    import tensorflow as tf
    from src.neural_models import (
        fit_train_only_scalers, apply_train_only_scalers, inverse_target_scaling,
    )
    seq, aux, target = (payload[k] for k in ('sequence', 'auxiliary', 'target_kw'))
    positions = dict(continuous_positions=payload['continuous_positions'],
                     binary_positions=payload['binary_positions'])
    fit_idx = np.flatnonzero(inner['inner_train_mask'])
    stop_idx = np.flatnonzero(inner['inner_validation_mask'])
    inner_idx = np.concatenate((fit_idx, stop_idx))
    inner_train_mask = np.arange(len(inner_idx)) < len(fit_idx)
    scalers = fit_train_only_scalers(seq[inner_idx], aux[inner_idx], target[inner_idx],
                                    inner_train_mask, **positions)
    data = apply_train_only_scalers(seq[inner_idx], aux[inner_idx], target[inner_idx], scalers)
    boundary = len(fit_idx)
    model = _make_model(model_name, configuration, aux.shape[1])
    count = int(model.count_params())
    start = perf_counter()
    try:
        history = model.fit(
            [data['sequence'][:boundary], data['auxiliary'][:boundary]],
            data['target_scaled'][:boundary],
            validation_data=(
                [data['sequence'][boundary:], data['auxiliary'][boundary:]],
                data['target_scaled'][boundary:],
            ),
            epochs=MAX_SELECTION_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA, restore_best_weights=False,
            )],
        )
        selection_seconds = perf_counter() - start
        losses = history.history.get('val_loss')
        best_epoch, best_loss = choose_best_epoch(losses)
        completed = len(losses)
    finally:
        tf.keras.backend.clear_session()
        del model, data, scalers

    # The outer-validation TARGETS are never passed into either scaler/fit.
    outer_mask = np.ones(len(train_idx), dtype=bool)
    outer_scalers = fit_train_only_scalers(seq[train_idx], aux[train_idx],
                                           target[train_idx], outer_mask, **positions)
    outer_data = apply_train_only_scalers(seq[train_idx], aux[train_idx],
                                          target[train_idx], outer_scalers)
    prediction_data = apply_train_only_scalers(
        seq[val_idx], aux[val_idx], np.zeros(len(val_idx), dtype=np.float32), outer_scalers,
    )
    model = _make_model(model_name, configuration, aux.shape[1])
    if int(model.count_params()) != count:
        raise RuntimeError('Inner and refit models have different parameter counts')
    start = perf_counter()
    try:
        fit_history = model.fit(
            [outer_data['sequence'], outer_data['auxiliary']], outer_data['target_scaled'],
            epochs=best_epoch, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
        )
        refit_seconds = perf_counter() - start
        if len(fit_history.history.get('loss', [])) != best_epoch:
            raise RuntimeError('Outer refit did not run exactly the selected epochs')
        start = perf_counter()
        prediction_scaled = np.asarray(model.predict(
            [prediction_data['sequence'], prediction_data['auxiliary']],
            batch_size=BATCH_SIZE, verbose=0,
        )).reshape(-1)
        prediction_seconds = perf_counter() - start
        if prediction_scaled.shape != (len(val_idx),) or not np.isfinite(prediction_scaled).all():
            raise RuntimeError('Invalid outer-fold predictions')
        predicted = np.asarray(inverse_target_scaling(prediction_scaled, outer_scalers), dtype=float)
        delta = np.asarray(target[val_idx], dtype=float) - predicted
        if not np.isfinite(delta).all():
            raise RuntimeError('Invalid outer-fold residuals')
        mae = float(np.mean(np.abs(delta), dtype=np.float64))
        rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
    finally:
        tf.keras.backend.clear_session()
        del model
    return {
        'inner_training_rows': int(len(fit_idx)),
        'inner_validation_rows': int(len(stop_idx)),
        'inner_embargo_rows': int(inner['embargo_rows']),
        'outer_training_rows': int(len(train_idx)),
        'outer_validation_rows': int(len(val_idx)),
        'inner_epochs_completed': int(completed),
        'best_epoch': int(best_epoch),
        'best_inner_validation_loss_scaled': float(best_loss),
        'trainable_parameters': count,
        'inner_selection_seconds': float(selection_seconds),
        'outer_refit_seconds': float(refit_seconds),
        'outer_prediction_seconds': float(prediction_seconds),
        'validation_mae_kw': mae,
        'validation_rmse_kw': rmse,
    }


def checkpoint_name(model_name: str, horizon: int, family: str, config_index: int, fold: int) -> str:
    if model_name not in MODELS or horizon not in HORIZONS or family not in FAMILIES or type(config_index) is not int or not 0 <= config_index < 4 or fold not in (1, 2, 3):
        raise ValueError('Invalid frozen CV checkpoint key')
    return f'{model_name.lower()}_c{config_index+1}_h{horizon}_{family}_fold{fold}.json'


def _check_record(record: dict, identity: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError('Malformed CV checkpoint record')
    # Reuse the strict Milestone-3 metric-field validation by temporary
    # canonical gate count comparison? No: count validation is fold-specific.
    required = {'inner_training_rows','inner_validation_rows','inner_embargo_rows',
                'outer_training_rows','outer_validation_rows','inner_epochs_completed',
                'best_epoch','best_inner_validation_loss_scaled','trainable_parameters',
                'inner_selection_seconds','outer_refit_seconds','outer_prediction_seconds',
                'validation_mae_kw','validation_rmse_kw'}
    if set(record) != required:
        raise ValueError('Invalid CV record fields')
    for name in required:
        val = record[name]
        if type(val) not in (int, float) or not np.isfinite(val):
            raise ValueError(f'Nonfinite or invalid CV field: {name}')
    integer = {'inner_training_rows','inner_validation_rows','inner_embargo_rows',
               'outer_training_rows','outer_validation_rows','inner_epochs_completed',
               'best_epoch','trainable_parameters'}
    if any(type(record[k]) is not int or record[k] <= 0 for k in integer):
        raise ValueError('Invalid integer CV metric')
    if (record['outer_training_rows'],record['outer_validation_rows']) != (
        identity['outer_train_rows'], identity['outer_validation_rows']
    ):
        raise ValueError('Incorrect CV row counts in checkpoint')
    if record['inner_training_rows'] + record['inner_validation_rows'] + record['inner_embargo_rows'] != record['outer_training_rows']:
        raise ValueError('Inner fold accounting mismatch')
    if record['inner_embargo_rows'] != 12:
        raise ValueError('Inner gap mismatch')
    if not 1 <= record['best_epoch'] <= record['inner_epochs_completed'] <= MAX_SELECTION_EPOCHS:
        raise ValueError('Invalid selected epoch')
    for name in ('validation_mae_kw','validation_rmse_kw','best_inner_validation_loss_scaled'):
        if record[name] < 0:
            raise ValueError('Negative CV metric')
    for name in ('inner_selection_seconds','outer_refit_seconds','outer_prediction_seconds'):
        if record[name] <= 0:
            raise ValueError('Invalid runtime')


def run_one_cv(*, payload: dict, root: Path, model_name: str, config_index: int,
               source_sha256: str, output_dir: Path) -> tuple[dict, bool]:
    """Run/reuse ONE candidate-fold fit; fail closed on provenance changes."""
    config = dict(grid(model_name)[config_index]) if type(config_index) is int and 0 <= config_index < 4 else None
    if config is None:
        raise ValueError('Candidate index must be 0..3')
    train_idx, val_idx, inner = validate_campaign_payload(payload)
    if source_sha256 != campaign_source_sha256(root):
        raise ValueError('Frozen source fingerprint mismatch')
    import tensorflow as tf
    config_json = json.loads(json.dumps(config, sort_keys=True))
    identity = {
        'model':model_name, 'configuration':config_json, 'config_index':config_index,
        'horizon_minutes':payload['horizon_minutes'], 'feature_family':payload['family'],
        'outer_fold':payload['fold'], 'outer_train_rows':len(train_idx),
        'outer_validation_rows':len(val_idx),
        'inner_fraction':INNER_VALIDATION_FRACTION, 'inner_gap_minutes':INNER_LABEL_GAP_MINUTES,
        'max_epochs':MAX_SELECTION_EPOCHS, 'patience':EARLY_STOPPING_PATIENCE,
        'min_delta':EARLY_STOPPING_MIN_DELTA, 'batch_size':BATCH_SIZE, 'seed':SEED,
        'source_sha256':source_sha256, 'data_sha256':_hash_payload(payload),
        'tensorflow_version':tf.__version__, 'numpy_version':np.__version__,
    }
    path = Path(output_dir)/checkpoint_name(model_name, payload['horizon_minutes'], payload['family'], config_index, payload['fold'])
    path.parent.mkdir(parents=True,exist_ok=True)
    lock = path.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Checkpoint locked by other/stale run: {lock}') from exc
    os.close(fd)
    try:
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
            if state.get('schema_version') != SCHEMA_VERSION or state.get('kind') != KIND or state.get('identity') != identity:
                raise ValueError('CV checkpoint identity mismatch; refusing reuse')
            record = state.get('record')
            _check_record(record, identity)
            return record, False
        record = _train_one(payload=payload, model_name=model_name, configuration=config,
                            train_idx=train_idx, val_idx=val_idx, inner=inner)
        _check_record(record, identity)
        from src.revised_rf_checkpoint import atomic_text
        state = {'schema_version':SCHEMA_VERSION,'kind':KIND,'identity':identity,'record':record}
        atomic_text(path, json.dumps(state, indent=2, sort_keys=True, allow_nan=False)+'\n',refuse_different=True)
        return record, True
    finally:
        lock.unlink(missing_ok=True)


def read_campaign_progress(output_dir: Path) -> pd.DataFrame:
    """Read checkpoint summary; never rank incomplete (candidate/fold) grids."""
    rows=[]
    for path in sorted(Path(output_dir).glob('*.json')):
        state=json.loads(path.read_text(encoding='utf-8'))
        if state.get('schema_version') != SCHEMA_VERSION or state.get('kind') != KIND:
            raise ValueError(f'Unexpected score checkpoint type: {path}')
        identity=state['identity']
        if path.name != checkpoint_name(identity['model'],identity['horizon_minutes'],identity['feature_family'],identity['config_index'],identity['outer_fold']):
            raise ValueError('Checkpoint filename/identity mismatch')
        if identity['configuration'] != json.loads(json.dumps(grid(identity['model'])[identity['config_index']])):
            raise ValueError('Checkpoint grid identity mismatch')
        _check_record(state.get('record'),identity)
        r=state['record']
        rows.append({
            'model':identity['model'],'horizon_minutes':identity['horizon_minutes'],
            'feature_family':identity['feature_family'], 'config_index':identity['config_index'],
            'candidate_id':identity['configuration']['candidate_id'], 'fold':identity['outer_fold'],
            'validation_mae_kw':r['validation_mae_kw'],'validation_rmse_kw':r['validation_rmse_kw'],
            'best_epoch':r['best_epoch'], 'selection_seconds':r['inner_selection_seconds'],
            'refit_seconds':r['outer_refit_seconds'],
        })
    if rows:
        df=pd.DataFrame(rows).sort_values(['model','horizon_minutes','feature_family','config_index','fold']).reset_index(drop=True)
        if df.duplicated(['model','horizon_minutes','feature_family','config_index','fold']).any():
            raise ValueError('Duplicate complete CV record')
        return df
    return pd.DataFrame(columns=[
        'model','horizon_minutes','feature_family','config_index','candidate_id',
        'fold','validation_mae_kw','validation_rmse_kw','best_epoch','selection_seconds','refit_seconds',
    ])


def complete_rankings(output_dir: Path) -> pd.DataFrame:
    """Rank only complete model/horizon/family grids with all 4 x 3 scores."""
    data = read_campaign_progress(output_dir)
    ranked=[]
    for (model,h,family),frame in data.groupby(['model','horizon_minutes','feature_family']):
        if len(frame) == 12 and all(frame.groupby('config_index')['fold'].apply(lambda x: set(x)=={1,2,3})):
            ranked.append(rank_neural_cv_scores(frame))
    return pd.concat(ranked,ignore_index=True) if ranked else pd.DataFrame()


def audit_milestone3_gates(*, root: Path, payload: dict, directory: Path) -> dict:
    """READ ONLY: validate existing M3 gates; never promote them into M4 CV.

    M3 training implementation and M4 campaign sources have different SHA
    identities. Therefore M3 scores are reported as pilot evidence only.
    """
    from src.revised_neural_full_cv import (
        gate_source_fingerprint, validate_fullsize_payload, _read_checkpoint,
        GATE_KIND, GATE_SEED, GATE_BATCH_SIZE, GATE_CONFIG_INDEX,
        GATE_TRAIN_ROWS, GATE_VALIDATION_ROWS, GATE_FAMILY, GATE_HORIZON, GATE_FOLD,
    )
    validate_fullsize_payload(payload)
    import tensorflow as tf
    old_sha = gate_source_fingerprint(root)
    data_sha = _hash_payload(payload)
    results = {}
    for model_name in MODELS:
        path = Path(directory) / f'{model_name.lower()}_c1_h15_F0_fold1.json'
        if not path.is_file():
            results[model_name] = None
            continue
        config = json.loads(json.dumps(grid(model_name)[0], sort_keys=True))
        identity = {
            'model': model_name, 'configuration': config,
            'candidate_index': GATE_CONFIG_INDEX,
            'horizon_minutes': GATE_HORIZON, 'feature_family': GATE_FAMILY,
            'outer_fold': GATE_FOLD, 'outer_train_rows': GATE_TRAIN_ROWS,
            'outer_validation_rows': GATE_VALIDATION_ROWS,
            'inner_fraction': INNER_VALIDATION_FRACTION,
            'inner_gap_minutes': INNER_LABEL_GAP_MINUTES,
            'max_epochs': MAX_SELECTION_EPOCHS, 'patience': EARLY_STOPPING_PATIENCE,
            'min_delta': EARLY_STOPPING_MIN_DELTA, 'batch_size': GATE_BATCH_SIZE,
            'seed': GATE_SEED, 'source_sha256': old_sha,
            'data_sha256': data_sha, 'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__,
        }
        record = _read_checkpoint(path, identity)
        if record is None:
            raise RuntimeError(f'Failed to read existing M3 gate: {path}')
        results[model_name] = record
    return results
