"""Milestone 2: source-frozen neural CV policy and bounded resource pilots.

No training, TensorFlow import or file writes on module import. The resource
pilot performs at most ONE epoch for LSTM_C1 and TCN_C1, using an inner split
of OUTER-TRAIN samples and no outer-validation labels. Pilot checkpoint records
are deliberately incompatible with selection/CV scores.

Full nested-CV model-selection runs are intentionally NOT enabled here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

INNER_VALIDATION_FRACTION = 0.10
INNER_LABEL_GAP_MINUTES = 60
MAX_SELECTION_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
EPOCH_RANKING = "minimum inner-validation loss; first epoch on ties"
OUTER_CV_RANKING = "unweighted mean of 3 outer-fold kW MAEs; then config index"
PILOT_HORIZON = 15
PILOT_FAMILY = "F0"
PILOT_FOLD = 1
PILOT_MAX_TRAIN_ROWS = 2048
PILOT_MAX_INNER_VAL_ROWS = 512
PILOT_EPOCHS = 1
PILOT_BATCH_SIZE = 128
PILOT_SEED = 42
PILOT_SCHEMA_VERSION = 1

# Candidate grids are historical MODEL SHAPES, not historical selected winners.
# Actual selections will be made afresh under the revised development protocol.
LSTM_CANDIDATES = (
    {"candidate_id": "LSTM_C1", "units": 32, "n_recurrent_layers": 1,
     "dropout": 0.0, "fusion_dense_units": 32},
    {"candidate_id": "LSTM_C2", "units": 64, "n_recurrent_layers": 1,
     "dropout": 0.0, "fusion_dense_units": 32},
    {"candidate_id": "LSTM_C3", "units": 64, "n_recurrent_layers": 1,
     "dropout": 0.20, "fusion_dense_units": 32},
    {"candidate_id": "LSTM_C4", "units": 64, "n_recurrent_layers": 2,
     "dropout": 0.20, "fusion_dense_units": 32},
)
TCN_CANDIDATES = (
    {"candidate_id": "TCN_C1", "filters": 32, "kernel_size": 3,
     "dilations": (1, 2, 4), "convolutions_per_block": 2,
     "dropout": 0.0, "fusion_dense_units": 32},
    {"candidate_id": "TCN_C2", "filters": 64, "kernel_size": 3,
     "dilations": (1, 2, 4), "convolutions_per_block": 2,
     "dropout": 0.0, "fusion_dense_units": 32},
    {"candidate_id": "TCN_C3", "filters": 64, "kernel_size": 2,
     "dilations": (1, 2, 4), "convolutions_per_block": 2,
     "dropout": 0.0, "fusion_dense_units": 32},
    {"candidate_id": "TCN_C4", "filters": 64, "kernel_size": 3,
     "dilations": (1, 2, 4, 8), "convolutions_per_block": 2,
     "dropout": 0.0, "fusion_dense_units": 32},
)


def _masks_and_times(payload: dict) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    seq = np.asarray(payload["sequence"])
    aux = np.asarray(payload["auxiliary"])
    target = np.asarray(payload["target_kw"])
    if seq.ndim != 3 or seq.shape[1:] != (13, 9):
        raise ValueError("Expected 13 x 9 historical input sequences")
    n = seq.shape[0]
    if aux.ndim != 2 or aux.shape[0] != n or target.shape != (n,):
        raise ValueError("Neural arrays have inconsistent row counts")
    train = np.asarray(payload["train_mask"])
    outer_val = np.asarray(payload["validation_mask"])
    if train.dtype != bool or outer_val.dtype != bool or train.shape != (n,) or outer_val.shape != (n,):
        raise ValueError("Outer masks must be row-aligned booleans")
    if np.any(train & outer_val) or not train.any() or not outer_val.any():
        raise ValueError("Outer masks overlap or are empty")
    if not np.all(train | outer_val):
        raise ValueError("Payload must contain ONLY the outer-train and outer-validation rows")
    times = payload["issue_time"]
    valid = payload["valid_time"]
    if not isinstance(times, pd.Series) or not isinstance(valid, pd.Series) or len(times) != n or len(valid) != n:
        raise ValueError("Invalid time series")
    if not times.index.equals(valid.index) or str(times.dt.tz) != "UTC" or str(valid.dt.tz) != "UTC":
        raise ValueError("Issue/valid times must be UTC aligned")
    if times.isna().any() or valid.isna().any() or not times.is_monotonic_increasing:
        raise ValueError("Invalid or unordered forecast times")
    horizon = int(payload["horizon_minutes"])
    if horizon not in (15, 30, 60) or not (valid == times + pd.Timedelta(minutes=horizon)).all():
        raise ValueError("Incorrect issue/target time alignment")
    train_idx, outer_idx = np.flatnonzero(train), np.flatnonzero(outer_val)
    if train_idx[-1] >= outer_idx[0]:
        raise ValueError("Outer training must precede validation")
    if valid.iloc[train_idx[-1]] >= times.iloc[outer_idx[0]]:
        raise ValueError("Outer labels are not available at first validation issue")
    return train, outer_val, times, valid


def make_inner_split(payload: dict, *, fraction: float = INNER_VALIDATION_FRACTION) -> dict:
    """Nested early-stopping split entirely INSIDE the outer training window.

    Inner train excludes labels in the last 60 min before inner validation
    starts, so its labels already exist at first inner forecast issue.
    The outer-validation labels are never used for epoch selection.
    """
    if not isinstance(fraction, (float, int)) or not 0 < fraction < 0.5:
        raise ValueError("Inner validation fraction must be between zero and one-half")
    train, outer_val, issue, valid = _masks_and_times(payload)
    positions = np.flatnonzero(train)
    stop = math.floor((1.0 - fraction) * len(positions))
    if stop < 10 or len(positions) - stop < 5:
        raise ValueError("Not enough outer-training rows for nested early stopping")
    inner_idx = positions[stop:]
    inner_start_valid = valid.iloc[inner_idx[0]]
    # Deliberately use the common maximum-horizon gap (not only this horizon).
    eligible = (valid < inner_start_valid - pd.Timedelta(minutes=INNER_LABEL_GAP_MINUTES)).to_numpy(dtype=bool)
    inner_train = train & eligible
    inner_val = np.zeros(len(train), dtype=bool)
    inner_val[inner_idx] = True
    if not inner_train.any() or not inner_val.any():
        raise ValueError("Empty inner split")
    if np.any(inner_train & inner_val) or np.any((inner_train | inner_val) & outer_val):
        raise ValueError("Inner split crosses into outer validation")
    if valid.iloc[np.flatnonzero(inner_train)[-1]] >= issue.iloc[inner_idx[0]]:
        raise ValueError("Inner training labels leak into early stopping")
    if inner_train.sum() + inner_val.sum() > train.sum():
        raise ValueError("Inner masks not contained in outer training")
    return {"inner_train_mask": inner_train, "inner_validation_mask": inner_val,
            "outer_train_mask": train, "outer_validation_mask": outer_val,
            "embargo_rows": int(train.sum() - inner_train.sum() - inner_val.sum())}


def pilot_indices(payload: dict, *, train_limit: int = PILOT_MAX_TRAIN_ROWS,
                  validation_limit: int = PILOT_MAX_INNER_VAL_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded chronological subsamples of nested-TRAIN/INNER-VAL only."""
    if type(train_limit) is not int or train_limit < 32 or type(validation_limit) is not int or validation_limit < 16:
        raise ValueError("Pilot limits too small or nonintegral")
    masks = make_inner_split(payload)
    inner_fit = np.flatnonzero(masks["inner_train_mask"])[-train_limit:]
    inner_val = np.flatnonzero(masks["inner_validation_mask"])[:validation_limit]
    if not len(inner_fit) or not len(inner_val) or inner_fit[-1] >= inner_val[0]:
        raise ValueError("Pilot chronology is invalid")
    return inner_fit, inner_val


def rank_neural_cv_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Require complete three-fold candidates; choose UNWEIGHTED mean kW MAE.

    This is for future COMPLETE CV only; resource-pilot rows are ineligible.
    """
    columns = ("model", "horizon_minutes", "feature_family", "config_index",
               "candidate_id", "fold", "validation_mae_kw", "validation_rmse_kw", "best_epoch")
    if not isinstance(scores, pd.DataFrame) or scores.empty or any(c not in scores for c in columns):
        raise ValueError("Missing full neural CV scores or columns")
    if scores.isna().any().any() or scores.duplicated(list(columns[:4]) + ["fold"]).any():
        raise ValueError("Missing values or duplicate candidate/fold scores")
    if not scores["model"].isin(("LSTM", "TCN")).all() or not scores["horizon_minutes"].isin((15, 30, 60)).all() or not scores["feature_family"].isin(("F0", "F1", "F2", "F3", "F4")).all():
        raise ValueError("Invalid model/horizon/family")
    if not scores["fold"].isin((1, 2, 3)).all():
        raise ValueError("Unexpected outer fold")
    if not np.isfinite(scores[["validation_mae_kw", "validation_rmse_kw", "best_epoch"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite CV metrics")
    if (scores[["validation_mae_kw", "validation_rmse_kw"]].to_numpy(dtype=float) < 0).any() or (scores["best_epoch"] < 1).any():
        raise ValueError("Invalid scores/epochs")
    keys = list(columns[:5])
    groups = scores.groupby(keys, sort=True)
    if not groups["fold"].agg(lambda x: set(x) == {1, 2, 3} and len(x) == 3).all():
        raise ValueError("Every candidate must have exactly three distinct folds")
    rows = groups.agg(cv_mae_kw=("validation_mae_kw", "mean"),
                      cv_rmse_kw=("validation_rmse_kw", "mean"),
                      folds=("fold", "count")).reset_index()
    # Candidate identity must not vary for a given model/config_index.
    if rows.duplicated(["model", "horizon_minutes", "feature_family", "config_index"]).any():
        raise ValueError("Same config index maps to multiple candidate IDs")
    rows = rows.sort_values(["model", "horizon_minutes", "feature_family", "cv_mae_kw", "config_index"], kind="stable")
    rows.insert(0, "rank", rows.groupby(["model", "horizon_minutes", "feature_family"]).cumcount() + 1)
    return rows.reset_index(drop=True)


def _array_signature(parts: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in parts:
        arr = np.ascontiguousarray(arr)
        h.update(repr((arr.shape, arr.dtype.str)).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _checkpoint(path: Path, identity: dict, record: dict | None = None) -> dict | None:
    """Refuse stale/malformed pilot checkpoints, atomically commit new ones."""
    from src.revised_rf_checkpoint import atomic_text
    path = Path(path)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded.get("schema_version") != PILOT_SCHEMA_VERSION or loaded.get("kind") != "resource-pilot-not-cv" or loaded.get("identity") != identity:
            raise ValueError(f"Pilot checkpoint identity mismatch: {path}")
        existing = loaded.get("record")
        if not isinstance(existing, dict) or set(existing) != {"elapsed_seconds", "trainable_parameters", "train_rows", "inner_validation_rows", "epochs"}:
            raise ValueError("Incomplete/corrupted pilot checkpoint")
        if (not np.isfinite(existing["elapsed_seconds"]) or existing["elapsed_seconds"] <= 0 or
                type(existing["trainable_parameters"]) is not int or existing["trainable_parameters"] <= 0 or
                type(existing["train_rows"]) is not int or existing["train_rows"] <= 0 or
                type(existing["inner_validation_rows"]) is not int or existing["inner_validation_rows"] <= 0 or
                existing["epochs"] != 1):
            raise ValueError("Invalid pilot checkpoint metrics")
        return existing
    if record is None:
        return None
    obj = {"schema_version": PILOT_SCHEMA_VERSION, "kind": "resource-pilot-not-cv",
           "identity": identity, "record": record}
    atomic_text(path, json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n", refuse_different=True)
    return record


def source_fingerprint(root: Path) -> str:
    paths = ("src/forecasting_protocol.py", "src/neural_models.py",
             "src/revised_neural_cv.py", "src/revised_neural_training.py")
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()


def run_resource_pilot(*, payload: dict, root: Path, model_name: str,
                       source_sha256: str, output_dir: Path) -> tuple[dict, bool]:
    """One inner-only one-epoch feasibility fit; NEVER outer-CV evaluation.

    Exactly 2 independently checkpointed pilot models are permitted.
    Reruns skip an already matching checkpoint; no holdout arrays or outer
    validation labels reach TensorFlow fit/scaling/prediction.
    """
    if (int(payload["horizon_minutes"]), payload["family"], int(payload["fold"])) != (PILOT_HORIZON, PILOT_FAMILY, PILOT_FOLD):
        raise ValueError("Resource pilot is restricted to 15min/F0/fold1")
    if model_name not in ("LSTM", "TCN"):
        raise ValueError("Pilot supports only LSTM and TCN")
    if not isinstance(source_sha256, str) or source_sha256 != source_fingerprint(root):
        raise ValueError("Source fingerprint mismatch")
    from src.neural_models import (fit_train_only_scalers, apply_train_only_scalers,
                                   build_recurrent_fusion_model, build_tcn_fusion_model)
    import tensorflow as tf
    train_idx, val_idx = pilot_indices(payload)
    selected = np.concatenate((train_idx, val_idx))
    training_mask = np.zeros(len(selected), dtype=bool)
    training_mask[:len(train_idx)] = True
    seq = np.asarray(payload["sequence"])[selected]
    aux = np.asarray(payload["auxiliary"])[selected]
    targets = np.asarray(payload["target_kw"])[selected]
    config = dict((LSTM_CANDIDATES if model_name == "LSTM" else TCN_CANDIDATES)[0])
    data_sha = _array_signature([seq, aux, targets, np.asarray(payload["issue_time"].iloc[selected].astype("int64"))])
    identity = {"source_sha256": source_sha256, "data_sha256": data_sha,
                "tensorflow_version": tf.__version__, "numpy_version": np.__version__,
                "model": model_name, "configuration": json.loads(json.dumps(config)),
                "horizon_minutes": PILOT_HORIZON, "family": PILOT_FAMILY,
                "outer_fold": PILOT_FOLD, "inner_gap_minutes": INNER_LABEL_GAP_MINUTES,
                "seed": PILOT_SEED, "train_limit": PILOT_MAX_TRAIN_ROWS,
                "inner_validation_limit": PILOT_MAX_INNER_VAL_ROWS,
                "epochs": PILOT_EPOCHS, "batch_size": PILOT_BATCH_SIZE}
    # Use a separate PILOT namespace, not the full-CV model selection archive.
    path = Path(output_dir) / f"{model_name.lower()}_c1_one_epoch.json"
    prior = _checkpoint(path, identity)
    if prior is not None:
        return prior, False
    stats = fit_train_only_scalers(seq, aux, targets, training_mask,
                                   continuous_positions=payload["continuous_positions"],
                                   binary_positions=payload["binary_positions"])
    scaled = apply_train_only_scalers(seq, aux, targets, stats)
    if model_name == "LSTM":
        model = build_recurrent_fusion_model("LSTM", aux.shape[1], config, seed=PILOT_SEED)
    else:
        from src.neural_models import reset_neural_state
        reset_neural_state(PILOT_SEED)
        model = build_tcn_fusion_model(config, aux.shape[1])
    started = perf_counter()
    try:
        history = model.fit(
            [scaled["sequence"][:len(train_idx)], scaled["auxiliary"][:len(train_idx)]],
            scaled["target_scaled"][:len(train_idx)],
            validation_data=(
                [scaled["sequence"][len(train_idx):], scaled["auxiliary"][len(train_idx):]],
                scaled["target_scaled"][len(train_idx):],
            ),
            epochs=PILOT_EPOCHS, batch_size=PILOT_BATCH_SIZE,
            shuffle=True, verbose=0,
        )
        duration = perf_counter() - started
        if len(history.history.get("loss", [])) != PILOT_EPOCHS:
            raise RuntimeError("Pilot did not complete the single required epoch")
        record = {"elapsed_seconds": float(duration), "trainable_parameters": int(model.count_params()),
                  "train_rows": int(len(train_idx)), "inner_validation_rows": int(len(val_idx)), "epochs": PILOT_EPOCHS}
        _checkpoint(path, identity, record)
    finally:
        tf.keras.backend.clear_session()
        del model
    return record, True
