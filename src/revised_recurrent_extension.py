"""GRU/BiLSTM development CV extension to the frozen LSTM/TCN campaign.

This file never imports TensorFlow, fits models, or writes files on import.
F0 reuses the V1 validation/training and scaling procedure; F1-F4 reuse
V2, including the inner-TRAIN-only treatment of constant calendar year.
The original 360 checkpoints and their source-fingerprint files stay unchanged.

The frozen V1/V2 functions are rebound into private function namespaces,
NOT monkeypatched. Only candidate choice, recurrent model constructor,
archive filenames and provenance hash differ from the frozen runners.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from src import revised_neural_campaign as v1
from src import revised_neural_campaign_v2 as v2
from src.revised_neural_batch_v2 import read_mixed_progress
from src.revised_neural_full_cv import _hash_payload
from src.revised_neural_training import LSTM_CANDIDATES

MODELS = ("GRU", "BiLSTM")
HORIZONS = (15, 30, 60)
FAMILIES = ("F0", "F1", "F2", "F3", "F4")
F0_ARCHIVE = "revised_recurrent_extension_v1_F0"
V2_ARCHIVE = "revised_recurrent_extension_v2_year_std1"
EXPECTED_FITS = 360  # 2 models x 3 horizons x 5 families x 4 settings x 3 folds


def candidates(model: str) -> tuple[dict, ...]:
    """The same four frozen recurrent shapes as LSTM; new model-specific IDs."""
    if model not in MODELS:
        raise ValueError(f"Unsupported new architecture: {model}")
    return tuple({**cfg, "candidate_id": f"{model}_C{i}"}
                 for i, cfg in enumerate(LSTM_CANDIDATES, 1))


def checkpoint_name(model: str, horizon: int, family: str, config: int, fold: int) -> str:
    if (model not in MODELS or type(horizon) is not int or horizon not in HORIZONS
            or family not in FAMILIES or type(config) is not int or not 0 <= config < 4
            or type(fold) is not int or fold not in (1, 2, 3)):
        raise ValueError("Invalid GRU/BiLSTM checkpoint key")
    return f"{model.lower()}_c{config+1}_h{horizon}_{family}_fold{fold}.json"


def source_fingerprint(root: Path) -> str:
    """Bind new checkpoints to both frozen implementations and this extension."""
    root = Path(root).resolve()
    digest = sha256()
    for key, value in (
        ("frozen-v1", v1.campaign_source_sha256(root)),
        ("frozen-v2", v2.campaign_source_sha256(root)),
        ("recurrent-extension", (root / "src/revised_recurrent_extension.py").read_bytes()),
    ):
        digest.update(key.encode() + b"\0")
        digest.update(value if isinstance(value, bytes) else value.encode())
    return digest.hexdigest()


def _build_model(model: str, configuration: dict, aux_dim: int):
    from src.neural_models import build_recurrent_fusion_model, reset_neural_state
    if model not in MODELS:
        raise ValueError("Only GRU and BiLSTM may be fitted by this extension")
    reset_neural_state(v2.SEED)
    return build_recurrent_fusion_model(model, aux_dim, configuration, seed=v2.SEED)


def _bind(original: Callable, **symbols) -> Callable:
    """Make a private call namespace for a frozen training function.

    Neither frozen module's globals nor its files are changed. The same code
    object still controls inner stopping, refitting, scaling and outer scoring.
    """
    namespace = dict(original.__globals__)
    namespace.update(symbols)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


_F0_TRAIN = _bind(v1._train_one, _make_model=_build_model)
_V2_TRAIN = _bind(v2._train_one, _make_model=_build_model)
_F0_ONE = _bind(v1.run_one_cv, grid=candidates, _make_model=_build_model,
                _train_one=_F0_TRAIN, checkpoint_name=checkpoint_name,
                campaign_source_sha256=source_fingerprint)
_V2_ONE = _bind(v2.run_one_cv, grid=candidates, _make_model=_build_model,
                _train_one=_V2_TRAIN, checkpoint_name=checkpoint_name,
                campaign_source_sha256=source_fingerprint)
_F0_READ = _bind(v1.read_campaign_progress, grid=candidates,
                 checkpoint_name=checkpoint_name)
_V2_READ = _bind(v2.read_campaign_progress, grid=candidates,
                 checkpoint_name=checkpoint_name)


def expected_keys(models: Sequence[str] = MODELS) -> tuple[tuple, ...]:
    chosen = tuple(models)
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen).issubset(MODELS):
        raise ValueError("Choose unique models from GRU and BiLSTM")
    return tuple(product(chosen, HORIZONS, FAMILIES, range(4), (1, 2, 3)))


def _check_archive(folder: Path, *, family_scope: str, fingerprint: str) -> pd.DataFrame:
    reader = _F0_READ if family_scope == "F0" else _V2_READ
    table = reader(folder)  # strict filename/schema/config/finite-metric checks
    families = {"F0"} if family_scope == "F0" else set(FAMILIES[1:])
    if not set(table["feature_family"]).issubset(families):
        raise ValueError(f"Wrong feature family in {folder}")
    for path in Path(folder).glob("*.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"].get("source_sha256") != fingerprint:
            raise ValueError(f"Changed extension/frozen source fingerprint: {path}")
        if family_scope != "F0" and saved["identity"].get("scaling_policy") != v2.SCALING_POLICY:
            raise ValueError(f"Incorrect V2 scaling policy: {path}")
    return table


def read_progress(root: Path, *, verify_source: bool = True) -> pd.DataFrame:
    """Read ONLY new GRU and BiLSTM checkpoints; no training or writes."""
    root = Path(root).resolve()
    base = root / "results/development_only"
    fingerprint = source_fingerprint(root) if verify_source else None
    if verify_source:
        a = _check_archive(base/F0_ARCHIVE, family_scope="F0", fingerprint=fingerprint)
        b = _check_archive(base/V2_ARCHIVE, family_scope="F1-F4", fingerprint=fingerprint)
    else:
        a, b = _F0_READ(base/F0_ARCHIVE), _V2_READ(base/V2_ARCHIVE)
    result = pd.concat([a, b], ignore_index=True)
    key = ["model", "horizon_minutes", "feature_family", "config_index", "fold"]
    if result.duplicated(key).any() or len(result) > EXPECTED_FITS:
        raise ValueError("Repeated or excessive extension scores")
    observed = set(tuple(row) for row in result[key].itertuples(index=False, name=None))
    if not observed.issubset(set(expected_keys())):
        raise ValueError("Unexpected experiment in extension archives")
    return result.sort_values(key).reset_index(drop=True)


def complete_rankings(root: Path) -> pd.DataFrame:
    """Unweighted three-fold CV MAE with the same index tie-break as LSTM."""
    df = read_progress(root)
    ranked = []
    for (model, horizon, family), frame in df.groupby(["model", "horizon_minutes", "feature_family"]):
        if len(frame) != 12 or any(set(block["fold"]) != {1, 2, 3}
                                     for _, block in frame.groupby("config_index")):
            continue
        rows = frame.groupby(["model", "horizon_minutes", "feature_family", "config_index", "candidate_id"], as_index=False).agg(
            cv_mae_kw=("validation_mae_kw", "mean"),
            cv_rmse_kw=("validation_rmse_kw", "mean"), folds=("fold", "count"))
        rows = rows.sort_values(["cv_mae_kw", "config_index"], kind="stable").reset_index(drop=True)
        rows.insert(0, "rank", np.arange(1, len(rows)+1))
        ranked.append(rows)
    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def _reference_identity(root: Path, horizon: int, family: str, fold: int) -> dict:
    """Require exact LSTM checkpoint's canonical data population/fingerprint."""
    folder = "revised_neural_cv_v1" if family == "F0" else "revised_neural_cv_v2_year_std1"
    path = root / "results/development_only" / folder / f"lstm_c1_h{horizon}_{family}_fold{fold}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved["identity"]


def _verify_existing_data(root: Path, frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        folder = F0_ARCHIVE if row.feature_family == "F0" else V2_ARCHIVE
        path = root/"results/development_only"/folder/checkpoint_name(
            row.model, row.horizon_minutes, row.feature_family, row.config_index, row.fold)
        identity = json.loads(path.read_text(encoding="utf-8"))["identity"]
        reference = _reference_identity(root, row.horizon_minutes, row.feature_family, row.fold)
        if identity["data_sha256"] != reference["data_sha256"]:
            raise ValueError(f"Extension and LSTM input data differ: {path}")


def run_batch(*, root: Path, payload_factory: Callable[[int, str, int], dict],
              max_new_fits: int = 1, models: Sequence[str] = MODELS,
              enabled: bool = False, report: Callable[[str], None] = print) -> dict:
    """Bounded resumable campaign; completely read-only unless enabled=True."""
    if type(enabled) is not bool or type(max_new_fits) is not int or not 1 <= max_new_fits <= EXPECTED_FITS:
        raise ValueError("Invalid training switch or batch limit")
    keys = expected_keys(models)
    if not callable(payload_factory) or not callable(report):
        raise ValueError("Invalid payload factory or report")
    root = Path(root).resolve()
    base = root / "results/development_only"
    f0_dir, v2_dir = base/F0_ARCHIVE, base/V2_ARCHIVE

    original = read_mixed_progress(base/"revised_neural_cv_v1",
                                   base/"revised_neural_cv_v2_year_std1")
    if len(original) != 360:
        raise ValueError("Original 360 LSTM/TCN records must remain complete")
    prior = read_progress(root)
    _verify_existing_data(root, prior)
    before = len(prior)
    if not enabled:
        return {"enabled": False, "before": before, "after": before,
                "new_fits": 0, "total_targets": EXPECTED_FITS}

    # No existing scores are modified; one lock for both extension directories.
    f0_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    lock = f0_dir/".recurrent_extension_batch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Concurrent or stale extension batch lock: {lock}") from exc
    os.close(fd)
    new_fits = 0
    try:
        existing = set(tuple(x) for x in prior[
            ["model", "horizon_minutes", "feature_family", "config_index", "fold"]
        ].itertuples(index=False, name=None))
        fingerprint = source_fingerprint(root)
        for model, horizon, family, config, fold in keys:
            key = (model, horizon, family, config, fold)
            if key in existing:
                continue
            if new_fits >= max_new_fits:
                break
            payload = payload_factory(horizon, family, fold)
            identity = _reference_identity(root, horizon, family, fold)
            if _hash_payload(payload) != identity["data_sha256"]:
                raise ValueError(f"Canonical input mismatch: {model} h{horizon} {family} fold{fold}")
            func = _F0_ONE if family == "F0" else _V2_ONE
            folder = f0_dir if family == "F0" else v2_dir
            record, created = func(payload=payload, root=root, model_name=model,
                                   config_index=config, source_sha256=fingerprint,
                                   output_dir=folder)
            if not created:
                raise RuntimeError("Unexpected existing checkpoint after preflight")
            new_fits += 1
            report(f"NEW {new_fits}/{max_new_fits}: {model} h{horizon} {family} C{config+1} "
                   f"fold{fold} MAE={record['validation_mae_kw']:.6f} kW; checkpoint saved")
            del payload, record
        after = read_progress(root)
        _verify_existing_data(root, after)
        if len(after) != before+new_fits:
            raise RuntimeError("Extension checkpoint accounting mismatch")
        report(f"RECURRENT EXTENSION BATCH: PASSED — new {new_fits}; total {len(after)}/{EXPECTED_FITS}")
        return {"enabled": True, "before": before, "after": len(after),
                "new_fits": new_fits, "total_targets": EXPECTED_FITS}
    finally:
        lock.unlink(missing_ok=True)
