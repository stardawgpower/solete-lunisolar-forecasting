"""Milestone 5: opt-in orchestration of the frozen revised neural CV campaign.

This module changes ONLY scheduling, not fitted model code, feature building,
selection policy or model scores. Each CV evaluation is independently persisted
by src.revised_neural_campaign.run_one_cv. No work runs on import.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from src.revised_neural_campaign import (
    FAMILIES, HORIZONS, KIND, MODELS, SCHEMA_VERSION,
    campaign_source_sha256, checkpoint_name, grid,
    read_campaign_progress, run_one_cv,
)


def campaign_keys(models: Sequence[str] = MODELS) -> tuple[tuple[str, int, str, int, int], ...]:
    """Canonical, deterministic 360-key grid (or a model-restricted subset)."""
    models = tuple(models)
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("models must be distinct names from LSTM and TCN")
    return tuple(
        (model, horizon, family, candidate, fold)
        for model in models
        for horizon in HORIZONS
        for family in FAMILIES
        for candidate in range(len(grid(model)))
        for fold in (1, 2, 3)
    )


def _audit_prior_source(output_dir: Path, *, source_sha256: str) -> int:
    """Reject source-incompatible checkpoint collections before fitting more."""
    progress = read_campaign_progress(output_dir)
    for path in sorted(output_dir.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION or state.get("kind") != KIND:
            raise ValueError(f"Unexpected CV checkpoint kind: {path}")
        identity = state.get("identity", {})
        if identity.get("source_sha256") != source_sha256:
            raise ValueError(f"Archived CV source fingerprint differs: {path}")
    return len(progress)


def run_checkpointed_batch(
    *,
    root: Path,
    output_dir: Path,
    payload_factory: Callable[[int, str, int], dict],
    max_new_fits: int = 1,
    models: Sequence[str] = ("LSTM",),
    enabled: bool = False,
    report: Callable[[str], None] = print,
) -> dict:
    """Resume a bounded number of *new* fits, verifying every reused record.

    The payload factory must mask terminal targets and call the audited
    `prepare_development_fold`. `run_one_cv` independently validates every
    payload and its data/source fingerprints before fit/reuse. One global lock
    prevents two simultaneous batch launchers in the same output directory.
    An interrupted fit is not counted and may be retried; completed files are
    immutable and reused, not overwritten. Exceptions propagate/fail closed.
    """
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    if type(max_new_fits) is not int or not 1 <= max_new_fits <= 360:
        raise ValueError("max_new_fits must be an integer from 1 to 360")
    if not callable(payload_factory) or not callable(report):
        raise ValueError("payload_factory and report must be callable")
    keys = campaign_keys(models)
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    actual_sha = campaign_source_sha256(root)
    initial = _audit_prior_source(output_dir, source_sha256=actual_sha)
    if not enabled:
        return {"enabled": False, "new_fits": 0, "reused_fits": 0,
                "before": initial, "after": initial, "total_targets": len(keys)}

    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".neural_batch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Neural batch already running or stale lock: {lock}") from exc
    os.close(fd)
    new_fits = 0
    reused_fits = 0
    last_key = None
    try:
        # Fresh audit under lock also catches concurrent modifications between calls.
        _audit_prior_source(output_dir, source_sha256=actual_sha)
        for model, horizon, family, candidate, fold in keys:
            path = output_dir / checkpoint_name(model, horizon, family, candidate, fold)
            # Stop *before* a new fit when the cap is reached; do not validate
            # every remaining completed checkpoint unnecessarily.
            if new_fits >= max_new_fits and not path.is_file():
                break
            payload = payload_factory(horizon, family, fold)
            record, did_fit = run_one_cv(
                payload=payload, root=root, model_name=model,
                config_index=candidate, source_sha256=actual_sha,
                output_dir=output_dir,
            )
            last_key = (model, horizon, family, candidate, fold)
            if did_fit:
                new_fits += 1
                report(f"NEW {new_fits}/{max_new_fits}: {model} h{horizon} {family} C{candidate+1} fold{fold} "
                       f"MAE={record['validation_mae_kw']:.6f} kW; checkpoint saved")
            else:
                reused_fits += 1
            del payload, record
            if new_fits >= max_new_fits:
                break
        after = _audit_prior_source(output_dir, source_sha256=actual_sha)
        report(f"MILESTONE 5 BATCH: PASSED — new {new_fits}, reused {reused_fits}; "
               f"archive {after}/360; target subset {len(keys)}")
        return {"enabled": True, "new_fits": new_fits, "reused_fits": reused_fits,
                "before": initial, "after": after, "total_targets": len(keys),
                "last_key": last_key}
    finally:
        lock.unlink(missing_ok=True)
