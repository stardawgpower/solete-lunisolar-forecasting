"""Milestone 6: auditable combined routing of immutable V1/F0 + V2/F1-F4.

F0 keeps the exact original training code/checkpoint fingerprints; new
F1-F4 folds use the year-safe V2 scaler and a separate checkpoint archive.
No work runs on import. No terminal labels or terminal predictors are used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable, Sequence

import pandas as pd

from src.revised_neural_batch import campaign_keys, _audit_prior_source
from src.revised_neural_campaign import (
    campaign_source_sha256 as v1_source, checkpoint_name as v1_name,
    read_campaign_progress as v1_progress, run_one_cv as run_v1,
)
from src.revised_neural_campaign_v2 import (
    campaign_source_sha256 as v2_source, checkpoint_name as v2_name,
    read_campaign_progress as v2_progress, run_one_cv as run_v2,
    SCALING_POLICY, KIND as V2_KIND, SCHEMA_VERSION as V2_SCHEMA,
)
from src.revised_neural_training import rank_neural_cv_scores

V2_DIRNAME = 'revised_neural_cv_v2_year_std1'


def _audit_v2(directory: Path, fingerprint: str) -> int:
    scores = v2_progress(directory)
    for path in sorted(Path(directory).glob('*.json')):
        state = json.loads(path.read_text(encoding='utf-8'))
        ident = state.get('identity', {})
        if (state.get('kind') != V2_KIND or state.get('schema_version') != V2_SCHEMA
                or ident.get('source_sha256') != fingerprint
                or ident.get('scaling_policy') != SCALING_POLICY):
            raise ValueError(f'V2 checkpoint source/scaling-policy fingerprint mismatch: {path}')
    if not scores.empty and not scores['feature_family'].isin(('F1', 'F2', 'F3', 'F4')).all():
        raise ValueError('V2 archive contains an F0 checkpoint')
    return len(scores)


def read_mixed_progress(v1_dir: Path, v2_dir: Path) -> pd.DataFrame:
    one, two = v1_progress(v1_dir), v2_progress(v2_dir)
    if not one.empty and not (one['feature_family'] == 'F0').all():
        raise ValueError('V1 archive unexpectedly contains F1-F4 evaluations')
    if not two.empty and not two['feature_family'].isin(('F1','F2','F3','F4')).all():
        raise ValueError('V2 archive unexpectedly contains F0 evaluations')
    present = [frame for frame in (one, two) if not frame.empty]
    combined = pd.concat(present, ignore_index=True) if present else one.copy()
    key = ['model','horizon_minutes','feature_family','config_index','fold']
    if combined.duplicated(key).any() or len(combined) > 360:
        raise ValueError('Overlapping or impossible combined CV records')
    return combined.sort_values(key).reset_index(drop=True)


def complete_mixed_rankings(v1_dir: Path, v2_dir: Path) -> pd.DataFrame:
    scores = read_mixed_progress(v1_dir, v2_dir)
    completed = []
    for _, block in scores.groupby(['model','horizon_minutes','feature_family']):
        if (len(block) == 12 and len(block['config_index'].unique()) == 4
                and all(set(f['fold']) == {1,2,3} for _, f in block.groupby('config_index'))):
            completed.append(rank_neural_cv_scores(block))
    return pd.concat(completed, ignore_index=True) if completed else pd.DataFrame()


def run_mixed_batch(*, root: Path, v1_dir: Path, v2_dir: Path,
                    payload_factory: Callable[[int,str,int],dict],
                    max_new_fits: int = 1, models: Sequence[str] = ('LSTM',),
                    enabled: bool = False, report: Callable[[str],None] = print) -> dict:
    """At most max_new_fits new evaluations, exact checkpoints, route by family.

    Both archives and both frozen sources are verified *before* training, even
    if there are no new evaluations. A single V1 lock prevents overlapping
    legacy batch launches while this process owns the F1-F4 archive too.
    """
    if type(enabled) is not bool or type(max_new_fits) is not int or not 1 <= max_new_fits <= 360:
        raise ValueError('Invalid enabled switch or new-fit cap')
    if not callable(payload_factory) or not callable(report):
        raise ValueError('Invalid payload factory or reporting callback')
    keys = campaign_keys(models)
    root = Path(root).resolve()
    v1_dir, v2_dir = Path(v1_dir).resolve(), Path(v2_dir).resolve()
    if v1_dir == v2_dir:
        raise ValueError('Original and corrected checkpoint directories must differ')
    sha1, sha2 = v1_source(root), v2_source(root)
    prior_v1 = _audit_prior_source(v1_dir, source_sha256=sha1)
    prior_v2 = _audit_v2(v2_dir, sha2)
    read_mixed_progress(v1_dir, v2_dir)
    initial = prior_v1 + prior_v2
    if not enabled:
        return {'enabled': False, 'new_fits':0, 'reused_fits':0,
                'before':initial, 'after':initial, 'total_targets':len(keys)}
    v1_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    lock = v1_dir / '.neural_batch.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Batch already running or stale lock: {lock}') from exc
    os.close(fd)
    done = reused = 0
    last = None
    try:
        _audit_prior_source(v1_dir, source_sha256=sha1)
        _audit_v2(v2_dir, sha2)
        for model, horizon, family, candidate, fold in keys:
            v1 = family == 'F0'
            directory = v1_dir if v1 else v2_dir
            name = v1_name if v1 else v2_name
            existing = directory / name(model,horizon,family,candidate,fold)
            if done >= max_new_fits and not existing.is_file():
                break
            payload = payload_factory(horizon,family,fold)
            record, is_new = (run_v1 if v1 else run_v2)(
                payload=payload,root=root,model_name=model,
                config_index=candidate,source_sha256=sha1 if v1 else sha2,
                output_dir=directory,
            )
            last = (model,horizon,family,candidate,fold)
            if is_new:
                done += 1
                report(f'NEW {done}/{max_new_fits}: {model} h{horizon} {family} '
                       f'C{candidate+1} fold{fold} MAE={record["validation_mae_kw"]:.6f} kW; '
                       f'{"V1" if v1 else "V2 year-safe"} checkpoint saved')
            else:
                reused += 1
            del payload, record
            if done >= max_new_fits:
                break
        after_v1 = _audit_prior_source(v1_dir, source_sha256=sha1)
        after_v2 = _audit_v2(v2_dir, sha2)
        combined = read_mixed_progress(v1_dir, v2_dir)
        assert len(combined) == after_v1 + after_v2
        report(f'MILESTONE 6 MIXED BATCH: PASSED — new {done}, reused {reused}; '
               f'V1 F0={after_v1}; V2 F1-F4={after_v2}; total {len(combined)}/360')
        return {'enabled':True,'new_fits':done,'reused_fits':reused,
                'before':initial,'after':len(combined),'total_targets':len(keys),
                'v1':after_v1,'v2':after_v2,'last_key':last}
    finally:
        lock.unlink(missing_ok=True)
