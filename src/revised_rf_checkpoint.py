"""Development-only, per-fit restart-safe Random Forest CV checkpoints.

No model fitting occurs on import. The scientific folds/models are supplied by
Notebook 10 and its existing frozen implementation. Checkpoints contain only
validation metrics and run provenance; no predictions or terminal labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
FIELDS = ('config_index', 'fold', 'train_rows', 'validation_rows',
          'validation_mae', 'validation_rmse', 'elapsed_seconds')


def _as_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def atomic_text(path: Path, content: str, *, refuse_different: bool = False) -> None:
    """Durable write; optionally refuse to replace different existing content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_text(encoding='utf-8')
        if previous == content:
            return
        if refuse_different:
            raise ValueError(f'Existing development archive differs: {path}')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def fingerprint(X, y, issue_time, valid_time, forbidden_mask, folds,
                configurations, *, horizon_minutes, feature_family,
                source_sha256):
    """Fingerprint exact rows/values, fold assignments, grid, and frozen code.

    Labels are supplied already masked; the terminal target cannot enter this
    fingerprint or any estimator. Hash values are recorded, not data itself.
    """
    assert y.loc[forbidden_mask].isna().all()
    assert X.index.equals(y.index) and y.index.equals(valid_time.index)
    assert y.index.equals(issue_time.index) and y.index.equals(forbidden_mask.index)
    assert len(folds) == 3 and len(configurations) == 8
    h = hashlib.sha256()
    for val in (SCHEMA_VERSION, horizon_minutes, feature_family,
                source_sha256, list(map(str, X.columns)),
                list(map(str, X.dtypes)), list(configurations)):
        h.update(_as_json(val).encode('utf-8'))
        h.update(b'\n')
    for frame in (X, y, issue_time, valid_time, forbidden_mask):
        # Per-row hashes make X, timestamps, target labels and index part of identity.
        values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype='uint64')
        h.update(values.tobytes())
    for fold in folds:
        h.update(str(fold['fold']).encode('ascii'))
        for name in ('train_mask', 'validation_mask'):
            arr = fold[name].to_numpy(dtype=bool)
            h.update(np.packbits(arr).tobytes())
    return h.hexdigest()


def _check_record(record, *, expected_counts, configs_count):
    assert set(record) == set(FIELDS), f'Checkpoint record fields incorrect: {record}'
    index = record['config_index']
    fold = record['fold']
    assert type(index) is int and 0 <= index < configs_count
    assert type(fold) is int and 1 <= fold <= 3
    assert (record['train_rows'], record['validation_rows']) == tuple(expected_counts[fold - 1])
    for name in ('validation_mae', 'validation_rmse', 'elapsed_seconds'):
        assert isinstance(record[name], (float, int)) and np.isfinite(record[name])
        assert record[name] >= 0 if name != 'elapsed_seconds' else record[name] > 0
    return (index, fold)


def _ranking(records, configurations):
    if len(records) != len(configurations) * 3:
        return None
    fold_frame = pd.DataFrame(records)
    mean = (fold_frame.groupby('config_index', sort=True)
            [['validation_mae', 'validation_rmse']].mean())
    assert mean.index.tolist() == list(range(len(configurations)))
    rank = mean.rename(columns={'validation_mae':'cv_mae',
                                'validation_rmse':'cv_rmse'}).reset_index()
    rank = rank.sort_values(['cv_mae','cv_rmse','config_index'],kind='stable').reset_index(drop=True)
    rank.insert(0,'rank',range(1,len(configurations)+1))
    rank['folds'] = 3
    rank['configuration'] = [
        _as_json(dict(configurations[i])) for i in rank['config_index']
    ]
    return rank


def run_block(*, X, y, issue_time, valid_time, forbidden_mask,
              folds, configurations, model_factory, evaluator,
              horizon_minutes, feature_family, expected_counts,
              checkpoint_path, source_sha256, max_new_fits):
    """Fit up to max_new_fits missing candidate/fold pairs, atomically saving each.

    Fail-closed on checkpoint mismatch/corruption. No implicit reset or
    overwrite of previous scores with different protocol/data.
    """
    checkpoint_path = Path(checkpoint_path)
    if type(max_new_fits) is not int or not 1 <= max_new_fits <= 24:
        raise ValueError('max_new_fits must be an integer between 1 and 24.')
    assert len(folds) == 3 and len(configurations) == 8
    assert int(forbidden_mask.sum()) == 26496
    assert y.loc[forbidden_mask].isna().all()
    assert y.loc[~forbidden_mask].notna().all()
    assert X.index.equals(y.index) and issue_time.index.equals(X.index)
    assert valid_time.index.equals(X.index) and forbidden_mask.index.equals(X.index)
    assert len(expected_counts) == 3
    for fold_index, fold in enumerate(folds, 1):
        assert fold['fold'] == fold_index
        train = fold['train_mask']; val = fold['validation_mask']
        assert (int(train.sum()), int(val.sum())) == tuple(expected_counts[fold_index - 1])
        assert not (train & val).any()
        assert not ((train | val) & forbidden_mask).any()
        assert valid_time.loc[train].max() < issue_time.loc[val].min()
        assert valid_time.loc[val].max() < issue_time.loc[forbidden_mask].min()

    signature = fingerprint(X, y, issue_time, valid_time, forbidden_mask,
                            folds, configurations,
                            horizon_minutes=horizon_minutes,
                            feature_family=feature_family,
                            source_sha256=source_sha256)
    identity = {
        'schema_version': SCHEMA_VERSION,
        'model': 'RandomForestRegressor',
        'horizon_minutes': int(horizon_minutes),
        'feature_family': feature_family,
        'signature_sha256': signature,
        'source_sha256': source_sha256,
        'candidate_configurations': list(configurations),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    lock = checkpoint_path.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Checkpoint locked (another run, or stale lock): {lock}') from exc
    os.close(fd)
    try:
        if checkpoint_path.exists():
            state = json.loads(checkpoint_path.read_text(encoding='utf-8'))
            if any(state.get(key) != val for key, val in identity.items()):
                raise ValueError(f'Checkpoint identity mismatch; refusing reuse: {checkpoint_path}')
            records = state.get('fits')
            if not isinstance(records, list):
                raise ValueError('Checkpoint fit records missing.')
        else:
            state = dict(identity, fits=[])
            records = state['fits']
        existing = set()
        for record in records:
            key = _check_record(record, expected_counts=expected_counts,
                                configs_count=len(configurations))
            if key in existing:
                raise ValueError(f'Duplicate checkpoint fit: {key}')
            existing.add(key)

        performed = 0
        for index, config in enumerate(configurations):
            for fold in folds:
                key = (index, fold['fold'])
                if key in existing:
                    continue
                if performed >= max_new_fits:
                    break
                started = perf_counter()
                per_fold, ranked_unused = evaluator(
                    X=X, y=y, issue_time=issue_time, valid_time=valid_time,
                    horizon_minutes=horizon_minutes, folds=(fold,),
                    configurations=(dict(config),), model_factory=model_factory,
                    forbidden_mask=forbidden_mask,
                )
                duration = perf_counter() - started
                assert len(per_fold) == len(ranked_unused) == 1
                row = per_fold.iloc[0]
                assert int(row['config_index']) == 0  # local single candidate
                assert int(row['fold']) == fold['fold']
                record = {
                    'config_index': index,
                    'fold': int(row['fold']),
                    'train_rows': int(row['train_rows']),
                    'validation_rows': int(row['validation_rows']),
                    'validation_mae': float(row['validation_mae']),
                    'validation_rmse': float(row['validation_rmse']),
                    'elapsed_seconds': float(duration),
                }
                _check_record(record, expected_counts=expected_counts,
                              configs_count=len(configurations))
                records.append(record)
                # This write occurs only AFTER a successful fully checked fit.
                atomic_text(checkpoint_path, _as_json(state) + '\n')
                existing.add(key)
                performed += 1
                print(f'RF {horizon_minutes} min {feature_family} '
                      f'candidate {index+1}/8 fold {fold["fold"]}/3: '
                      f'MAE={record["validation_mae"]:.6f} kW, '
                      f'{duration:.1f}s; saved {len(existing)}/24', flush=True)
            if performed >= max_new_fits:
                break
        fold_frame = pd.DataFrame(sorted(records, key=lambda r: (r['config_index'],r['fold'])),
                                  columns=FIELDS)
        rank_frame = _ranking(records, configurations)
        return fold_frame, rank_frame, performed
    finally:
        lock.unlink(missing_ok=True)


def archive_lightgbm(*, fold_scores, rankings, selected, runtimes,
                     directory, source_sha256):
    """Archive previously computed development CV data; never run any fit."""
    assert len(fold_scores) == 360
    assert len(rankings) == 120
    assert len(selected) == 15
    assert len(runtimes) == 15
    assert set(fold_scores['fold']) == {1,2,3}
    assert set(fold_scores['horizon_minutes']) == {15,30,60}
    assert set(fold_scores['feature_family']) == {'F0','F1','F2','F3','F4'}
    assert not fold_scores.duplicated(['horizon_minutes','feature_family','config_index','fold']).any()
    assert fold_scores.groupby(['horizon_minutes','feature_family','config_index']).size().eq(3).all()
    assert rankings.groupby(['horizon_minutes','feature_family']).size().eq(8).all()
    assert selected.groupby(['horizon_minutes','feature_family']).size().eq(1).all()
    assert np.isfinite(fold_scores[['validation_mae','validation_rmse']]).all().all()
    directory = Path(directory)
    data = [('lightgbm_fold_scores.csv',fold_scores),
            ('lightgbm_candidate_rankings.csv',rankings),
            ('lightgbm_selected.csv',selected),
            ('lightgbm_runtimes.csv',runtimes)]
    manifest = {'kind': 'development-only CV metrics',
                'note':'No predictions, raw targets, or terminal scores.',
                'source_sha256':source_sha256,'files':{}}
    archive_contents = []
    for filename, frame in data:
        content = frame.to_csv(index=False, float_format='%.17g')
        manifest['files'][filename] = {'rows':len(frame),
            'sha256':hashlib.sha256(content.encode('utf-8')).hexdigest()}
        archive_contents.append((filename,content))
    archive_contents.append(('manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n'))
    # Check all existing files before writing any, avoiding partial conflicts.
    for filename,content in archive_contents:
        path = directory / filename
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Development archive exists with different content: {path}')
    for filename,content in archive_contents:
        atomic_text(directory / filename,content,refuse_different=True)
    return manifest
