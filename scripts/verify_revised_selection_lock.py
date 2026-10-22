#!/usr/bin/env python3
"""Read-only proposed selection-lock preflight for SOLETE revised CV.

Install within project as scripts/verify_revised_selection_lock.py, and the lock as
results/development_only/revised_model_selection_lock_v1.json. From repo root:
    python scripts/verify_revised_selection_lock.py

No imports from training modules, no model fitting, no test/holdout data access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPECTED_FOLD_COUNTS = {
    15: ((52401, 17856), (70257, 16992), (87249, 17556)),
    30: ((52398, 17856), (70254, 16992), (87246, 17556)),
    60: ((52392, 17856), (70248, 16992), (87240, 17556)),
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, label: str) -> None:
    check(math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), abs_tol=1e-12, rel_tol=0),
        f"{label}: {actual} != expected {expected}")


def git_file_sha(root: Path, relative: str) -> str:
    p = subprocess.run(['git', 'hash-object', relative], cwd=root, check=True,
                       capture_output=True, text=True)
    return p.stdout.strip()


def verify(root: Path, lock_file: Path) -> None:
    root, lock_file = root.resolve(), lock_file.resolve()
    check((root / '.git').exists(), f'Expected Git checkout: {root}')
    check(lock_file.is_file(), f'Lock JSON not found: {lock_file}')
    lock = json.loads(lock_file.read_text(encoding='utf-8'))
    check(lock['freeze_state'] == 'PROPOSED_LOCK_NOT_YET_COMMITTED',
          'Unexpected lock state; audit before adopting changes')
    check(lock['archived_audit']['archived_fold_scores'] == 1665,
          'Unexpected campaign size')
    check(lock['evaluation_policy']['terminal_access_authorized'] is False,
          'Terminal evaluation must stay disabled')
    check(lock['evaluation_policy']['extra_training_authorized'] is False,
          'New training must stay disabled')
    p = subprocess.run(['git', 'merge-base', '--is-ancestor',
                        lock['baseline_immutable_commit'], 'HEAD'],
                       cwd=root, check=False)
    check(p.returncode == 0, 'Baseline source commit must be an ancestor of HEAD')

    for relative, expected_sha in lock['reproducibility']['git_blob_shas'].items():
        check(git_file_sha(root, relative) == expected_sha,
              f'Frozen source changed: {relative}')

    base = root / 'results/development_only/revised_classical_cv_v1'
    # Byte-level archive checks, as recorded by the original model campaign.
    all_fold_tables = {}
    for key, expected_size in [('ridge', 225), ('lightgbm', 360)]:
        p = base / key
        manifest = json.loads((p / 'manifest.json').read_text())
        for name, item in manifest['files'].items():
            actual = (p / name).read_bytes()
            actual_sha = hashlib.sha256(actual).hexdigest()
            check(actual_sha == item['sha256'], f'Archived {name} SHA256 mismatch')
            check(len(pd.read_csv(p / name)) == item['rows'],
                  f'Archived {name} row-count mismatch')
        df = pd.read_csv(p / f'{key}_fold_scores.csv')
        check(len(df) == expected_size, f'{key}: expected {expected_size} fold scores')
        all_fold_tables[key] = df

    # Random-Forest archives: 15 blocks x 8 configurations x 3 folds.
    rf_source = ('src/classical_models.py', 'src/revised_classical_cv.py',
                 'src/forecasting_protocol.py', 'src/revised_rf_checkpoint.py')
    source_sha = hashlib.sha256(b''.join((root / s).read_bytes() for s in rf_source)).hexdigest()
    rf_blocks = {}
    for h in (15, 30, 60):
        for family in ('F0', 'F1', 'F2', 'F3', 'F4'):
            path = base / 'random_forest' / f'h{h}_{family}.json'
            d = json.loads(path.read_text())
            check(d.get('source_sha256') == source_sha,
                  f'RF source fingerprint mismatch: {path}')
            check(d['horizon_minutes'] == h and d['feature_family'] == family,
                  f'RF archive identity mismatch: {path}')
            check(len(d['candidate_configurations']) == 8 and len(d['fits']) == 24,
                  f'RF archive incomplete: {path}')
            unique = {(r['config_index'], r['fold']) for r in d['fits']}
            check(unique == {(c, f) for c in range(8) for f in (1, 2, 3)},
                  f'RF archive missing/duplicate config-fold: {path}')
            for row in d['fits']:
                check((row['train_rows'], row['validation_rows']) ==
                      EXPECTED_FOLD_COUNTS[h][row['fold']-1],
                      f'RF fold-count mismatch: {path}')
            rf_blocks[(h, family)] = d
    check(len(rf_blocks) == 15, 'Expected 15 RF archives')

    # The neural archive count was source/data audited in Notebook 12. Here the
    # defensive check is only for archive presence/count (no training imports).
    counts = {}
    for name, expected_count in [
        ('revised_neural_cv_v1', 72),
        ('revised_neural_cv_v2_year_std1', 288),
        ('revised_recurrent_extension_v1_F0', 72),
        ('revised_recurrent_extension_v2_year_std1', 288),
    ]:
        folder = root / 'results/development_only' / name
        counts[name] = len(list(folder.glob('*.json')))
        check(counts[name] == expected_count,
              f'Neural archive incomplete: {name} {counts[name]}/{expected_count}')
    check(225 + 360 + 15*24 + sum(counts.values()) == 1665,
          'Unified development campaign is incomplete')

    # Strong neural source/schema/data-identity audits; all are strictly read-only.
    # Expose project root for `python scripts/verify...py` execution from a checkout.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.revised_neural_campaign import campaign_source_sha256 as v1_source
    from src.revised_neural_campaign_v2 import campaign_source_sha256 as v2_source
    from src.revised_neural_batch import _audit_prior_source
    from src.revised_neural_batch_v2 import _audit_v2, read_mixed_progress
    from src.revised_recurrent_extension import (
        read_progress as read_recurrent_progress, _verify_existing_data,
    )

    dev = root / 'results/development_only'
    original_v1 = dev / 'revised_neural_cv_v1'
    original_v2 = dev / 'revised_neural_cv_v2_year_std1'
    check(_audit_prior_source(original_v1, source_sha256=v1_source(root)) == 72,
          'Neural original F0 source/schema check failed')
    check(_audit_v2(original_v2, v2_source(root)) == 288,
          'Neural original F1-F4 source/schema check failed')
    check(len(read_mixed_progress(original_v1, original_v2)) == 360,
          'Neural original 360/360 check failed')
    recurrent = read_recurrent_progress(root, verify_source=True)
    check(len(recurrent) == 360, 'Recurrent extension 360/360 check failed')
    _verify_existing_data(root, recurrent)

    wins = lock['selected_candidates']
    check(len(wins) == 3 and {w['horizon_minutes'] for w in wins} == {15,30,60},
          'Exactly one winner for each horizon is required')
    for w in wins:
        h, family = w['horizon_minutes'], w['feature_family']
        cidx = w['configuration_index_zero_based']
        check(w['model'] == 'LightGBM' and cidx == 0 and
              w['configuration'] == {'learning_rate': .03, 'n_estimators': 300, 'num_leaves': 15},
              f'Unexpected winner config at horizon {h}')
        part = all_fold_tables['lightgbm'].loc[
            (all_fold_tables['lightgbm']['horizon_minutes'] == h) &
            (all_fold_tables['lightgbm']['feature_family'] == family) &
            (all_fold_tables['lightgbm']['config_index'] == cidx)
        ].sort_values('fold')
        check(list(part['fold']) == [1,2,3], f'Missing winner folds: {h}')
        info = w['cv_selection']
        for j, row in enumerate(part.itertuples(index=False)):
            check((row.train_rows,row.validation_rows) == EXPECTED_FOLD_COUNTS[h][j],
                  f'Winner fold population differs: {h}, fold {j+1}')
            close(row.validation_mae, info['fold_mae_kw'][j], f'LightGBM {h} fold {j+1} MAE')
            close(row.validation_rmse, info['fold_rmse_kw'][j], f'LightGBM {h} fold {j+1} RMSE')
        close(part['validation_mae'].mean(), info['unweighted_fold_mean_mae_kw'],
              f'LightGBM {h} mean MAE')
        close(part['validation_rmse'].mean(), info['unweighted_fold_mean_rmse_kw'],
              f'LightGBM {h} mean RMSE')
        challenger = w['challenger_descriptive_only']
        rf = rf_blocks[(h, challenger['feature_family'])]
        check(rf['candidate_configurations'][challenger['configuration_index_zero_based']] ==
              challenger['configuration'], f'RF challenger config identity: {h}')
        rf_folds = sorted((r for r in rf['fits'] if r['config_index'] ==
                           challenger['configuration_index_zero_based']),
                          key=lambda row: row['fold'])
        check([r['fold'] for r in rf_folds] == [1,2,3], f'Incomplete RF challenger: {h}')
        for j, row in enumerate(rf_folds):
            close(row['validation_mae'], challenger['fold_mae_kw'][j],
                  f'RF {h} fold {j+1} MAE')
            close(row['validation_rmse'], challenger['fold_rmse_kw'][j],
                  f'RF {h} fold {j+1} RMSE')
        rf_mean = sum(row['validation_mae'] for row in rf_folds)/3
        close(rf_mean, challenger['unweighted_fold_mean_mae_kw'], f'RF {h} mean MAE')
        close(rf_mean-part['validation_mae'].mean(), challenger['winner_margin_kw'],
              f'Margin {h}')
        print(f"{h:2d} min: LightGBM {family} C1  MAE={part['validation_mae'].mean():.6f} kW  "
              f"RF challenger MAE={rf_mean:.6f} kW")

    print('MODEL-SELECTION LOCK PREFLIGHT: PASSED — source and archive evidence consistent')
    print('1,665 / 1,665 candidate-fold evaluations source/identity audited')
    print('NO MODEL TRAINING / NO TEST DATA ACCESS / NO OUTPUT FILE WRITES')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd(),
                    help='Git repository root (default: current directory)')
    ap.add_argument('--lock', type=Path, default=None,
                    help='selection-lock JSON (default: <root>/results/development_only/revised_model_selection_lock_v1.json)')
    args = ap.parse_args()
    root = args.root.resolve()
    lock = args.lock or (root / 'results/development_only/revised_model_selection_lock_v1.json')
    verify(root, lock)


if __name__ == '__main__':
    main()
