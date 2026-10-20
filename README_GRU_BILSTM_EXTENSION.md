# Revised Notebook 11 — GRU and BiLSTM extension

This package adds two recurrent architectures to the completed LSTM/TCN development-CV study. The first 360 evaluations remain unchanged; the extra 360 fits have their own checkpoints.

## Scope

- Models: GRU and BiLSTM.
- Horizons: 15, 30, 60 minutes.
- Feature sets: F0, F1, F2, F3, F4.
- Four **fixed historical LSTM-shaped recurrent configurations** per model (C1–C4), tested separately for each model and feature/horizon setting. This is a like-for-like structural comparison, not use of historical test-selected winners.
- Three expanding chronological CV folds per setting.
- F0 uses the existing frozen V1 train-only scaling/fit implementation; F1–F4 use the V2 year-safe train-only implementation.
- Inner epoch selection, 60-minute training-label embargo, outer refit, seed 42 and MAE evaluation use the **same frozen code objects** as the LSTM/TCN campaign. The extension builds its own GRU and BiLSTM models but leaves original modules unchanged.
- The terminal June–August 2019 holdout is never used by the extension.

## Installation (from project root)

**Create a safety backup/commit of your current state first.** Unzip this package into your project root; it adds new files under `src/`, `tests/` and `tools/` without overwriting existing files.

```bash
python tools/install_recurrent_extension.py
python tools/install_recurrent_extension.py --apply
python -m pytest -q
```

The first command is a dry run. The second appends five cells to `notebooks/11_revised_neural_models.ipynb` and saves an adjacent backup named `11_revised_neural_models_before_recurrent_extension_backup.ipynb`. Do not stage the backup automatically; keep it as a local recovery copy.

Now open Notebook 11 and select **Restart Kernel + Run All** **once** to confirm the read-only path. The new batch switch defaults to `EXT_RUN_BATCH = False`.

## First two feasibility fits

Run only the extension batch cell at the end of Notebook 11, first with:

```python
EXT_RUN_BATCH = True
EXT_MAX_NEW_FITS = 1
EXT_MODELS = ("GRU",)
```

Wait for a saved GRU C1 / h15 / F0 / fold1 checkpoint, then restore `EXT_RUN_BATCH = False` and save the notebook. On another run, use:

```python
EXT_RUN_BATCH = True
EXT_MAX_NEW_FITS = 1
EXT_MODELS = ("BiLSTM",)
```

These are **full nested-CV single-fold fits**, not shortened pilot scores. If either is unstable, investigate it before large batches. After both succeed, continue in batches of 10–15 new fits and restore the switch to False between batches. Re-running the batch cell skips checkpointed work.

Result directories:

```
results/development_only/revised_recurrent_extension_v1_F0/
results/development_only/revised_recurrent_extension_v2_year_std1/
```

Maximum count 360 new JSON records; 720 combined with the original completed 360.

## Validation and reproducibility

The installer deliberately does **not** edit any existing `src/` module or training checkpoint. Historical experiment source fingerprints remain valid.

Each new checkpoint includes: immutable model/configuration identity; original canonical sample payload SHA; frozen-source plus extension fingerprint; scaling-policy tag for F1–F4; fold sizes, best epoch, training timings, MAE and RMSE. Prior to any new fit the runner checks that the canonical payload has the same data hash as the original LSTM run for that horizon, family and fold.

Even when disabled, the extension confirms that the original 360 checkpoint grid remains present, reads new archive records, and checks their provenance. It does not import TensorFlow or create directories in disabled mode.

**Scientific limitation:** This package was validated for Python syntax, notebook structure and idempotent notebook installation in the authoring environment, but could not be run end-to-end on the project's local SOLETE arrays or Apple M2/TensorFlow installation. The first local test and single-fit logs are required before proceeding.

## Git checkpoint

Once read-only Run All and tests pass, inspect the diff before staging. Commit only the new module, test, installer and revised notebook. As folds accumulate, commit their new development-only JSON files; do not stage backup notebooks or modify the original 360 files.
