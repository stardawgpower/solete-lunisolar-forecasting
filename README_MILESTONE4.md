# SOLETE Revised Neural Notebook 11 — Milestone 4

**Status:** development-only, per-candidate/fold CV infrastructure; **no training runs automatically**. Keep the historical `notebooks/11_deep_learning_models.ipynb` unchanged. Do not overwrite any committed classical-model results.

## Files

- `src/revised_neural_campaign.py`: general 15/30/60 minute, F0–F4, 3-fold LSTM/TCN training with 4 frozen candidates each, inner early stopping, fresh outer-only refit, atomic per-fit metrics checkpoint.
- `tests/test_revised_neural_campaign.py`: synthetic chronology, holdout exclusion, immutable restart and ranking tests; no real TensorFlow training.
- `scripts/append_neural_m4.py`: additive idempotent extension of your **current** executed Notebook 11. Preserves all previous cells and saved output. Also handles the extra audit scratch cells after Milestone 3.

The updated notebook is also available as a separate **preview file** named `11_revised_neural_models_milestone4.ipynb`. Prefer the additive script below to preserve any notebook work done since the preview was generated.

## Install safely in the project root

1. Save and **close** the existing Notebook 11 editor and stop its kernel before editing on disk.
2. Back up your executed notebook.
3. Extract the ZIP with `unzip -n` (ZIP has no notebook replacement).
4. Run the new regression tests and `python scripts/append_neural_m4.py`.

```sh
cp notebooks/11_revised_neural_models.ipynb ~/Downloads/11_revised_neural_models_m3_backup.ipynb
unzip -n ~/Downloads/revised_neural_notebook11_milestone4.zip -d .
python -m pytest -q tests/test_revised_neural_campaign.py
python -m pytest -q
python -m compileall -q src scripts
python scripts/append_neural_m4.py
```

Expected: `MILESTONE 4 APPENDED: PASSED`, with **campaign disabled**. Re-running the append script will print `MILESTONE 4 ALREADY INSTALLED` without changing the notebook.

## How to run the preflight without expensive retraining

- Open Notebook 11 with the same `.venv` kernel.
- Because Milestone 3 has finished, you should only execute the new **Milestone 4 preflight, policy, and run** cells after the previously defined canonical arrays and metadata have been loaded in the current kernel. In a fresh kernel, first run the earlier canonical data-building/preflight cells. Keep the prior Milestone 3 gates switched off; their checkpoints already exist.
- The preflight verifies both previously saved Milestone 3 JSON files against **their own source and data fingerprints**. It does not reuse or copy these M3 feasibility records into the distinct M4 training archive.
- The policy defines `M4_RUN_ONE=False` by default. The training cell must print `NEURAL MILESTONE 4 TRAINING: DISABLED`.

**Do not switch to True or launch all 360 evaluations yet.** Await a first real CV checkpoint evaluation/audit. When the user later explicitly opts in, one invocation can execute at most one candidate/fold and saves its completed kW MAE and RMSE. Partial/interrupted fits are not counted and are retried from the beginning.

## Data and provenance

- Every payload is created by `prepare_development_fold` from **terminal-masked** labels and an explicit revised forecast mask. Strict counts, five-minute grid, horizon, CV boundaries, inner 60-minute availability embargo, and F0–F4 tensor dimensions are checked.
- Inner early-stopping uses only an inner split of outer-training rows; its scaler is fitted on inner training only.
- A new scaler/model is fitted on all outer-training labels for `best_epoch` epochs; the outer validation labels are evaluated only **after** prediction. No June–August 2019 terminal labels enter this system.
- Each completed evaluation is saved under `results/development_only/revised_neural_cv_v1/` with model, candidate, horizon, family, fold, versions, and source/data hashes. A matching checkpoint is reused; a changed identity or malformed result aborts.
- A block is eligible for ranking only when all **4 configurations × 3 folds** are complete for that model/horizon/family. Exact MAE ties resolve by configuration index, not RMSE.
- The two previously completed M3 gate metrics are archived *separately*, as feasibility evidence, and are not asserted equivalent to the new M4 code.
- The full 360-evaluation campaign and terminal test remain unstarted.

## Expected new test count

12 new M4 tests; previous repository suite had 207, so expect **219 tests** total if the repository/environment is unchanged.
