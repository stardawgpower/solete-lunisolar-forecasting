# Revised SOLETE Neural CV — Milestone Notes

This is a lossless consolidation of the former root-level `README_MILESTONEN.md` notes.
Original source names and SHA-256 digests are retained for audit. See Git history for the independently committed originals.


---

## Milestone 1 — original `README_MILESTONE1.md`

Source SHA-256: `863e4b19f5a4d179f3051649dedc79bcaa823d2f5a12fd89e929853546405686`

# Revised SOLETE Neural Notebook 11 — Milestone 1

This package adds a new development-only neural forecasting notebook. It does **not** overwrite the completed historical `notebooks/11_deep_learning_models.ipynb` or the completed revised Notebook 10.

## Files to add in your repository

- `notebooks/11_revised_neural_models.ipynb` — parity and scaling preflight; **no model fitting**.
- `src/revised_neural_cv.py` — neural 13×9 sequence/auxiliary builder and fail-closed development-fold materializer.
- `tests/test_revised_neural_cv.py` — 11 synthetic regression tests for chronology, F0–F4 dimensions, holdout exclusions, and binary features.

## Run from the project root

Copy the three files into the corresponding folders without overwriting any existing notebooks or sources. Then run:

```sh
python -m pytest -q tests/test_revised_neural_cv.py
python -m pytest -q
python -m compileall -q src
```

Open `notebooks/11_revised_neural_models.ipynb` with the pinned project's Python 3.12 environment and execute its 11 code cells in order. The full SOLETE HDF5 (`solete_dataset/SOLETE_Pombo_5min.h5`), DE440s (`ephemeris/de440s.bsp`), Notebook 09 and the project source modules must be present as for revised Notebook 10.

Expect a final line:

```
NEURAL MILESTONE 1: PASSED — no model fits, predictions, terminal evaluation or writes
```

If any check fails, stop and share the traceback. This notebook rebuilds astronomical feature tables from scratch; the local dataset run is *not* replicated by the synthetic tests.

## Scientific boundaries

- Development CV: 3 expanding folds and 60-minute label-availability guard.
- Input: 13 time steps × 9 observed channels; auxiliary widths F0=1, F1=16, F2=21, F3=30, F4=127.
- Binary auxiliary features (not standardized): `valid_utc_is_weekend`, `valid_solar_daylight`, 97 Panchang indicators.
- Holdout June–August 2019: exactly 26,496 terminal rows; none enter the fold tensors.
- Stage only after notebook preflight and tests pass. Do **not** rerun or replace the historical notebooks.

The next milestone will define a revised LSTM and TCN architecture/epoch-selection grid and checkpoint policy **before** any neural training; no historical test-selected configurations will be treated as new winners.


---

## Milestone 2 — original `README_MILESTONE2.md`

Source SHA-256: `546520a00e5e3afc970c5ba0d151a7f5de75e9fd345fd0b53a5228365d8971c9`

# SOLETE revised Notebook 11 — Milestone 2 (resource feasibility only)

The historical notebook `notebooks/11_deep_learning_models.ipynb` and all of revised Notebook 10 are unchanged. This milestone extends the already-executed revised Notebook 11 from GitHub commit `3750b9a` with an inner chronological early-stopping split and **two 1-epoch capped, checkpointed resource pilots**; full three-fold neural-CV training and terminal testing are NOT run here.

## Install from the repository root

**First close the revised Notebook 11 editor and stop its kernel** (to avoid VS Code/Jupyter writing an older copy over the updated file). Keep a backup before overwriting. Verify `git status --short` is clean (or reconcile your local changes) and that commit `3750b9a` is current.

```sh
cp notebooks/11_revised_neural_models.ipynb ~/Downloads/11_revised_neural_models_m1_backup.ipynb
unzip -o ~/Downloads/revised_neural_notebook11_milestone2.zip -d .
python -m pytest -q tests/test_revised_neural_training.py
python -m pytest -q
python -m compileall -q src
```

**Caution:** `unzip -o` overwrites the revised Notebook 11 as intended, so back it up first. The ZIP contains exactly the four paths shown; it does not include any historical or classical notebook.

## Notebook execution

Open `notebooks/11_revised_neural_models.ipynb` with the existing `.venv`. Run the pre-existing Milestone 1 cells through its preflight, *then* execute the two added code cells under **Milestone 2** in order. This rebuilds astronomical features if using a fresh kernel, as expected. **Only the last new code cell fits models**, and those are two capped **1-epoch pilot** fits. Do not run any historical notebook.

Expected output:

```
Inner chronology and outer-validation exclusion: PASSED
LSTM: <runtime> seconds; params=<count>; completed new pilot
TCN: <runtime> seconds; params=<count>; completed new pilot
NEURAL MILESTONE 2 RESOURCE PILOT: PASSED
```

The 2 JSON pilot checkpoints appear under `results/development_only/revised_neural_pilot_v1`. They contain only runtime/parameter summaries plus source/data fingerprints; they are NOT outer-CV scores, do not determine architecture winners, and must not be copied into the full-CV selection archive. Re-executing only the pilot cell resumes previously completed models when its exact fingerprint matches.

## Scientific selection policy for the next stage (NOT trained in M2)

Four candidate shapes each for LSTM and causal TCN, drawn from historical grids but *without reusing winners*. Three outer expanding folds with max-horizon availability gaps. For each candidate+outer fold: first select an epoch using an inner early-stopping validation tail of the outer training data (10% of its rows), separated by a 60-minute label-availability gap; fit inner-only scalers on the inner fit subset. Then rebuild a fresh model, refit scalers on **all outer training data**, fit exactly that epoch count with no outer-validation early stopping; evaluate once on the held-out outer-validation fold in kW. Aggregate by unweighted mean of all three outer-fold MAEs; ties broken by candidate grid index, not RMSE. No revised final-test access while model selection remains open.

## Stop gate

Do not commit/push until synthetic tests **and the actual one-epoch pilots** finish successfully. Send the pilot output for review. If an interrupted run changes the notebook's saved outputs, only stage the intended revised notebook, source, tests, and this milestone documentation, plus validated pilot checkpoint JSON files after reviewing identities. Leave Notebook 10 and all historical notebooks unchanged.


---

## Milestone 3 — original `README_MILESTONE3.md`

Source SHA-256: `d7a41c8da7f5b3dce4972db13538ed60869e8108204620162be0d9232888c289`

# Revised SOLETE Neural Notebook 11 — Milestone 3

## Scientific scope

Milestone 3 provides a **controlled full-size nested-CV resource gate**. It does **not** launch the 360-fit LSTM/TCN candidate search, freeze any winning architecture, refit a terminal model, or evaluate the June–August 2019 test holdout.

- Exactly one full-size fold is eligible: **15 min / F0 / fold 1** (`52,401` training rows, `17,856` outer-validation rows).
- Exactly two candidate models are eligible: `LSTM_C1` and `TCN_C1`. Train **LSTM first**; review its runtime before enabling TCN.
- Split the outer TRAIN portion: earliest 90% for inner fitting, last 10% for inner early stopping, with a 60-minute target-label-availability gap.
- Stage 1: fit inner-training-only scalers, train up to 80 epochs; EarlyStopping patience 8, min delta `1e-4`; choose the earliest epoch with the lowest observed inner-validation `val_loss` (standardized MAE).
- Stage 2: discard the trained model; fit **new scalers on all 52,401 outer-training rows**, construct a fresh same-architecture model with seed 42, train for the chosen epoch count without validation data, then predict and score the 17,856 outer-validation rows in physical kW.
- Save only counts, frozen identity/source/data SHA-256, runtimes, selected inner epoch and outer-fold MAE/RMSE to an atomic JSON file. Do not save per-row predictions or any terminal-test data.
- Default switches **both False**, so accidentally running all cells will not run the expensive fits.

A completed checkpoint is resumed without retraining when the exact source, data, TensorFlow version, protocol, and configuration match. A failed/interrupted run is **not** epoch-resumable; it will start that model again. A `.lock` file blocks concurrent runs and may require manual investigation if an interrupted kernel leaves it behind.

## Install (from the Git repository root)

Back up and close the previously executed revised Notebook 11 in VS Code/Jupyter to prevent autosave overwrites. Then:

```sh
cp notebooks/11_revised_neural_models.ipynb ~/Downloads/11_revised_neural_models_m2_backup.ipynb
unzip -n ~/Downloads/revised_neural_notebook11_milestone3.zip -d .
python -m pytest -q tests/test_revised_neural_full_cv.py
python -m pytest -q
python -m compileall -q src scripts
python scripts/append_neural_m3.py
```

The append script operates on the *local executed notebook*, preserves all 22 existing cells and their saved outputs, adds 7 cells, and changes only milestone metadata. It fails closed if Notebook 11 has unexpected content or if terminal-evaluation metadata changed. Running it twice is safe.

## Execution

1. Open `notebooks/11_revised_neural_models.ipynb` with the project's existing `.venv` kernel.
2. Running all notebook cells **with the default flags unchanged** performs the original development preflights and resumes the two short, already-checkpointed resource pilots; it **does not** train the full-size models. Check for `NEURAL MILESTONE 3 PREFLIGHT: PASSED`, and both `DISABLED` messages.
3. **Only if the preflight succeeds:** in the Milestone 3 policy cell, change `M3_RUN_LSTM = False` to `M3_RUN_LSTM = True`, keep `M3_RUN_TCN = False`, and rerun that policy cell followed by the LSTM gate cell **only**. This may be long and emits per-epoch progress.
4. Wait for `LSTM FULL-SIZE GATE: PASSED`, then share the best epoch, inner-selection seconds, outer-refit seconds, outer-fold MAE, and checkpoint path. **Do not run TCN yet**; review the LSTM wall-clock/compute costs first.
5. Do not commit while the notebook is still executing. Do not run terminal evaluation, historical selected-epoch maps, or the full CV grid.

Checkpoints (when a gate completes):

```text
results/development_only/revised_neural_fullsize_gate_v1/
  lstm_c1_h15_F0_fold1.json
  tcn_c1_h15_F0_fold1.json  # only after explicit later approval
```

These are *single-fold feasibility/correctness results*, never standalone architecture rankings.

## Package contents

- `src/revised_neural_full_cv.py` — development-only two-stage nested-CV gate and fail-closed immutable single-fold checkpoint.
- `tests/test_revised_neural_full_cv.py` — synthetic chronology, epoch, identity, leakage, fresh refit, and append tests; no TensorFlow training.
- `scripts/append_neural_m3.py` — idempotent additive notebook extension; **do not** overwrite the completed Milestone 2 notebook with a template.
- `README_MILESTONE3.md` — these instructions.

The historical `notebooks/11_deep_learning_models.ipynb` and all old scientific source files remain unchanged.


---

## Milestone 4 — original `README_MILESTONE4.md`

Source SHA-256: `4644e46f1b89aa5b9a46a9199f7de038c67348b675777061c27db9db91fc46a7`

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


---

## Milestone 5 — original `README_MILESTONE5.md`

Source SHA-256: `b6cc9430080e5f4213b8762b11dbad64cc4df8c347ec30f9f077d0697cf8905f`

# Revised SOLETE Neural Notebook 11 — Milestone 5 (bounded campaign scheduler)

**Provenance baseline:** commit `156c0cf`. This package adds a **scheduling** layer and notebook cells; the existing Milestone 4 scientific fit logic and its SHA-256 validation are untouched. Neither historical Notebook 11 nor revised Notebook 10 is modified.

## Contents

- `src/revised_neural_batch.py`: source-aware, restart-safe opt-in batch orchestration; by default 1 new fit per invocation.
- `tests/test_revised_neural_batch.py`: 18 synthetic tests covering all 360 keys, limiting, resuming, stale checkpoints and lock cleanup, with no TensorFlow training.
- `scripts/append_neural_m5.py`: additive and idempotent installer; preserves every pre-existing Notebook 11 cell and output.
- This README.

## Install from repository root

```sh
cp notebooks/11_revised_neural_models.ipynb ~/Downloads/notebook11_before_m5_backup.ipynb
unzip -n ~/Downloads/revised_neural_notebook11_milestone5.zip -d .
python -m pytest -q tests/test_revised_neural_batch.py
python -m pytest -q
python -m compileall -q src scripts
python scripts/append_neural_m5.py
```

Expected new tests: `18 passed`. The expected full suite, from the verified 220-test baseline, is **238 passed**. You should see `NEURAL MILESTONE 5 APPENDED: PASSED`. Repeating the installer should instead print `NEURAL MILESTONE 5 ALREADY INSTALLED: PASSED` and make no changes.

## Safe verification

Open the updated Notebook 11, and use **Restart Kernel → Run All** with the existing Milestone 3/4 flags OFF. Milestone 5 is also **OFF** by default:

```python
M5_RUN_BATCH = False
M5_MAX_NEW_FITS = 1
M5_MODELS = ('LSTM',)
```

This does not perform full-size fits. The new Milestone 5 cell should print `NEURAL MILESTONE 5 BATCH: DISABLED — no training executed`.

## First real campaign fit — opt in explicitly

Only after the full preflight passes and the old single-fit/gate switches are off, change the Milestone 5 cell to `M5_RUN_BATCH = True`, leave `M5_MAX_NEW_FITS = 1`, and run **ONLY the Milestone 5 scheduler cell**. Do **not** choose Run All while the batch switch is on.

The first candidate is `LSTM_C1`, horizon 15 minutes, F0, fold 1. **It must train again in the new campaign namespace** even though a single-fold Milestone 3 gate exists: the prior feasibility implementation and new campaign have different frozen provenance. Both records must remain separately documented.

Outputs are stored only in `results/development_only/revised_neural_cv_v1/`. Every successful fit is an immutable JSON checkpoint. An interruption during an unfinished fit does not write a score; resume retries that fit. Rerunning this cell with a cap of 1 will validate/reuse earlier exact-source-and-data checkpoints and then fit the next unfinished key.

If the first fit completes and the checkpoint is verified, use a gradual batch size such as `M5_MAX_NEW_FITS = 4` or `12`; only later consider `360` with `M5_MODELS = ('LSTM', 'TCN')` if machine resources allow. This may take **many hours**; do not assume the one-fold pilot timing predicts total time.

**No terminal holdout evaluation** is included. No final selection can be declared before all 3 folds of every candidate in each model/horizon/family grid are available. Full completion requires **360** campaign records and **30** complete ranking blocks (2 models × 3 horizons × 5 families), not including any Milestone 3 gate records.

Do not modify `src/revised_neural_campaign.py`, `src/revised_neural_full_cv.py` or the frozen model/protocol sources during this campaign: changed fingerprints correctly cause existing checkpoints to be rejected rather than mixed.

Do not run two batch-launching notebook kernels simultaneously. A `.neural_batch.lock` lock file prevents multiple batch launchers. If the kernel is killed abruptly, check that no process is still training before manually removing a stale lock.

Before committing this package, share the preflight result and first verified single-campaign-fit output. Do not commit an enabled training switch into the notebook.


---

## Milestone 6 — original `README_MILESTONE6.md`

Source SHA-256: `d59a4434ddcc2023e476e3178fb1166421d6722fa0a05dd2e545573e7a0bbbe4`

# Revised Neural Notebook 11 — Milestone 6: constant calendar-year train-only scaling

## Why this is necessary

The first **15-minute / F1 / fold-1** campaign evaluation failed with
`ValueError: Zero-variance or invalid continuous auxiliary feature found.`
The audited live diagnostic isolated **valid_utc_year**, auxiliary position 1.
All **inner-training** and **outer-training** values are **2018**, so their
standard deviation is *exactly zero*. This is a real constant historical
predictor, not corruption or a missing-value bug.

The frozen `src.neural_models.fit_train_only_scalers()` intentionally refuses
any zero-variance continuous predictor. **DO NOT MODIFY that source**, because
its bytes are part of both the completed Milestone 3 and the original V1 CV
source fingerprints. It also keeps the 12 completed and committed F0 records
readable and reproducible.

## Versioned, narrowly scoped correction

- `src/revised_neural_scaling_v2.py`: `fit_train_only_scalers_v2` delegates to
  the *unchanged frozen scaler* whenever the training continuous predictors
  have positive variance. Only if **valid_utc_year** is the **sole** constant
  continuous training predictor does it calculate the frozen train-only
  channel, auxiliary and target statistics with population `ddof=0`, then
  center the train calendar year and use **scale=1.0**. That means year=2018
  maps to 0 on the training window and year=2019 maps to 1 when predicted.
  All other constant predictors, invalid binary indicators, nonfinite values,
  zero-variance sequence channels, and zero-variance targets still fail.
  The outer-validation TARGETS are never used to fit any scaler.
- `src/revised_neural_campaign_v2.py`: separate schema version/kind/policy;
  retains the original nested train-only epoch selection, 60-minute embargo,
  fresh outer-train refit, MAE/RMSE metrics and candidate shapes. V2 only
  accepts F1–F4. The source digest includes both frozen V1 sources and both
  new V2 training/scaling modules.
- `src/revised_neural_batch_v2.py`: immutable mixed-source scheduler. Routes
  **F0 -> old V1 campaign/checkpoint directory** and **F1–F4 -> V2 campaign**
  in a distinct archive. Audits both sources before new training, uses the
  old V1 batch lock, and can resume one or more new fits. Existing V1 F0 JSONs
  are never promoted or rewritten as V2 data. Combined total remains /360.
- `scripts/append_neural_m6.py`: idempotently appends a disabled M6 cell to
  Notebook 11; earlier cells, stored training logs and checkpoints preserved.
- `tests/`: 26 unit/integration tests, including a real frozen-scaler
  integration test run in the user's repository. No live training in tests.

**Scientific reporting:** F0 and F1+ must retain their **distinct source
fingerprints and scaling policy names**. F0 did not need the correction;
when V2 sees nonconstant year it delegates unchanged V1 scaling. These
are not silently merged as if they came from one identical source file.
Candidate selection remains based on all 4 candidates × 3 fold MAEs within
one model/horizon/family block, never on a partial result or terminal test.

## Install (from the repo root; close Notebook 11 first)

Save the `revised_neural_notebook11_milestone6.zip` into `~/Downloads/`.

```bash
cp notebooks/11_revised_neural_models.ipynb ~/Downloads/notebook11_before_m6_backup.ipynb
unzip -n ~/Downloads/revised_neural_notebook11_milestone6.zip -d .
python -m pytest -q
python -m compileall -q src scripts
python scripts/append_neural_m6.py
```

Expected about **264 tests**, assuming no other tests were added to the
prior 238-test suite. The installer refuses to proceed if the existing M5
notebook switch is enabled: set `M5_RUN_BATCH = False`, save, and retry.
The exact archive path for the V2 records is
`results/development_only/revised_neural_cv_v2_year_std1/`.

## First step: safe verification

Open Notebook 11 with the project's `.venv` kernel. With M3, M4, M5, M6
switches all OFF, **Restart Kernel -> Run All**. The new final cell should
report:

```
MILESTONE 6 combined archived: 12 /360
MILESTONE 6 BATCH: DISABLED — no model training
```

## Controlled trial: ONE new F1 fit

Only after the safe preflight passes, edit the last M6 code cell (not M5):

```python
M6_RUN_BATCH = True
M6_MAX_NEW_FITS = 1
M6_MODELS = ('LSTM',)
```

Execute **only that M6 cell**. It verifies and reuses the 12 V1/F0
checkpoints, then trains the first new V2 15-minute/F1/C1/fold-1 evaluation.
If successful, expect a `NEW 1/1 ... V2 year-safe checkpoint saved`
message followed by `total 13/360`. Record the actual MAE; do not invent it.
Check the JSON's `identity.scaling_policy`, `identity.source_sha256`, and
`identity.data_sha256` before enabling any larger batch. Turn `M6_RUN_BATCH`
back to `False` and save the notebook before committing.

**Never** run old M5 opt-in on F1. It uses V1 scaling and will fail again.
Never edit old source modules or old checkpoint JSONs; do not retrain F0 to
"fix" a feature it doesn't include. Keep terminal June–August 2019
holdout labels excluded.
