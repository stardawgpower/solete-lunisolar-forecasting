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
