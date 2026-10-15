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
