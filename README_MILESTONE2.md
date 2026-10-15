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
