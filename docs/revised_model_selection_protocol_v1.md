# SOLETE revised model-selection and final-refit protocol (v1)

**Status:** Proposed lock, **not committed**. Baseline: `a55524c` (completed Notebook 12). Prepared: 2026-10-22.

## Scope

This protocol fixes **development-only** choices based on the frozen 1,665 candidate-fold evaluations from revised Notebooks 10–12. It creates **no refits, predictions, holdout scores, or model tuning**. It is intended to be committed before any subsequent evaluation.

### Selected primary models (one per horizon)

| Horizon | Architecture | Features | Configuration | CV MAE (kW) | CV RMSE (kW) |
|---|---|---|---|---:|---:|
| 15 min | LightGBM | F1 | C1 (`learning_rate=.03`, `n_estimators=300`, `num_leaves=15`) | 0.177400 | 0.451460 |
| 30 min | LightGBM | F2 | C1 (`learning_rate=.03`, `n_estimators=300`, `num_leaves=15`) | 0.219329 | 0.517169 |
| 60 min | LightGBM | F2 | C1 (`learning_rate=.03`, `n_estimators=300`, `num_leaves=15`) | 0.280951 | 0.612719 |

Selection rule: select the minimum unweighted mean of the **three outer-fold MAEs** among all recorded configurations and feature families per horizon. No new tolerance or significance-based switching rule is introduced. The 555-way search creates selection optimism; CV MAE is not an unbiased prospective test estimate.

### Second-place challengers and stability

| Horizon | Runner-up | Mean CV MAE (kW) | Runner minus winner (kW) | RF-favored folds |
|---|---|---:|---:|---|
| 15 min | Random Forest F2 C3 | 0.177618 | 0.000218 | 1 |
| 30 min | Random Forest F1 C8 | 0.225092 | 0.005763 | none |
| 60 min | Random Forest F1 C8 | 0.286863 | 0.005912 | 1 |

The 15-minute lead is only ~0.000218 kW (0.123% of RF CV MAE), and Random Forest wins fold 1. The 60-minute RF likewise wins fold 1. Three aggregated folds do not support a significance test. We retain RF as **descriptive challenger**, not as an additional model chosen after seeing later outcomes.

## Evidence and exact provenance

- Verified completed synthesis: `notebooks/12_revised_development_synthesis.ipynb`, commit `a55524c`; full classical and neural development archives.
- Winning LightGBM fold metrics: `results/development_only/revised_classical_cv_v1/lightgbm/lightgbm_fold_scores.csv`; its `manifest.json` has SHA-256 and row-count expectations.
- Challengers: 15 `results/development_only/revised_classical_cv_v1/random_forest/h{horizon}_{family}.json` archives; specific files are `h15_F2.json`, `h30_F1.json`, and `h60_F1.json`.
- Frozen estimator implementation: `src/classical_models.py:build_lightgbm_model`. The JSON companion includes the exact nine module Git-blob SHAs, each fold vector, fixed LightGBM parameters, and source paths.
- Version reference: `pyproject.toml` pins Python 3.12, LightGBM 4.7.0, scikit-learn 1.9.0, NumPy 2.0.2 and pandas 3.0.5. Verify the local environment against these pins before a real refit.

## Future final refit (planned only; NOT RUN)

1. Reconstruct F1 for 15 minutes and F2 for 30/60 minutes using **exactly** the Notebook 10 feature construction and source conventions. F1 contains Gregorian calendar features; F2 adds deterministic solar geometry. Never add F3 or F4 to these three primary candidates after seeing later scores.
2. Use the frozen `src.forecasting_protocol.make_terminal_refit_mask(valid_time)` and check the exact development training counts: **104,817 / 104,814 / 104,808** for horizons 15/30/60 respectively. The last admissible target-valid timestamp is **strictly before 2019-05-31 23:00 UTC** (60-minute embargo before the June 1 test boundary).
3. Fit separate `src.classical_models.build_lightgbm_model({"n_estimators":300,"learning_rate":0.03,"num_leaves":15})` estimators with the unmodified `LIGHTGBM_FIXED_PARAMS` (seed 42). Fit once per horizon on its **entire admissible development population**, **without early-stopping or validation on the historical terminal period**. No additional tuning, adaptation, or feature selection is permitted.
4. Save refitted model artifacts with fitted-source commit, training row count, full candidate specification, package versions, input schema/hash, timestamps, fitted model hash, training and inference runtime, and a clear `terminal_evaluated=false` flag. This is a proposal for the next gated execution, not permission to run it.
5. A subsequent authorized evaluation must use fixed forecasts and predetermined scoring; report MAE/RMSE in kW, compare with predeclared baselines and descriptive challenger CV metrics **without changing the winner**. Do not treat the previously inspected June–August 2019 window as an untouched confirmatory test.

## Holdout integrity, interpretation, and next stage

- June–August 2019 labels were historically evaluated in the earlier Notebook 11 project lineage. Revised Notebooks 10–12 excluded them from the **new CV and model-selection** stages, but a future reuse would be a **historically exposed retrospective evaluation**, not blind independent confirmation.
- For a stronger confirmatory claim, plan truly new data or a rigorously independent external dataset with compatible feature availability, deployment assumptions and station-wise validation; PVOD is a distinct domain, not a plug-in substitute for a fresh SOLETE time block.
- The fact that F1/F2 win for the overall forecasting model does **not** prove that lunar/Panchang representations never help. Analyze matched F3→F4 effects for predeclared configurations and appropriate developmental or external evidence, separate from selection of the primary forecasting model.
- No existing `src/*.py`, historic Notebook 12–15, neural checkpoint, or test artifact should be altered to enact this protocol.

## Acceptance gates

- [ ] Place JSON under `results/development_only/revised_model_selection_lock_v1.json` and verify with the companion read-only verifier.
- [ ] Review the 15-minute near tie and formally accept the *strict minimum mean MAE* decision rule; do not adopt a post-hoc tie tolerance.
- [ ] Run `python -m pytest -q` (last reported 272 passed), stage the protocol and lock, inspect diff, commit/push.
- [ ] Record the commit SHA as the effective lock baseline; no downstream training is authorized by the commit alone.
- [ ] Plan revised Notebook 13 as a separate **read-only selection-lock / refit preflight**, with execution gate disabled; reserve feature-mechanism studies for a subsequent distinct stage.

**Disposition:** This is a proposed development-selection lock suitable for review and commitment, **not evidence that terminal refit or validation has occurred**.
