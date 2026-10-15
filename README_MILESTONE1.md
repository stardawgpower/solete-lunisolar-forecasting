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
