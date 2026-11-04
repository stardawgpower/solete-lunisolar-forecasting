# Evaluating Categorical Lunisolar Features for Short-Term Photovoltaic Forecasting

Research code and experimental results for evaluating whether **Panchang (lunisolar calendar) features** improve short-term photovoltaic (PV) power forecasts beyond historical observations, conventional calendar features, and continuous solar and lunar geometry.

The main comparison is **F3 versus F4**: does adding categorical Panchang features improve forecasts when the model and evaluation observations are held fixed?

## Study overview

| Item | Design |
|---|---|
| Datasets | SOLETE (Denmark) and 10 PVOD stations (China) |
| Forecast horizons | 15, 30, and 60 minutes |
| Models | Ridge, Random Forest, LightGBM, LSTM, GRU, BiLSTM, and TCN |
| SOLETE model selection | Three expanding chronological cross-validation folds, with a 60-minute training-label guard |
| Feature comparisons | Fixed LightGBM configuration with matched observations |
| Evaluation | MAE, RMSE, daylight-only results, and paired bootstrap intervals |

The SOLETE architecture comparison covers 555 candidate configurations and 1,665 candidate-fold evaluations across the three expanding folds.

### Feature families

Each feature family includes all features from the preceding family.

| Family | Added information | SOLETE features | PVOD features |
|---|---|---:|---:|
| F0 | Historical PV power and meteorology | 118 | 71 |
| F1 | Gregorian calendar and cyclical time | 133 | 86 |
| F2 | Solar geometry | 138 | 91 |
| F3 | Continuous lunar and Sun–Moon geometry | 147 | 100 |
| F4 | 97 Panchang indicators: Tithi, Paksha, Karana, Nakshatra, and Yoga | 244 | 197 |

Measured predictors use information available by forecast issue time; deterministic calendar and astronomy features use forecast-valid time. SOLETE astronomy uses a validated **+2-minute** timestamp convention; PVOD uses recorded UTC timestamps. Panchang sidereal coordinates follow the documented Lahiri convention.

## Main findings

- LightGBM achieved the lowest SOLETE development errors with **F1 at 15 minutes** and **F2 at 30 and 60 minutes**.
- Adding Panchang categories produced small, horizon-dependent improvements in some SOLETE comparisons, but **no consistent aggregate benefit on PVOD**.
- Results depend on model architecture, forecast horizon, astronomical representation, and temporal alignment. More features do not automatically improve accuracy.

These are empirical forecasting comparisons, **not evidence that Panchang categories physically cause changes in PV output**. The later SOLETE and PVOD evaluations were historically exposed and are not claimed as untouched blind tests.

## Repository structure

| Path | Purpose |
|---|---|
| `notebooks/01_*.ipynb`–`05_*.ipynb` | SOLETE data preparation, astronomy, Panchang features, and forecasting protocol |
| `notebooks/06_*.ipynb`–`07_*.ipynb` | Audit completed classical and neural development-CV experiments |
| `notebooks/08_*.ipynb`–`12_*.ipynb` | SOLETE feature comparisons, controls, retrospective evaluation, and architecture synthesis |
| `notebooks/13_*.ipynb`–`16_*.ipynb` | PVOD preparation and evaluation, followed by cross-dataset synthesis |
| `src/` | Reusable forecasting, model, astronomy, calendar, and evaluation code |
| `tests/` | Automated tests for implementation and scientific conventions |
| `scripts/` | Development-selection and final-model verification utilities; retained tested milestone helpers |
| `results/` | Completed experiment outputs and reproducibility records |
| `requirements/research-freeze-direct.txt` | Historical record of pinned direct dependencies |
| `pyproject.toml`, `.python-version` | Package configuration and Python version |

Important results are organized under `results/model_architectures/`, `results/feature_comparison/`, `results/representation_controls/`, `results/solete_retrospective/`, `results/solete_architecture_synthesis/`, `results/pvod_feature_comparison/`, and `results/cross_dataset_synthesis/`. The `results/development_only/` directory preserves model-selection and chronological-CV evidence; it is **not** an instruction to retrain the historical experiments.

## Installation

**Python 3.12.14** is the recorded and tested environment. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ml,dl,notebook,dev]'
```

The full editable installation and all applicable pinned direct dependency versions were verified in a **fresh Python 3.12.14 virtual environment on macOS**. `pyproject.toml` is the installation specification. `requirements/research-freeze-direct.txt` documents historical direct versions but is **not a complete transitive lock file**.

### Required external files

The datasets and astronomical ephemeris are **not distributed in this repository**. Place them under the following paths:

```text
solete_dataset/
  SOLETE_Pombo_5min.h5
external_data/PVOD/
  metadata.csv
  station00.csv ... station09.csv
ephemeris/
  de440s.bsp
```

- SOLETE dataset: https://doi.org/10.11583/DTU.17040767
- PVOD dataset: https://doi.org/10.11922/sciencedb.01094

Additional SOLETE resolutions are useful for some data checks. The notebook code contains the precise input and population checks for each experiment.

## Verification

Check the installation and run the automated tests:

```bash
python -m pip check
python -m pytest -q
```

**270 automated tests passed** in the existing research environment and in a separate fresh virtual environment (November 2026). This checks software behavior; it does not retrain the full forecasting campaign.

If you have the **locally retained fitted LightGBM files**, you can also run the read-only final-refit integrity audit:

```bash
python scripts/verify_all_three_final_refits.py
```

The audit verifies the recorded 15-, 30-, and 60-minute final model files, metadata, model-selection lock, and source fingerprints. It does not retrain, score, or independently recompute the original input-data hashes. Some fitted model files are excluded from GitHub, so this audit **may not pass from a source-only clone**.

Ruff is available for optional lint diagnostics (`python -m ruff check .`), but the existing repository is **not lint-clean**; linting is not presented as a passing reproducibility check.

## Reproducibility scope

The notebook sequence, completed results, and development archives document the study. SOLETE development model selection used three chronological folds, while PVOD used station-specific chronological splits. The historical model-development campaign and all original fitted models are not automatically recreated by running the tests. Neural F0 experiments used a different scaling convention from F1–F4, so F0–F1 neural comparisons require caution.

Some integrity checks depend on frozen source-file fingerprints. Avoid editing frozen scientific modules solely to satisfy formatting tools without first assessing the effect on result provenance.

Earlier experimental stages remain in Git commit history; the active repository contains the unified 16-notebook workflow.
