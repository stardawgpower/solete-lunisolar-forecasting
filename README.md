# Categorical Lunisolar Representations for Photovoltaic Forecasting

This repository studies whether categorical **Panchang/lunisolar representations**
add forecasting value beyond historical power and weather, Gregorian time, solar
geometry, and continuous lunar geometry. Its central comparison is **F3 vs F4**;
F0–F4 progression is reported to place the Panchang increment in context.
Panchang features are deterministic representations, not evidence of a causal
astronomical influence on photovoltaic generation.

## Datasets and forecasting tasks

**SOLETE (Denmark).** Five-minute meteorological and power measurements;
`P_Solar[kW]` is the PV-power forecasting target. Horizons are 15, 30 and 60
minutes. Dataset: <https://doi.org/10.11583/DTU.17040767>. The SOLETE source
files are provided separately and expected under `solete_dataset/`.

**PVOD (China).** Ten independent photovoltaic stations with 15-minute records;
forecast errors use power per nominal station capacity (p.u.). Station-specific
65/15/20 chronological divisions and gap-safe samples are retained. Dataset:
<https://doi.org/10.11922/sciencedb.01094>; data-description paper:
<https://doi.org/10.1016/j.solener.2021.09.050>. The source files are expected
under `external_data/PVOD/` (`metadata.csv`, `station00.csv`–`station09.csv`).
Source datasets are not redistributed in this repository.

For **solar PV**, models are trained on the established full chronological
populations (including night); both all-hours and daylight-only errors are
reported, with daylight accuracy emphasized for operational interpretation.
All-hours development-CV model selection is preserved as originally specified.
A future **wind forecasting** task would evaluate all hours with a separate
target, history and baselines; no wind-forecasting results are claimed here.

## Nested feature representations

| Family | Increment relative to prior family | SOLETE width | PVOD width |
|---|---|---:|---:|
| F0 | Observed historical power and meteorology | 118 | 71 |
| F1 | Gregorian time / cyclical calendar representation | 133 | 86 |
| F2 | Deterministic solar geometry | 138 | 91 |
| F3 | Continuous lunar and Sun–Moon geometry | 147 | 100 |
| F4 | Fixed one-hot Panchang categories (97 indicators) | 244 | 197 |

The five Panchang components are Tithi, Paksha, Karana, Nakshatra and Yoga.
SOLETE uses a validated timestamp +2-minute astronomical convention; PVOD uses
exact recorded UTC timestamps and station-specific solar geometry. The sidereal
convention is standard Lahiri with the documented non-nutating ayanamsha.
Continuous sidereal-angle parent controls accompany the primary F3–F4 study.
Historical observations use forecast-*issue* time; deterministic astronomical
and calendar information uses forecast-*valid* time.

## Unified, executed notebook sequence

| Notebook | Scope |
|---|---|
| [01](notebooks/01_data_exploration.ipynb) | SOLETE data exploration |
| [02](notebooks/02_data_preparation.ipynb) | SOLETE data preparation |
| [03](notebooks/03_astronomical_features.ipynb) | Gregorian, solar and continuous lunar features |
| [04](notebooks/04_panchang_features.ipynb) | Panchang definitions and independent astronomical checks |
| [05](notebooks/05_forecasting_dataset.ipynb) | Forecast populations, feature alignment, CV and leakage control |
| [06](notebooks/06_classical_models.ipynb) | Archived Ridge, Random Forest and LightGBM training/CV |
| [07](notebooks/07_neural_models.ipynb) | Archived LSTM, TCN, GRU and BiLSTM training/CV |
| [08](notebooks/08_lightgbm_model_development.ipynb) | Fixed-LightGBM development refits across feature families |
| [09](notebooks/09_feature_comparison.ipynb) | F0–F4 ablations and paired uncertainty |
| [10](notebooks/10_panchang_analysis.ipynb) | Panchang alignment, coding and component controls |
| [11](notebooks/11_solete_evaluation.ipynb) | SOLETE June–August 2019 retrospective evaluation |
| [12](notebooks/12_solete_cross_architecture_comparison.ipynb) | Seven-architecture development synthesis and LightGBM-run reconciliation |
| [13](notebooks/13_pvod_data_exploration.ipynb) | PVOD station-level exploratory checks |
| [14](notebooks/14_pvod_preparation_protocol.ipynb) | PVOD splits, gaps, populations and F0–F4 audit |
| [15](notebooks/15_pvod_evaluation.ipynb) | Station-specific PVOD retrospective evaluation |
| [16](notebooks/16_results_summary.ipynb) | Cross-dataset synthesis |

## Model training, evaluation and inference

Three expanding chronological SOLETE development folds use a 60-minute
training-label availability guard. The archived development campaign compares
**seven architectures, 555 candidate configurations and 1,665 candidate-fold
evaluations** (three classical and four neural architectures, F0–F4, three
horizons). Notebooks 06–07 audit the completed historical training records;
they do not initiate a second costly training campaign. Model definitions and
verification functions are retained in `src/`. Notebook 08 separately executes
the fixed LightGBM C1 configuration (300 trees, 0.03 learning rate, 15 leaves)
for matched F0–F4 and continuous-sidereal representations.

The frozen development-selection result is **LightGBM C1 / F1** for 15 minutes,
and **LightGBM C1 / F2** for 30 and 60 minutes; see
`results/development_only/revised_model_selection_lock_v1.json` and
`results/solete_architecture_synthesis/`.

For the key F3–F4 contrast, positive `MAE(F3) − MAE(F4)` means that adding
categorical Panchang features lowers error. Comparisons match the horizon,
architecture, configuration and evaluated population. The SOLETE and PVOD
retrospective evaluations have prior historical exposure and **are not blind
independent tests**. Cross-architecture development CV is distinct from the
newer fixed-LightGBM run; Notebook 12 documents their non-identical scores.

## Scientific interpretation

The observed Panchang effect is small or model-/horizon-/dataset-dependent; the
results do **not** establish a universal advantage over continuous astronomical
representations. Solar forecasting conclusions emphasize daytime metrics while
retaining all-hours comparability with the frozen model-selection design.
Bootstrap intervals estimate uncertainty in sampled periods and stations under
frozen predictions, not full model-training or hyperparameter uncertainty.
Multiple ablations are exploratory and are not a multiplicity-controlled
hypothesis-testing program.

## Results and methods

- `results/model_architectures/` — classical and neural development CV audits.
- `results/feature_comparison/` — fixed-LightGBM development predictions and F0–F4 comparisons.
- `results/representation_controls/` — alignment, permutation, coding and component controls.
- `results/solete_retrospective/` — SOLETE retrospective predictions and intervals.
- `results/solete_architecture_synthesis/` — seven-architecture comparison and reconciliation.
- `results/pvod_methodology/` — PVOD EDA, population and feature-schema audits.
- `results/pvod_feature_comparison/` — ten-station PVOD retrospective results.
- `results/cross_dataset_synthesis/` — cross-dataset summary tables.
- `results/development_only/` — frozen model-selection and training archives.

Methods are implemented in `src/` and verified by tests in `tests/`.
Historical pre-unification notebooks remain on the Git branch
`archive/pre-unified-panchang-methodology`; historical documents in `docs/`
retain their original experiment-stage wording as archival provenance.
The original research freeze also remains at tag `research-freeze-v1`
(commit `d6c18db`). Historical results were not re-estimated when notebooks
were reorganized; only notebook filenames, Markdown headings and references
were updated.

## Reproducibility

The recorded research Python version is **3.12.14**. With a Python 3.12
environment and separately obtained datasets and ephemeris:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ml,dl,notebook,dev]'
python -m pytest -q
ruff check .
```

Direct package provenance is retained in
`requirements/research-freeze-direct.txt`. Some historical archive readers
check exact source fingerprints; changes to `src/` can invalidate those checks.
The executed notebooks and archived metrics are research records rather than
instructions to retrain or retune models on retrospective data.
