# Categorical Lunisolar Representations for Photovoltaic Forecasting

Research repository for a cross-dataset study of categorical Panchang/lunisolar
calendar representations in short-horizon photovoltaic power forecasting.

## Scientific scope

This project studies **forecasting representation effects**. Panchang categories
are treated as alternative representations of deterministic temporal and
astronomical state.

The experiments do **not** establish a causal or physical effect of Panchang
categories on photovoltaic generation.

The completed study examines how representation effects vary with:

- model architecture
- forecast horizon
- dataset
- temporal alignment
- categorical coding
- interactions among Panchang feature families

## Datasets

### SOLETE

Primary development dataset:

- 15 months of meteorology and co-located renewable-power measurements
- Denmark
- primary forecasting resolution in this study: 5 minutes
- PV target: `P_Solar[kW]`
- DOI: <https://doi.org/10.11583/DTU.17040767>

Large SOLETE HDF5 files are not redistributed by this repository.

The required local files are expected under:

```text
solete_dataset/
```

The upstream SOLETE dataset description is retained in:

```text
solete_dataset/README.txt
```

### PVOD

PVOD is used for external validation across ten PV stations.

The external dataset is not redistributed by this repository. The frozen
validation notebook expects it locally under:

```text
external_data/PVOD/
├── metadata.csv
├── station00.csv
├── station01.csv
├── ...
└── station09.csv
```

The validation protocol preserves temporal gaps rather than interpolating them
and performs station-specific model refits.

## Forecasting design

Forecast horizons are 15, 30, and 60 minutes.

### SOLETE feature families

| Family | Representation | Dimension |
| --- | --- | ---: |
| F0 | observational history / autoregressive predictors | 118 |
| F1 | F0 + Gregorian temporal representation | 133 |
| F2 | F1 + deterministic solar geometry | 138 |
| F3 | F2 + continuous lunar and Sun-Moon geometry | 147 |
| F4 | F3 + categorical Panchang one-hot representation | 244 |

### PVOD feature families

| Family | Dimension |
| --- | ---: |
| F0 | 71 |
| F1 | 86 |
| F2 | 91 |
| F3 | 100 |
| F4 | 197 |

Deterministic future-known features are aligned to forecast-valid time.
Observed future meteorological measurements are not treated as future-known
predictors.

## Models

The completed study includes classical and neural forecasting architectures:

- Ridge
- Histogram Gradient Boosting
- Extra Trees
- Random Forest
- XGBoost
- LightGBM
- LSTM
- GRU
- Bidirectional LSTM
- TCN

Frozen model definitions and training protocols used for publication hardening
are implemented in `src/` and covered by tests.

## Representation controls

Mechanism analyses include:

- aligned categorical Panchang representation
- circular shifts of 17, 37, and 61 days
- split-wise joint permutations with seeds 1301, 1302, and 1303
- alternative/reference categorical coding
- individual Panchang component additions
- non-additivity and interaction analyses

These controls investigate representation, temporal alignment, regularization,
and architecture sensitivity rather than causal effects.

## Repository structure

```text
.
├── notebooks/       # executed research record
├── results/         # frozen tables, predictions, figures, and audits
├── src/             # reusable scientific methods
├── tests/           # scientific regression and unit tests
├── requirements/    # frozen environment provenance
├── solete_dataset/  # SOLETE attribution/readme; large data excluded
└── pyproject.toml
```

Important reusable modules include:

```text
src/
├── data_loader.py
├── preprocessing.py
├── astronomy_features.py
├── vedic_calendar.py
├── feature_families.py
├── representation_controls.py
├── forecasting_protocol.py
├── classical_models.py
├── neural_models.py
├── evaluation.py
├── bootstrap.py
└── pvod.py
```

Executed notebooks are retained as the historical research record rather than
being automatically reformatted during publication hardening.

## Reproducibility

### Frozen scientific result

The completed experimental result was frozen before publication hardening:

```text
tag:    research-freeze-v1
commit: d6c18db
```

Publication-hardening commits extract and test the scientific method without
changing the already executed experimental results.

### Python environment

Frozen Python version:

```text
Python 3.12.14
```

Create and activate a Python 3.12 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[ml,dl,notebook,dev]'
```

Direct package versions observed in the completed research environment are
recorded in:

```text
requirements/research-freeze-direct.txt
```

This file records direct dependency provenance and is not claimed to be a
complete transitive lock file.

## Publication QA

Run:

```bash
ruff check .
python -m pytest -q
git diff --check
```

At the current publication-hardening milestone:

```text
128 passed
```

## Frozen results

Major result directories include:

```text
results/
├── notebook09_primary_experiment/
├── notebook10_secondary_ml_models/
├── notebook11_deep_learning_models/
├── notebook12_phase1_final_synthesis/
├── notebook13_phase2_f4_representation_analysis/
├── pvod_external_validation/
└── notebook15_final_results_synthesis/
```

The final synthesis directory contains publication-oriented figures,
corresponding figure-data CSV files, and audit/provenance artifacts.

## Interpretation limitations

The completed analysis has several important limitations:

- neural-model uncertainty does not include repeated retraining across random
  seeds
- bootstrap intervals quantify sampling/block uncertainty under frozen fitted
  predictions rather than full training uncertainty
- many representation comparisons are exploratory or secondary rather than a
  familywise multiplicity-controlled hypothesis-testing programme
- PVOD external validation contains ten stations from one external dataset
- Phase-2 representation controls are mechanistic diagnostics rather than
  causal tests
- shifted or permuted controls can sometimes outperform the aligned
  representation, which is consistent with nuisance representation,
  regularization, or optimization effects

Accordingly, the evidence supports architecture-dependent representation
sensitivity, not a universal categorical Panchang advantage and not a causal
effect of Panchang categories on photovoltaic generation.

## Research provenance

Executed notebooks are retained as the historical record of the completed
experiments. Reusable scientific definitions were subsequently extracted into
`src/` and covered by tests so that the publication methods can be audited
without rewriting the original notebook executions.