"""
Frozen classical-model definitions for the PV forecasting study.

This module extracts the model specifications and validation-selection
semantics used in Notebooks 09 and 10 without rerunning or retuning any
experiment.

Included model families
-----------------------
* Ridge with StandardScaler
* Histogram Gradient Boosting (HGBR)
* Extra Trees
* XGBoost
* LightGBM
* Random Forest

The notebooks remain the record of the completed experimental runs.
This module provides reusable, testable definitions of the methods that
produced those runs.
"""

from __future__ import annotations

from itertools import product

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------

RIDGE_ALPHA_GRID = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
)


# ---------------------------------------------------------------------
# Histogram Gradient Boosting
# ---------------------------------------------------------------------

HGBR_LEARNING_RATES = (
    0.05,
    0.10,
)

HGBR_MAX_LEAF_NODES = (
    15,
    31,
)

HGBR_L2_REGULARIZATION = (
    0.0,
    1.0,
)

HGBR_MAX_ITER = 300

HGBR_PARAMETER_GRID = tuple(
    product(
        HGBR_LEARNING_RATES,
        HGBR_MAX_LEAF_NODES,
        HGBR_L2_REGULARIZATION,
    )
)


# ---------------------------------------------------------------------
# Extra Trees
# ---------------------------------------------------------------------

EXTRA_TREES_GRID = {
    "n_estimators": (
        300,
    ),
    "max_features": (
        0.5,
        1.0,
    ),
    "min_samples_leaf": (
        1,
        3,
    ),
    "max_depth": (
        None,
        30,
    ),
}

EXTRA_TREES_FIXED_PARAMS = {
    "criterion":
        "squared_error",
    "bootstrap":
        False,
    "random_state":
        RANDOM_STATE,
    "n_jobs":
        -1,
}

EXTRA_TREES_PARAMETER_GRID = tuple(
    product(
        EXTRA_TREES_GRID[
            "n_estimators"
        ],
        EXTRA_TREES_GRID[
            "max_features"
        ],
        EXTRA_TREES_GRID[
            "min_samples_leaf"
        ],
        EXTRA_TREES_GRID[
            "max_depth"
        ],
    )
)


# ---------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------

XGB_N_ESTIMATORS = (
    300,
    600,
)

XGB_LEARNING_RATES = (
    0.03,
    0.05,
)

XGB_MAX_DEPTHS = (
    4,
    6,
)

XGB_FIXED_PARAMS = {
    "objective":
        "reg:squarederror",
    "tree_method":
        "hist",
    "subsample":
        0.8,
    "colsample_bytree":
        0.8,
    "min_child_weight":
        1.0,
    "reg_alpha":
        0.0,
    "reg_lambda":
        1.0,
    "gamma":
        0.0,
    "random_state":
        RANDOM_STATE,
    "n_jobs":
        -1,
    "verbosity":
        0,
}

XGBOOST_CONFIGS = tuple(
    {
        "n_estimators":
            int(
                n_estimators
            ),
        "learning_rate":
            float(
                learning_rate
            ),
        "max_depth":
            int(
                max_depth
            ),
        **XGB_FIXED_PARAMS,
    }
    for (
        n_estimators,
        learning_rate,
        max_depth,
    ) in product(
        XGB_N_ESTIMATORS,
        XGB_LEARNING_RATES,
        XGB_MAX_DEPTHS,
    )
)


# ---------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------

LIGHTGBM_FIXED_PARAMS = {
    "objective":
        "regression",
    "boosting_type":
        "gbdt",
    "max_depth":
        -1,
    "subsample":
        0.8,
    "subsample_freq":
        1,
    "colsample_bytree":
        0.8,
    "min_child_samples":
        20,
    "reg_alpha":
        0.0,
    "reg_lambda":
        1.0,
    "random_state":
        RANDOM_STATE,
    "deterministic":
        True,
    "force_col_wise":
        True,
    "n_jobs":
        -1,
    "verbosity":
        -1,
}

LIGHTGBM_N_ESTIMATORS = (
    300,
    600,
)

LIGHTGBM_LEARNING_RATES = (
    0.03,
    0.05,
)

LIGHTGBM_NUM_LEAVES = (
    15,
    31,
)

LIGHTGBM_CONFIGS = tuple(
    {
        "n_estimators":
            n_estimators,
        "learning_rate":
            learning_rate,
        "num_leaves":
            num_leaves,
    }
    for n_estimators in (
        LIGHTGBM_N_ESTIMATORS
    )
    for learning_rate in (
        LIGHTGBM_LEARNING_RATES
    )
    for num_leaves in (
        LIGHTGBM_NUM_LEAVES
    )
)


# ---------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------

RANDOM_FOREST_FIXED_PARAMS = {
    "n_estimators":
        300,
    "criterion":
        "squared_error",
    "bootstrap":
        True,
    "max_samples":
        None,
    "random_state":
        RANDOM_STATE,
    "n_jobs":
        -1,
    "verbose":
        0,
}

RANDOM_FOREST_MAX_FEATURES = (
    0.5,
    1.0,
)

RANDOM_FOREST_MIN_SAMPLES_LEAF = (
    1,
    3,
)

RANDOM_FOREST_MAX_DEPTH = (
    None,
    30,
)

RANDOM_FOREST_CONFIGS = tuple(
    {
        "max_features":
            max_features,
        "min_samples_leaf":
            min_samples_leaf,
        "max_depth":
            max_depth,
    }
    for max_features in (
        RANDOM_FOREST_MAX_FEATURES
    )
    for min_samples_leaf in (
        RANDOM_FOREST_MIN_SAMPLES_LEAF
    )
    for max_depth in (
        RANDOM_FOREST_MAX_DEPTH
    )
)


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------


def build_ridge_model(
    alpha: float,
) -> Pipeline:
    """
    Build the frozen StandardScaler -> Ridge pipeline.

    The scaler is part of the estimator pipeline so it is fitted only
    on whatever fitting population is passed to ``fit``.
    """

    alpha = float(
        alpha
    )

    if alpha not in (
        RIDGE_ALPHA_GRID
    ):
        raise ValueError(
            f"Alpha {alpha} is not in the frozen Ridge grid."
        )

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha
                ),
            ),
        ]
    )


def build_hgbr_model(
    config,
) -> HistGradientBoostingRegressor:
    """Build one frozen HGBR configuration."""

    required = (
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "max_iter",
    )

    _require_config_keys(
        config,
        required,
    )

    learning_rate = float(
        config[
            "learning_rate"
        ]
    )

    max_leaf_nodes = int(
        config[
            "max_leaf_nodes"
        ]
    )

    l2_regularization = float(
        config[
            "l2_regularization"
        ]
    )

    max_iter = int(
        config[
            "max_iter"
        ]
    )

    if learning_rate not in (
        HGBR_LEARNING_RATES
    ):
        raise ValueError(
            "Unexpected HGBR learning_rate."
        )

    if max_leaf_nodes not in (
        HGBR_MAX_LEAF_NODES
    ):
        raise ValueError(
            "Unexpected HGBR max_leaf_nodes."
        )

    if l2_regularization not in (
        HGBR_L2_REGULARIZATION
    ):
        raise ValueError(
            "Unexpected HGBR l2_regularization."
        )

    if max_iter != HGBR_MAX_ITER:
        raise ValueError(
            "Unexpected HGBR max_iter."
        )

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=(
            l2_regularization
        ),
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def build_extra_trees_model(
    config,
) -> ExtraTreesRegressor:
    """Build one frozen Extra Trees configuration."""

    required = (
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "max_depth",
    )

    _require_config_keys(
        config,
        required,
    )

    return ExtraTreesRegressor(
        n_estimators=int(
            config[
                "n_estimators"
            ]
        ),
        max_features=float(
            config[
                "max_features"
            ]
        ),
        min_samples_leaf=int(
            config[
                "min_samples_leaf"
            ]
        ),
        max_depth=_optional_int(
            config[
                "max_depth"
            ]
        ),
        **EXTRA_TREES_FIXED_PARAMS,
    )


def build_xgboost_model(
    config,
) -> XGBRegressor:
    """Build one frozen XGBoost configuration."""

    required = (
        "n_estimators",
        "learning_rate",
        "max_depth",
    )

    _require_config_keys(
        config,
        required,
    )

    params = {
        "n_estimators":
            int(
                config[
                    "n_estimators"
                ]
            ),
        "learning_rate":
            float(
                config[
                    "learning_rate"
                ]
            ),
        "max_depth":
            int(
                config[
                    "max_depth"
                ]
            ),
        **XGB_FIXED_PARAMS,
    }

    return XGBRegressor(
        **params
    )


def build_lightgbm_model(
    config,
) -> LGBMRegressor:
    """Build one frozen LightGBM configuration."""

    required = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
    )

    _require_config_keys(
        config,
        required,
    )

    params = {
        **LIGHTGBM_FIXED_PARAMS,
        "n_estimators":
            int(
                config[
                    "n_estimators"
                ]
            ),
        "learning_rate":
            float(
                config[
                    "learning_rate"
                ]
            ),
        "num_leaves":
            int(
                config[
                    "num_leaves"
                ]
            ),
    }

    return LGBMRegressor(
        **params
    )


def build_random_forest_model(
    config,
) -> RandomForestRegressor:
    """Build one frozen Random Forest configuration."""

    required = (
        "max_features",
        "min_samples_leaf",
        "max_depth",
    )

    _require_config_keys(
        config,
        required,
    )

    params = {
        **RANDOM_FOREST_FIXED_PARAMS,
        "max_features":
            float(
                config[
                    "max_features"
                ]
            ),
        "min_samples_leaf":
            int(
                config[
                    "min_samples_leaf"
                ]
            ),
        "max_depth":
            _optional_int(
                config[
                    "max_depth"
                ]
            ),
    }

    return RandomForestRegressor(
        **params
    )


# ---------------------------------------------------------------------
# Validation-result ordering
# ---------------------------------------------------------------------


def rank_validation_results(
    results: pd.DataFrame,
    *,
    mae_column: str,
    rmse_column: str | None = None,
) -> pd.DataFrame:
    """
    Rank validation configurations using the frozen criterion.

    Ridge and HGBR originally selected by validation MAE alone.
    Notebook-10 tree models used validation MAE as the primary
    criterion and validation RMSE as the secondary/tie-break criterion.
    """

    if not isinstance(
        results,
        pd.DataFrame,
    ):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    if results.empty:
        raise ValueError(
            "results must contain at least one configuration."
        )

    columns = [
        mae_column,
    ]

    if rmse_column is not None:
        columns.append(
            rmse_column
        )

    missing = [
        column
        for column in columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing validation metric columns: {missing}."
        )

    return (
        results
        .sort_values(
            columns,
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


def select_best_validation_result(
    results: pd.DataFrame,
    *,
    mae_column: str,
    rmse_column: str | None = None,
) -> pd.Series:
    """Return the first row under the frozen validation ranking."""

    ranked = rank_validation_results(
        results,
        mae_column=mae_column,
        rmse_column=rmse_column,
    )

    return ranked.iloc[
        0
    ].copy()


# ---------------------------------------------------------------------
# Executable selected-configuration freezing
# ---------------------------------------------------------------------


def freeze_ridge_config(
    selected_row,
) -> dict:
    """Convert one validation-selected Ridge row to executable config."""

    _require_config_keys(
        selected_row,
        (
            "alpha",
        ),
    )

    return {
        "alpha":
            float(
                selected_row[
                    "alpha"
                ]
            ),
    }


def freeze_hgbr_config(
    selected_row,
) -> dict:
    """Convert one validation-selected HGBR row to executable config."""

    required = (
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "max_iter",
    )

    _require_config_keys(
        selected_row,
        required,
    )

    return {
        "learning_rate":
            float(
                selected_row[
                    "learning_rate"
                ]
            ),
        "max_leaf_nodes":
            int(
                selected_row[
                    "max_leaf_nodes"
                ]
            ),
        "l2_regularization":
            float(
                selected_row[
                    "l2_regularization"
                ]
            ),
        "max_iter":
            int(
                selected_row[
                    "max_iter"
                ]
            ),
    }


def freeze_extra_trees_config(
    selected_row,
) -> dict:
    """Restore one selected Extra Trees row to executable parameters."""

    required = (
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "max_depth",
    )

    _require_config_keys(
        selected_row,
        required,
    )

    return {
        "n_estimators":
            int(
                selected_row[
                    "n_estimators"
                ]
            ),
        "max_features":
            float(
                selected_row[
                    "max_features"
                ]
            ),
        "min_samples_leaf":
            int(
                selected_row[
                    "min_samples_leaf"
                ]
            ),
        "max_depth":
            _optional_int(
                selected_row[
                    "max_depth"
                ]
            ),
        **EXTRA_TREES_FIXED_PARAMS,
    }


def freeze_xgboost_config(
    selected_row,
) -> dict:
    """Restore one selected XGBoost row to executable parameters."""

    required = (
        "n_estimators",
        "learning_rate",
        "max_depth",
    )

    _require_config_keys(
        selected_row,
        required,
    )

    return {
        "n_estimators":
            int(
                selected_row[
                    "n_estimators"
                ]
            ),
        "learning_rate":
            float(
                selected_row[
                    "learning_rate"
                ]
            ),
        "max_depth":
            int(
                selected_row[
                    "max_depth"
                ]
            ),
        **XGB_FIXED_PARAMS,
    }


def freeze_lightgbm_config(
    selected_row,
) -> dict:
    """Restore one selected LightGBM row to executable parameters."""

    required = (
        "n_estimators",
        "learning_rate",
        "num_leaves",
    )

    _require_config_keys(
        selected_row,
        required,
    )

    return {
        "n_estimators":
            int(
                selected_row[
                    "n_estimators"
                ]
            ),
        "learning_rate":
            float(
                selected_row[
                    "learning_rate"
                ]
            ),
        "num_leaves":
            int(
                selected_row[
                    "num_leaves"
                ]
            ),
        **LIGHTGBM_FIXED_PARAMS,
    }


def freeze_random_forest_config(
    selected_row,
) -> dict:
    """Restore one selected Random Forest row to executable parameters."""

    required = (
        "max_features",
        "min_samples_leaf",
        "max_depth",
    )

    _require_config_keys(
        selected_row,
        required,
    )

    return {
        **RANDOM_FOREST_FIXED_PARAMS,
        "max_features":
            float(
                selected_row[
                    "max_features"
                ]
            ),
        "min_samples_leaf":
            int(
                selected_row[
                    "min_samples_leaf"
                ]
            ),
        "max_depth":
            _optional_int(
                selected_row[
                    "max_depth"
                ]
            ),
    }


# ---------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------


def _require_config_keys(
    config,
    required,
) -> None:
    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:
        raise ValueError(
            f"Missing configuration keys: {missing}."
        )


def _optional_int(
    value,
) -> int | None:
    """
    Restore a nullable depth from pandas results.

    Pandas represents ``None`` as NaN in mixed numeric result columns;
    the fitted sklearn estimators require Python ``None`` again.
    """

    if pd.isna(
        value
    ):
        return None

    return int(
        value
    )
