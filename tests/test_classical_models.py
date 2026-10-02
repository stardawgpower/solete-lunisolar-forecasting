import numpy as np
import pandas as pd
import pytest
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

from classical_models import (
    EXTRA_TREES_PARAMETER_GRID,
    HGBR_MAX_ITER,
    HGBR_PARAMETER_GRID,
    LIGHTGBM_CONFIGS,
    RANDOM_FOREST_CONFIGS,
    RIDGE_ALPHA_GRID,
    XGBOOST_CONFIGS,
    build_extra_trees_model,
    build_hgbr_model,
    build_lightgbm_model,
    build_random_forest_model,
    build_ridge_model,
    build_xgboost_model,
    freeze_extra_trees_config,
    freeze_hgbr_config,
    freeze_lightgbm_config,
    freeze_random_forest_config,
    freeze_ridge_config,
    freeze_xgboost_config,
    rank_validation_results,
)


def test_ridge_grid_is_frozen():
    assert RIDGE_ALPHA_GRID == (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )


def test_hgbr_grid_is_frozen():
    assert len(
        HGBR_PARAMETER_GRID
    ) == 8

    assert set(
        HGBR_PARAMETER_GRID
    ) == {
        (
            learning_rate,
            leaves,
            l2,
        )
        for learning_rate in (
            0.05,
            0.10,
        )
        for leaves in (
            15,
            31,
        )
        for l2 in (
            0.0,
            1.0,
        )
    }

    assert HGBR_MAX_ITER == 300


def test_extra_trees_grid_is_frozen():
    assert len(
        EXTRA_TREES_PARAMETER_GRID
    ) == 8

    assert set(
        EXTRA_TREES_PARAMETER_GRID
    ) == {
        (
            300,
            max_features,
            min_samples_leaf,
            max_depth,
        )
        for max_features in (
            0.5,
            1.0,
        )
        for min_samples_leaf in (
            1,
            3,
        )
        for max_depth in (
            None,
            30,
        )
    }


def test_xgboost_grid_is_frozen():
    assert len(
        XGBOOST_CONFIGS
    ) == 8

    assert {
        config[
            "n_estimators"
        ]
        for config in XGBOOST_CONFIGS
    } == {
        300,
        600,
    }

    assert {
        config[
            "learning_rate"
        ]
        for config in XGBOOST_CONFIGS
    } == {
        0.03,
        0.05,
    }

    assert {
        config[
            "max_depth"
        ]
        for config in XGBOOST_CONFIGS
    } == {
        4,
        6,
    }


def test_lightgbm_grid_is_frozen():
    assert len(
        LIGHTGBM_CONFIGS
    ) == 8

    assert {
        config[
            "n_estimators"
        ]
        for config in LIGHTGBM_CONFIGS
    } == {
        300,
        600,
    }

    assert {
        config[
            "learning_rate"
        ]
        for config in LIGHTGBM_CONFIGS
    } == {
        0.03,
        0.05,
    }

    assert {
        config[
            "num_leaves"
        ]
        for config in LIGHTGBM_CONFIGS
    } == {
        15,
        31,
    }


def test_random_forest_grid_is_frozen():
    assert len(
        RANDOM_FOREST_CONFIGS
    ) == 8

    assert {
        config[
            "max_features"
        ]
        for config in RANDOM_FOREST_CONFIGS
    } == {
        0.5,
        1.0,
    }

    assert {
        config[
            "min_samples_leaf"
        ]
        for config in RANDOM_FOREST_CONFIGS
    } == {
        1,
        3,
    }

    assert {
        config[
            "max_depth"
        ]
        for config in RANDOM_FOREST_CONFIGS
    } == {
        None,
        30,
    }


def test_ridge_builder_preserves_pipeline_semantics():
    model = build_ridge_model(
        1.0
    )

    assert isinstance(
        model,
        Pipeline,
    )

    assert list(
        model.named_steps
    ) == [
        "scaler",
        "ridge",
    ]

    assert isinstance(
        model.named_steps[
            "scaler"
        ],
        StandardScaler,
    )

    assert isinstance(
        model.named_steps[
            "ridge"
        ],
        Ridge,
    )

    assert model.named_steps[
        "ridge"
    ].alpha == pytest.approx(
        1.0
    )


def test_hgbr_builder_preserves_frozen_parameters():
    model = build_hgbr_model(
        {
            "learning_rate":
                0.05,
            "max_leaf_nodes":
                15,
            "l2_regularization":
                1.0,
            "max_iter":
                300,
        }
    )

    assert isinstance(
        model,
        HistGradientBoostingRegressor,
    )

    assert model.loss == "squared_error"
    assert model.learning_rate == pytest.approx(
        0.05
    )
    assert model.max_leaf_nodes == 15
    assert model.l2_regularization == pytest.approx(
        1.0
    )
    assert model.max_iter == 300
    assert model.early_stopping is False
    assert model.random_state == 42


def test_extra_trees_builder_preserves_frozen_parameters():
    model = build_extra_trees_model(
        {
            "n_estimators":
                300,
            "max_features":
                0.5,
            "min_samples_leaf":
                3,
            "max_depth":
                None,
        }
    )

    assert isinstance(
        model,
        ExtraTreesRegressor,
    )

    assert model.n_estimators == 300
    assert model.max_features == pytest.approx(
        0.5
    )
    assert model.min_samples_leaf == 3
    assert model.max_depth is None
    assert model.criterion == "squared_error"
    assert model.bootstrap is False
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_xgboost_builder_preserves_frozen_parameters():
    model = build_xgboost_model(
        {
            "n_estimators":
                600,
            "learning_rate":
                0.03,
            "max_depth":
                6,
        }
    )

    assert isinstance(
        model,
        XGBRegressor,
    )

    params = model.get_params()

    assert params[
        "n_estimators"
    ] == 600

    assert params[
        "learning_rate"
    ] == pytest.approx(
        0.03
    )

    assert params[
        "max_depth"
    ] == 6

    assert params[
        "objective"
    ] == "reg:squarederror"

    assert params[
        "tree_method"
    ] == "hist"

    assert params[
        "subsample"
    ] == pytest.approx(
        0.8
    )

    assert params[
        "colsample_bytree"
    ] == pytest.approx(
        0.8
    )

    assert params[
        "random_state"
    ] == 42


def test_lightgbm_builder_preserves_frozen_parameters():
    model = build_lightgbm_model(
        {
            "n_estimators":
                600,
            "learning_rate":
                0.05,
            "num_leaves":
                31,
        }
    )

    assert isinstance(
        model,
        LGBMRegressor,
    )

    params = model.get_params()

    assert params[
        "n_estimators"
    ] == 600

    assert params[
        "learning_rate"
    ] == pytest.approx(
        0.05
    )

    assert params[
        "num_leaves"
    ] == 31

    assert params[
        "boosting_type"
    ] == "gbdt"

    assert params[
        "subsample"
    ] == pytest.approx(
        0.8
    )

    assert params[
        "subsample_freq"
    ] == 1

    assert params[
        "deterministic"
    ] is True

    assert params[
        "force_col_wise"
    ] is True

    assert params[
        "random_state"
    ] == 42


def test_random_forest_builder_preserves_frozen_parameters():
    model = build_random_forest_model(
        {
            "max_features":
                1.0,
            "min_samples_leaf":
                1,
            "max_depth":
                30,
        }
    )

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators == 300
    assert model.max_features == pytest.approx(
        1.0
    )
    assert model.min_samples_leaf == 1
    assert model.max_depth == 30
    assert model.criterion == "squared_error"
    assert model.bootstrap is True
    assert model.max_samples is None
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_validation_ranking_can_use_mae_only():
    results = pd.DataFrame(
        {
            "name":
                [
                    "a",
                    "b",
                    "c",
                ],
            "MAE_kW":
                [
                    0.3,
                    0.1,
                    0.2,
                ],
        }
    )

    ranked = rank_validation_results(
        results,
        mae_column="MAE_kW",
    )

    assert ranked[
        "name"
    ].tolist() == [
        "b",
        "c",
        "a",
    ]


def test_validation_ranking_uses_rmse_as_secondary_key():
    results = pd.DataFrame(
        {
            "name":
                [
                    "worse_tie",
                    "best_tie",
                    "higher_mae",
                ],
            "validation_mae":
                [
                    0.1,
                    0.1,
                    0.2,
                ],
            "validation_rmse":
                [
                    0.5,
                    0.3,
                    0.1,
                ],
        }
    )

    ranked = rank_validation_results(
        results,
        mae_column="validation_mae",
        rmse_column="validation_rmse",
    )

    assert ranked[
        "name"
    ].tolist() == [
        "best_tie",
        "worse_tie",
        "higher_mae",
    ]


def test_freeze_ridge_config():
    config = freeze_ridge_config(
        pd.Series(
            {
                "alpha":
                    0.1,
            }
        )
    )

    assert config == {
        "alpha":
            0.1,
    }


def test_freeze_hgbr_config():
    config = freeze_hgbr_config(
        pd.Series(
            {
                "learning_rate":
                    0.05,
                "max_leaf_nodes":
                    31.0,
                "l2_regularization":
                    1.0,
                "max_iter":
                    300.0,
            }
        )
    )

    assert config == {
        "learning_rate":
            0.05,
        "max_leaf_nodes":
            31,
        "l2_regularization":
            1.0,
        "max_iter":
            300,
    }


def test_freeze_extra_trees_config_restores_none_depth():
    config = freeze_extra_trees_config(
        pd.Series(
            {
                "n_estimators":
                    300.0,
                "max_features":
                    0.5,
                "min_samples_leaf":
                    3.0,
                "max_depth":
                    np.nan,
            }
        )
    )

    assert config[
        "max_depth"
    ] is None

    assert config[
        "bootstrap"
    ] is False

    assert config[
        "criterion"
    ] == "squared_error"

    assert config[
        "random_state"
    ] == 42


def test_freeze_xgboost_config_adds_fixed_parameters():
    config = freeze_xgboost_config(
        pd.Series(
            {
                "n_estimators":
                    600.0,
                "learning_rate":
                    0.03,
                "max_depth":
                    4.0,
            }
        )
    )

    assert config[
        "n_estimators"
    ] == 600

    assert config[
        "max_depth"
    ] == 4

    assert config[
        "tree_method"
    ] == "hist"

    assert config[
        "subsample"
    ] == pytest.approx(
        0.8
    )

    assert config[
        "random_state"
    ] == 42


def test_freeze_lightgbm_config_adds_fixed_parameters():
    config = freeze_lightgbm_config(
        pd.Series(
            {
                "n_estimators":
                    300.0,
                "learning_rate":
                    0.05,
                "num_leaves":
                    15.0,
            }
        )
    )

    assert config[
        "n_estimators"
    ] == 300

    assert config[
        "num_leaves"
    ] == 15

    assert config[
        "boosting_type"
    ] == "gbdt"

    assert config[
        "deterministic"
    ] is True

    assert config[
        "random_state"
    ] == 42


def test_freeze_random_forest_config_restores_none_depth():
    config = freeze_random_forest_config(
        pd.Series(
            {
                "max_features":
                    0.5,
                "min_samples_leaf":
                    3.0,
                "max_depth":
                    np.nan,
            }
        )
    )

    assert config[
        "n_estimators"
    ] == 300

    assert config[
        "max_depth"
    ] is None

    assert config[
        "bootstrap"
    ] is True

    assert config[
        "max_samples"
    ] is None

    assert config[
        "random_state"
    ] == 42
