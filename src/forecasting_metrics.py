"""
Regression-evaluation utilities for the frozen PV forecasting study.

The completed experiments used three closely related metric contracts:

* SOLETE classical / Phase-2 evaluation:
  MAE, RMSE, capacity-normalized MAE/RMSE, and R².

* SOLETE deep-learning evaluation:
  MAE, RMSE, and a manually computed R² that is NaN when the
  target has zero variance.

* PVOD external validation:
  MAE, RMSE, and R² in both per-unit and MW representations.

These contracts are kept distinct here so publication code preserves
the behavior of the frozen experiments rather than silently changing
edge-case semantics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SOLETE_PV_RATED_KW = 10.0


def _as_finite_1d(
    values,
    *,
    name: str,
) -> np.ndarray:
    """Convert one numeric input to a finite one-dimensional array."""

    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size == 0:
        raise ValueError(
            f"{name} must contain at least one value."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def _validate_pair(
    y_true,
    y_pred,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one target/prediction pair."""

    true = _as_finite_1d(
        y_true,
        name="y_true",
    )
    pred = _as_finite_1d(
        y_pred,
        name="y_pred",
    )

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    return true, pred


def solete_regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Compute the frozen SOLETE classical / Phase-2 metric contract.

    Returns
    -------
    dict
        ``MAE_kW``
            Mean absolute error in kW.

        ``RMSE_kW``
            Root-mean-square error in kW.

        ``nMAE_percent``
            MAE as percent of the nominal 10 kW PV capacity.

        ``nRMSE_percent``
            RMSE as percent of the nominal 10 kW PV capacity.

        ``R2``
            scikit-learn coefficient of determination.
    """

    true, pred = _validate_pair(
        y_true,
        y_pred,
    )

    mae = float(
        mean_absolute_error(
            true,
            pred,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                true,
                pred,
            )
        )
    )

    r2 = float(
        r2_score(
            true,
            pred,
        )
    )

    return {
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "nMAE_percent": (
            100.0
            * mae
            / SOLETE_PV_RATED_KW
        ),
        "nRMSE_percent": (
            100.0
            * rmse
            / SOLETE_PV_RATED_KW
        ),
        "R2": r2,
    }


def deep_regression_metrics_kw(
    y_true_kw,
    y_pred_kw,
) -> dict[str, float]:
    """
    Compute the frozen Notebook-11 deep-learning metric contract.

    R² is computed manually. If the true target has zero variance,
    ``R2`` is returned as ``NaN``, matching the frozen helper.
    """

    true, pred = _validate_pair(
        y_true_kw,
        y_pred_kw,
    )

    errors = pred - true
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)

    mae_kw = float(
        absolute_errors.mean()
    )

    rmse_kw = float(
        np.sqrt(
            squared_errors.mean()
        )
    )

    true_mean = float(
        true.mean()
    )

    total_sum_of_squares = float(
        np.square(
            true - true_mean
        ).sum()
    )

    residual_sum_of_squares = float(
        squared_errors.sum()
    )

    if total_sum_of_squares > 0.0:
        r2 = float(
            1.0
            - (
                residual_sum_of_squares
                / total_sum_of_squares
            )
        )
    else:
        r2 = float("nan")

    return {
        "MAE_kW": mae_kw,
        "RMSE_kW": rmse_kw,
        "R2": r2,
    }


def pvod_regression_metrics(
    y_true_pu,
    y_pred_pu,
    y_true_mw,
    y_pred_mw,
) -> dict[str, float | int]:
    """
    Compute the frozen PVOD metric contract in p.u. and MW.

    All four arrays must describe the same non-empty set of forecast
    cases.
    """

    true_pu, pred_pu = _validate_pair(
        y_true_pu,
        y_pred_pu,
    )

    true_mw, pred_mw = _validate_pair(
        y_true_mw,
        y_pred_mw,
    )

    if true_pu.shape != true_mw.shape:
        raise ValueError(
            "Per-unit and MW metric inputs must have "
            "identical lengths."
        )

    if true_pu.size <= 1:
        raise ValueError(
            "PVOD regression metrics require at least two rows."
        )

    return {
        "n": int(
            true_pu.size
        ),
        "MAE_power_pu": float(
            mean_absolute_error(
                true_pu,
                pred_pu,
            )
        ),
        "RMSE_power_pu": float(
            np.sqrt(
                mean_squared_error(
                    true_pu,
                    pred_pu,
                )
            )
        ),
        "R2_power_pu": float(
            r2_score(
                true_pu,
                pred_pu,
            )
        ),
        "MAE_MW": float(
            mean_absolute_error(
                true_mw,
                pred_mw,
            )
        ),
        "RMSE_MW": float(
            np.sqrt(
                mean_squared_error(
                    true_mw,
                    pred_mw,
                )
            )
        ),
        "R2_MW": float(
            r2_score(
                true_mw,
                pred_mw,
            )
        ),
    }
