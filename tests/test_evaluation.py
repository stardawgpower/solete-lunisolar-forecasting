import numpy as np
import pytest

from evaluation import (
    SOLETE_PV_RATED_KW,
    deep_regression_metrics_kw,
    pvod_regression_metrics,
    solete_regression_metrics,
)


def test_solete_rated_capacity_is_frozen():
    assert SOLETE_PV_RATED_KW == 10.0


def test_solete_regression_metrics_known_example():
    y_true = np.array(
        [0.0, 2.0, 4.0]
    )
    y_pred = np.array(
        [1.0, 2.0, 3.0]
    )

    metrics = solete_regression_metrics(
        y_true,
        y_pred,
    )

    expected_mae = 2.0 / 3.0
    expected_rmse = np.sqrt(
        2.0 / 3.0
    )

    assert metrics["MAE_kW"] == pytest.approx(
        expected_mae
    )
    assert metrics["RMSE_kW"] == pytest.approx(
        expected_rmse
    )
    assert metrics[
        "nMAE_percent"
    ] == pytest.approx(
        100.0
        * expected_mae
        / 10.0
    )
    assert metrics[
        "nRMSE_percent"
    ] == pytest.approx(
        100.0
        * expected_rmse
        / 10.0
    )
    assert metrics["R2"] == pytest.approx(
        0.75
    )


def test_deep_metrics_match_frozen_manual_r2():
    y_true = np.array(
        [1.0, 2.0, 3.0]
    )
    y_pred = np.array(
        [1.0, 2.0, 4.0]
    )

    metrics = deep_regression_metrics_kw(
        y_true,
        y_pred,
    )

    assert metrics["MAE_kW"] == pytest.approx(
        1.0 / 3.0
    )
    assert metrics["RMSE_kW"] == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )
    assert metrics["R2"] == pytest.approx(
        0.5
    )


def test_deep_metrics_constant_target_returns_nan_r2():
    metrics = deep_regression_metrics_kw(
        [2.0, 2.0, 2.0],
        [1.0, 2.0, 3.0],
    )

    assert np.isnan(
        metrics["R2"]
    )


def test_pvod_metrics_known_example():
    y_true_pu = np.array(
        [0.0, 0.5, 1.0]
    )
    y_pred_pu = np.array(
        [0.0, 0.4, 0.8]
    )

    capacity_mw = 2.0

    metrics = pvod_regression_metrics(
        y_true_pu,
        y_pred_pu,
        y_true_pu * capacity_mw,
        y_pred_pu * capacity_mw,
    )

    assert metrics["n"] == 3

    assert metrics[
        "MAE_power_pu"
    ] == pytest.approx(
        0.1
    )

    assert metrics[
        "RMSE_power_pu"
    ] == pytest.approx(
        np.sqrt(
            (0.0**2 + 0.1**2 + 0.2**2)
            / 3.0
        )
    )

    assert metrics["MAE_MW"] == pytest.approx(
        0.2
    )

    assert metrics["RMSE_MW"] == pytest.approx(
        2.0
        * metrics["RMSE_power_pu"]
    )

    assert metrics["R2_MW"] == pytest.approx(
        metrics["R2_power_pu"]
    )


def test_mismatched_prediction_length_is_rejected():
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        solete_regression_metrics(
            [1.0, 2.0],
            [1.0],
        )


def test_nonfinite_values_are_rejected():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        deep_regression_metrics_kw(
            [1.0, np.nan],
            [1.0, 2.0],
        )


def test_empty_inputs_are_rejected():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        solete_regression_metrics(
            [],
            [],
        )


def test_pvod_requires_matching_pu_and_mw_lengths():
    with pytest.raises(
        ValueError,
        match="identical lengths",
    ):
        pvod_regression_metrics(
            [0.0, 1.0, 0.5],
            [0.0, 0.9, 0.4],
            [0.0, 2.0],
            [0.0, 1.8],
        )


def test_pvod_requires_more_than_one_row():
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        pvod_regression_metrics(
            [0.5],
            [0.4],
            [1.0],
            [0.8],
        )
