import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    build_gregorian_features,
    clean_humidity_fraction,
    clean_pressure_mbar,
    clean_wind_direction_deg,
    wind_direction_cyclic_features,
)


def test_pressure_artifact_correction():
    pressure = pd.Series([1012.0, 2015.0, 1008.5])

    result = clean_pressure_mbar(pressure)

    expected = pd.Series([1012.0, 1015.0, 1008.5])
    pd.testing.assert_series_equal(result, expected)


def test_humidity_anomaly_is_replaced_causally():
    humidity = pd.Series([0.50, 1.20, 0.70])

    result = clean_humidity_fraction(humidity)

    expected = pd.Series([0.50, 0.50, 0.70])
    pd.testing.assert_series_equal(result, expected)


def test_wind_direction_wraps_to_zero_360_interval():
    direction = pd.Series([-10.0, 0.0, 360.0, 370.0])

    result = clean_wind_direction_deg(direction)

    expected = pd.Series([350.0, 0.0, 0.0, 10.0])
    pd.testing.assert_series_equal(result, expected)


def test_wind_direction_cyclic_encoding():
    direction = pd.Series([0.0, 90.0])

    wind_sin, wind_cos = wind_direction_cyclic_features(direction)

    np.testing.assert_allclose(wind_sin, [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(wind_cos, [1.0, 0.0], atol=1e-12)


def test_gregorian_feature_family_has_expected_dimension():
    timestamps = pd.date_range(
        "2018-06-01 00:00:00",
        periods=3,
        freq="5min",
        tz="UTC",
    )

    features = build_gregorian_features(timestamps)

    assert features.shape == (3, 15)
    assert features.index.equals(timestamps)
    assert not features.isna().any().any()


def test_gregorian_features_require_timezone_aware_input():
    timestamps = pd.date_range(
        "2018-06-01 00:00:00",
        periods=2,
        freq="5min",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        build_gregorian_features(timestamps)
