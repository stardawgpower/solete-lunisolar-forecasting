import numpy as np
import pandas as pd
import pytest

from astronomy_features import (
    moon_sun_phase_angle_deg,
    phase_cyclic_features,
    solete_astronomy_reference_times,
)


def test_solete_astronomy_reference_time_is_plus_two_minutes():
    timestamps = pd.DatetimeIndex(
        [
            "2018-06-01 00:00:00+00:00",
            "2018-06-01 12:35:00+00:00",
        ]
    )

    result = solete_astronomy_reference_times(timestamps)

    expected = timestamps + pd.Timedelta(minutes=2)

    assert result.equals(expected)


def test_solete_reference_time_normalizes_to_utc():
    timestamps = pd.DatetimeIndex(
        ["2018-06-01 02:00:00+02:00"]
    )

    result = solete_astronomy_reference_times(timestamps)

    expected = pd.DatetimeIndex(
        ["2018-06-01 00:02:00+00:00"]
    )

    assert result.equals(expected)


def test_solete_reference_time_requires_timezone():
    timestamps = pd.DatetimeIndex(["2018-06-01 00:00:00"])

    with pytest.raises(ValueError, match="timezone-aware"):
        solete_astronomy_reference_times(timestamps)


def test_moon_sun_phase_wraps_to_full_circle():
    sun = np.array([350.0, 20.0, 100.0])
    moon = np.array([10.0, 10.0, 100.0])

    result = moon_sun_phase_angle_deg(sun, moon)

    np.testing.assert_allclose(result, [20.0, 350.0, 0.0])


def test_phase_cyclic_features_are_consistent():
    phase = np.array([0.0, 90.0, 180.0, 270.0])

    phase_sin, phase_cos = phase_cyclic_features(phase)

    np.testing.assert_allclose(
        phase_sin,
        [0.0, 1.0, 0.0, -1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        phase_cos,
        [1.0, 0.0, -1.0, 0.0],
        atol=1e-12,
    )
