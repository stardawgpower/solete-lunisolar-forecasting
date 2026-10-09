import pandas as pd
import pytest
import swisseph as swe

import astronomy_features
from vedic_calendar import (
    LAHIRI_AYANAMSHA_FLAGS,
    LAHIRI_SIDEREAL_MODE,
    lahiri_ayanamsha_extended_deg,
)


def test_frozen_lahiri_nonut_flags():
    # Prevent accidental drift back to the pre-Notebook-08 nutation convention.
    assert LAHIRI_SIDEREAL_MODE == swe.SIDM_LAHIRI
    assert LAHIRI_AYANAMSHA_FLAGS == (swe.FLG_SWIEPH | swe.FLG_NONUT)


def test_frozen_lahiri_nonut_reference_value():
    # Notebook 08's first sample: 2018-06-01 00:02 UTC.
    jd_ut = swe.julday(2018, 6, 1, 2.0 / 60.0)
    ayanamsha_deg, return_flags = lahiri_ayanamsha_extended_deg(jd_ut)
    assert ayanamsha_deg == pytest.approx(
        24.11431760121269, abs=1e-7
    )
    assert return_flags & swe.FLG_NONUT


def test_solar_spa_uses_explicit_notebook07_conventions(monkeypatch):
    requested_times = pd.DatetimeIndex([
        "2018-06-01 00:00:00+00:00",
        "2018-06-01 12:35:00+00:00",
    ])
    seen = {}

    def fake_get_solarposition(
        *,
        time,
        latitude,
        longitude,
        altitude,
        method,
        delta_t,
    ):
        seen.update({
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "method": method,
            "delta_t": delta_t,
        })
        return pd.DataFrame(
            {"elevation": [5.0, 20.0]},
            index=time,
        )

    monkeypatch.setattr(
        astronomy_features.pvlib.solarposition,
        "get_solarposition",
        fake_get_solarposition,
    )

    output = astronomy_features.solar_position_spa(requested_times)

    assert seen["time"].equals(
        requested_times + pd.Timedelta(minutes=2)
    )
    assert seen["latitude"] == pytest.approx(55.6867)
    assert seen["longitude"] == pytest.approx(12.0985)
    assert seen["altitude"] == 0.0
    assert seen["method"] == "nrel_numpy"
    assert seen["delta_t"] is None
    assert output.index.equals(requested_times)
