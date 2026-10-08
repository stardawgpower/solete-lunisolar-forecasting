"""
Astronomical feature engineering for the SOLETE forecasting project.

This module contains reusable solar and lunar calculations validated in
`notebooks/07_astronomical_feature_engineering.ipynb`.

The SOLETE 5-minute timestamps represent intervals. Based on the
cross-resolution validation in Notebook 07, astronomical quantities are
evaluated at the interval midpoint:

    astronomical reference time = SOLETE timestamp + 2 minutes

Skyfield/JPL DE440s provides the common Sun/Moon geometry used by both
the continuous astronomical feature family (F3) and the Panchang
transformations (F4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from skyfield.api import load, load_file
from skyfield.framelib import ecliptic_frame

# ============================================================================
# SOLETE SITE
# ============================================================================

SOLETE_LATITUDE_DEG = 55.6867
SOLETE_LONGITUDE_DEG = 12.0985

# Revised Notebook 07 freezes an explicit 0 m computational
# convention because a verified site altitude was not available.
SOLETE_SOLAR_ALTITUDE_CONVENTION_M = 0.0

# ============================================================================
# ASTRONOMICAL TIME CONVENTION
# ============================================================================

SOLETE_ASTRONOMY_OFFSET = pd.Timedelta(minutes=2)

def solete_astronomy_reference_times(
    timestamps: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Return UTC astronomical reference times for SOLETE timestamps.

    The validated 5-minute convention is:

        reference time = timestamp + 2 minutes
    """

    if timestamps.tz is None:
        raise ValueError(
            "SOLETE timestamps must be timezone-aware."
        )

    timestamps_utc = timestamps.tz_convert("UTC")

    return timestamps_utc + SOLETE_ASTRONOMY_OFFSET

def load_jpl_ephemeris(
    ephemeris_path: str | Path,
):
    """Load a JPL BSP ephemeris file with Skyfield."""

    ephemeris_path = Path(ephemeris_path)

    if not ephemeris_path.is_file():
        raise FileNotFoundError(
            f"Ephemeris file not found: {ephemeris_path}"
        )

    return load_file(str(ephemeris_path))

def skyfield_times_from_utc(
    timestamps: pd.DatetimeIndex,
):
    """Convert timezone-aware timestamps to a Skyfield Time array."""

    if timestamps.tz is None:
        raise ValueError(
            "Astronomical timestamps must be timezone-aware."
        )

    timestamps_utc = timestamps.tz_convert("UTC")

    timescale = load.timescale()

    return timescale.from_datetimes(
        timestamps_utc.to_pydatetime()
    )

def apparent_tropical_ecliptic_longitudes(
    ephemeris,
    skyfield_times,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return apparent geocentric tropical ecliptic longitudes of Sun and Moon.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sun longitude and Moon longitude in degrees, wrapped to [0, 360).
    """

    earth = ephemeris["earth"]
    sun = ephemeris["sun"]
    moon = ephemeris["moon"]

    earth_at_times = earth.at(skyfield_times)

    sun_apparent = earth_at_times.observe(sun).apparent()
    moon_apparent = earth_at_times.observe(moon).apparent()

    _, sun_longitude, _ = sun_apparent.frame_latlon(ecliptic_frame)
    _, moon_longitude, _ = moon_apparent.frame_latlon(ecliptic_frame)

    sun_longitude_deg = np.mod(sun_longitude.degrees, 360.0)
    moon_longitude_deg = np.mod(moon_longitude.degrees, 360.0)

    return sun_longitude_deg, moon_longitude_deg

def tropical_longitudes_for_solete(
    timestamps: pd.DatetimeIndex,
    ephemeris,
) -> pd.DataFrame:
    """
    Return apparent tropical Sun/Moon longitudes at SOLETE reference times.

    Calculations use timestamp + 2 minutes, while the returned DataFrame
    retains the original SOLETE timestamp index.
    """

    reference_times = solete_astronomy_reference_times(timestamps)
    skyfield_times = skyfield_times_from_utc(reference_times)

    sun_longitude_deg, moon_longitude_deg = (
        apparent_tropical_ecliptic_longitudes(
            ephemeris,
            skyfield_times,
        )
    )

    return pd.DataFrame(
        {
            "sun_tropical_longitude_deg": sun_longitude_deg,
            "moon_tropical_longitude_deg": moon_longitude_deg,
        },
        index=timestamps,
    )

def moon_sun_phase_angle_deg(
    sun_longitude_deg: float | np.ndarray | pd.Series,
    moon_longitude_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return Moon-Sun ecliptic longitude separation in [0, 360).

    phase = (Moon longitude - Sun longitude) mod 360
    """

    return np.mod(
        moon_longitude_deg - sun_longitude_deg,
        360.0,
    )

def phase_cyclic_features(
    phase_angle_deg: float | np.ndarray | pd.Series,
) -> tuple[
    float | np.ndarray | pd.Series,
    float | np.ndarray | pd.Series,
]:
    """Return sine and cosine encoding of Moon-Sun phase angle."""

    phase_rad = np.deg2rad(phase_angle_deg)

    return np.sin(phase_rad), np.cos(phase_rad)

def longitude_cyclic_features(
    longitude_deg: float | np.ndarray | pd.Series,
) -> tuple[
    float | np.ndarray | pd.Series,
    float | np.ndarray | pd.Series,
]:
    """Return sine and cosine encoding of an angular longitude."""

    longitude_rad = np.deg2rad(longitude_deg)

    return np.sin(longitude_rad), np.cos(longitude_rad)

def moon_ecliptic_latitude_and_distance(
    ephemeris,
    skyfield_times,
) -> tuple[np.ndarray, np.ndarray]:
    """Return apparent Moon ecliptic latitude (deg) and Earth-Moon distance (km)."""

    earth = ephemeris["earth"]
    moon = ephemeris["moon"]

    moon_apparent = earth.at(skyfield_times).observe(moon).apparent()

    moon_latitude, _, moon_distance = moon_apparent.frame_latlon(
        ecliptic_frame
    )

    return (
        moon_latitude.degrees,
        moon_distance.km,
    )

def moon_illumination_fraction(
    ephemeris,
    skyfield_times,
) -> np.ndarray:
    """Return the apparent illuminated fraction of the Moon in [0, 1]."""

    earth = ephemeris["earth"]
    moon = ephemeris["moon"]
    sun = ephemeris["sun"]

    moon_apparent = earth.at(skyfield_times).observe(moon).apparent()

    return np.asarray(
        moon_apparent.fraction_illuminated(sun)
    )

# ============================================================================
# HIGH-LEVEL F3 LUNAR FEATURE GENERATION
# ============================================================================

def build_lunar_f3_features(
    timestamps: pd.DatetimeIndex,
    ephemeris,
) -> pd.DataFrame:
    """
    Build the nine continuous lunar/Sun-Moon features used by F3.

    Astronomical calculations use the validated SOLETE reference time:
    timestamp + 2 minutes. The returned DataFrame keeps the original
    SOLETE timestamps as its index.
    """

    reference_times = solete_astronomy_reference_times(timestamps)
    skyfield_times = skyfield_times_from_utc(reference_times)

    # Apparent tropical ecliptic longitudes.
    sun_longitude_deg, moon_longitude_deg = (
        apparent_tropical_ecliptic_longitudes(
            ephemeris,
            skyfield_times,
        )
    )

    # Moon-Sun phase geometry.
    phase_angle_deg = moon_sun_phase_angle_deg(
        sun_longitude_deg,
        moon_longitude_deg,
    )

    phase_sin, phase_cos = phase_cyclic_features(
        phase_angle_deg
    )

    # Cyclic tropical longitudes.
    moon_longitude_sin, moon_longitude_cos = (
        longitude_cyclic_features(
            moon_longitude_deg
        )
    )

    sun_longitude_sin, sun_longitude_cos = (
        longitude_cyclic_features(
            sun_longitude_deg
        )
    )

    # Additional lunar geometry.
    moon_latitude_deg, moon_distance_km = (
        moon_ecliptic_latitude_and_distance(
            ephemeris,
            skyfield_times,
        )
    )

    moon_illumination = moon_illumination_fraction(
        ephemeris,
        skyfield_times,
    )

    return pd.DataFrame(
        {
            "moon_phase_sin": phase_sin,
            "moon_phase_cos": phase_cos,
            "moon_longitude_sin": moon_longitude_sin,
            "moon_longitude_cos": moon_longitude_cos,
            "sun_ecliptic_longitude_sin": sun_longitude_sin,
            "sun_ecliptic_longitude_cos": sun_longitude_cos,
            "moon_ecliptic_latitude_deg": moon_latitude_deg,
            "moon_illumination_fraction": moon_illumination,
            "earth_moon_distance_km": moon_distance_km,
        },
        index=timestamps,
    )

# ============================================================================
# SOLAR POSITION
# ============================================================================

def solar_position_spa(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute solar position at the validated SOLETE reference times.

    The returned DataFrame keeps the original SOLETE timestamps as its index.
    """

    reference_times = solete_astronomy_reference_times(timestamps)

    solar_position = pvlib.solarposition.get_solarposition(
        time=reference_times,
        latitude=SOLETE_LATITUDE_DEG,
        longitude=SOLETE_LONGITUDE_DEG,
        altitude=SOLETE_SOLAR_ALTITUDE_CONVENTION_M,
        method="nrel_numpy",
        delta_t=None,
    )

    # Preserve the original SOLETE timestamps rather than the +2 min
    # astronomical calculation timestamps.
    solar_position.index = timestamps

    return solar_position

def build_solar_f2_features(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the five deterministic solar-geometry features used by F2."""

    solar_position = solar_position_spa(timestamps)

    solar_elevation_deg = solar_position["elevation"]

    solar_cos_zenith = np.cos(
        np.deg2rad(solar_position["zenith"])
    )

    solar_azimuth_rad = np.deg2rad(
        solar_position["azimuth"]
    )

    solar_azimuth_sin = np.sin(
        solar_azimuth_rad
    )

    solar_azimuth_cos = np.cos(
        solar_azimuth_rad
    )

    solar_daylight = (
        solar_elevation_deg > 0
    ).astype("int8")

    return pd.DataFrame(
        {
            "solar_elevation_deg": solar_elevation_deg,
            "solar_cos_zenith": solar_cos_zenith,
            "solar_azimuth_sin": solar_azimuth_sin,
            "solar_azimuth_cos": solar_azimuth_cos,
            "solar_daylight": solar_daylight,
        },
        index=timestamps,
    )

# ============================================================================
# COMBINED ASTRONOMICAL FEATURE GENERATION
# ============================================================================

def build_astronomical_features(
    timestamps: pd.DatetimeIndex,
    ephemeris,
) -> pd.DataFrame:
    """
    Build the complete deterministic astronomical feature table.

    Returns the 5 solar F2 features followed by the 9 lunar/Sun-Moon
    features introduced in F3.
    """

    df_solar = build_solar_f2_features(
        timestamps
    )

    df_lunar = build_lunar_f3_features(
        timestamps,
        ephemeris,
    )

    return pd.concat(
        [df_solar, df_lunar],
        axis=1,
    )

