"""
Preprocessing utilities for the SOLETE forecasting project.

This module contains reusable cleaning rules validated in
`notebooks/06_preprocessing_feature_engineering.ipynb`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# PRESSURE
# ============================================================================

def clean_pressure_mbar(
    pressure_mbar: pd.Series,
) -> pd.Series:
    """
    Correct the validated SOLETE pressure artifact.

    Values above 1100 mbar correspond to the midnight +1000 mbar anomaly
    identified in Notebook 06 and are corrected by subtracting 1000.
    """

    cleaned = pressure_mbar.copy()

    anomaly_mask = cleaned > 1100.0

    cleaned.loc[anomaly_mask] = (
        cleaned.loc[anomaly_mask] - 1000.0
    )

    return cleaned

# ============================================================================
# WIND DIRECTION
# ============================================================================

def clean_wind_direction_deg(
    wind_direction_deg: pd.Series,
) -> pd.Series:
    """Wrap wind direction into the interval [0, 360)."""

    return wind_direction_deg % 360.0

def wind_direction_cyclic_features(
    wind_direction_deg: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return sine and cosine encoding of wind direction."""

    angle_rad = np.deg2rad(wind_direction_deg)

    wind_sin = pd.Series(
        np.sin(angle_rad),
        index=wind_direction_deg.index,
        name="wind_direction_sin",
    )

    wind_cos = pd.Series(
        np.cos(angle_rad),
        index=wind_direction_deg.index,
        name="wind_direction_cos",
    )

    return wind_sin, wind_cos

# ============================================================================
# HUMIDITY
# ============================================================================

def clean_humidity_fraction(
    humidity: pd.Series,
) -> pd.Series:
    """
    Replace SOLETE humidity values above 1.0 and fill causally.

    Values > 1.0 are treated as anomalies and replaced using forward-fill.
    """

    cleaned = humidity.copy()

    anomaly_mask = cleaned > 1.0
    cleaned.loc[anomaly_mask] = np.nan

    return cleaned.ffill()

# ============================================================================
# P_GAIA QUALITY DIAGNOSTIC
# ============================================================================

def gaia_quality_flag(
    reconstruction_error_kw: pd.Series,
    threshold_kw: float = 3.0,
) -> pd.Series:
    """
    Flag suspicious midnight P_Gaia observations using reconstruction error.

    This is an offline diagnostic flag only and must not be used as a
    forecasting feature.
    """

    if not isinstance(reconstruction_error_kw.index, pd.DatetimeIndex):
        raise TypeError(
            "reconstruction_error_kw must use a DatetimeIndex."
        )

    midnight_mask = (
        (reconstruction_error_kw.index.hour == 0)
        & (reconstruction_error_kw.index.minute == 0)
    )

    return pd.Series(
        midnight_mask
        & (reconstruction_error_kw.abs() > threshold_kw),
        index=reconstruction_error_kw.index,
        name="gaia_quality_flag",
    )

# ============================================================================
# GREGORIAN TIME FEATURES
# ============================================================================

def build_gregorian_calendar_features(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the basic UTC Gregorian calendar features."""

    if timestamps.tz is None:
        raise ValueError(
            "Gregorian feature timestamps must be timezone-aware."
        )

    index = timestamps.tz_convert("UTC")

    utc_year = index.year
    utc_month = index.month
    utc_day = index.day
    utc_day_of_week = index.dayofweek
    utc_day_of_year = index.dayofyear
    utc_hour = index.hour
    utc_minute = index.minute

    utc_minute_of_day = (
        utc_hour * 60
        + utc_minute
    )

    utc_is_weekend = (
        pd.Index(utc_day_of_week)
        .isin([5, 6])
        .astype("int8")
    )

    return pd.DataFrame(
        {
            "utc_year": utc_year,
            "utc_month": utc_month,
            "utc_day": utc_day,
            "utc_day_of_week": utc_day_of_week,
            "utc_day_of_year": utc_day_of_year,
            "utc_hour": utc_hour,
            "utc_minute": utc_minute,
            "utc_minute_of_day": utc_minute_of_day,
            "utc_is_weekend": utc_is_weekend,
        },
        index=timestamps,
    )

def build_gregorian_cyclic_features(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build cyclic UTC daily, weekly, and annual Gregorian features."""

    if timestamps.tz is None:
        raise ValueError(
            "Gregorian feature timestamps must be timezone-aware."
        )

    index = timestamps.tz_convert("UTC")

    minute_of_day = (
        index.hour * 60
        + index.minute
    )

    # Daily cycle
    time_phase = (
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    # Weekly cycle
    weekday_phase = (
        2.0
        * np.pi
        * index.dayofweek
        / 7.0
    )

    # Annual cycle
    days_in_year = np.where(
        index.is_leap_year,
        366.0,
        365.0,
    )

    fractional_day_of_year = (
        (index.dayofyear - 1)
        + minute_of_day / 1440.0
    )

    year_phase = (
        2.0
        * np.pi
        * fractional_day_of_year
        / days_in_year
    )

    return pd.DataFrame(
        {
            "utc_time_sin": np.sin(time_phase),
            "utc_time_cos": np.cos(time_phase),
            "utc_weekday_sin": np.sin(weekday_phase),
            "utc_weekday_cos": np.cos(weekday_phase),
            "utc_year_sin": np.sin(year_phase),
            "utc_year_cos": np.cos(year_phase),
        },
        index=timestamps,
    )

def build_gregorian_features(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the complete Gregorian feature family used by F1."""

    basic = build_gregorian_calendar_features(
        timestamps
    )

    cyclic = build_gregorian_cyclic_features(
        timestamps
    )

    return pd.concat(
        [basic, cyclic],
        axis=1,
    )

# ============================================================================
# HIGH-LEVEL SOLETE CLEANING
# ============================================================================

def build_cleaned_solete_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the validated SOLETE cleaning rules used in Notebook 06.

    The original columns are preserved. Cleaned/model-ready columns are added.
    """

    required_columns = [
        "Pressure[mbar]",
        "HUMIDITY[%]",
        "WIND_DIR[deg]",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required SOLETE columns: {missing_columns}"
        )

    cleaned = df.copy()

    # Pressure correction.
    cleaned["Pressure_clean[mbar]"] = clean_pressure_mbar(
        cleaned["Pressure[mbar]"]
    )

    # Humidity correction.
    cleaned["Humidity_clean"] = clean_humidity_fraction(
        cleaned["HUMIDITY[%]"]
    )

    # Wind direction wrapping and circular encoding.
    cleaned["Wind_dir_clean[deg]"] = clean_wind_direction_deg(
        cleaned["WIND_DIR[deg]"]
    )

    wind_sin, wind_cos = wind_direction_cyclic_features(
        cleaned["Wind_dir_clean[deg]"]
    )

    cleaned["wind_direction_sin"] = wind_sin
    cleaned["wind_direction_cos"] = wind_cos

    return cleaned

# ============================================================================
# COMBINED PREPROCESSING + GREGORIAN FEATURES
# ============================================================================

def build_preprocessed_solete_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply validated SOLETE cleaning and add Gregorian F1 features.

    The original SOLETE columns are preserved.
    """

    cleaned = build_cleaned_solete_features(df)

    gregorian = build_gregorian_features(
        cleaned.index
    )

    return pd.concat(
        [cleaned, gregorian],
        axis=1,
    )

