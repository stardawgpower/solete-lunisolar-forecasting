"""
Vedic / Panchang calendar feature engineering.

This module contains reusable transformations validated in
`notebooks/08_vedic_calendar_features.ipynb`.

Scientific design
-----------------
The module does NOT calculate Skyfield Sun/Moon positions itself.

The intended pipeline is:

    astronomical geometry
        -> tropical Sun/Moon ecliptic longitudes
        -> Panchang transformations in this module

This preserves the controlled experimental distinction:

    F3 = continuous astronomical geometry
    F4 = structured Panchang representation

Canonical sidereal convention
-----------------------------
For ayanamsha-dependent quantities, the project uses:

    Swiss Ephemeris
    SIDM_LAHIRI
    get_ayanamsa_ex_ut()

with nutation included.

Canonical sidereal longitude is constructed as:

    tropical longitude - extended Lahiri ayanamsha

wrapped into [0, 360).

The dataset-specific SOLETE astronomical reference-time convention
(timestamp + 2 minutes) belongs to the astronomy/data pipeline and is
not hard-coded in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import swisseph as swe

# ============================================================================
# SIDEREAL CONVENTION
# ============================================================================

LAHIRI_SIDEREAL_MODE = swe.SIDM_LAHIRI

# Requested Swiss Ephemeris flag for the extended ayanamsha calculation.
# The actual returned calculation flag should still be retained/checked
# when the calculation is performed.
LAHIRI_AYANAMSHA_FLAGS = swe.FLG_SWIEPH

# ============================================================================
# PANCHANG ANGULAR CONSTANTS
# ============================================================================

FULL_CIRCLE_DEG = 360.0

TITHI_COUNT = 30
TITHI_WIDTH_DEG = FULL_CIRCLE_DEG / TITHI_COUNT

KARANA_SLOT_COUNT = 60
KARANA_WIDTH_DEG = FULL_CIRCLE_DEG / KARANA_SLOT_COUNT

NAKSHATRA_COUNT = 27
NAKSHATRA_WIDTH_DEG = FULL_CIRCLE_DEG / NAKSHATRA_COUNT

YOGA_COUNT = 27
YOGA_WIDTH_DEG = FULL_CIRCLE_DEG / YOGA_COUNT

# ============================================================================
# GENERIC ANGLE HELPERS
# ============================================================================

def wrap_degrees(
    angle_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Wrap an angle or array-like collection of angles into [0, 360).

    Parameters
    ----------
    angle_deg
        Angle value(s) in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Wrapped angle value(s) in the interval [0, 360).

    Notes
    -----
    The implementation deliberately uses modulo arithmetic only:

        angle % 360

    so that values such as:

        -1°   -> 359°
        360°  -> 0°
        721°  -> 1°

    are handled consistently.
    """

    return angle_deg % FULL_CIRCLE_DEG

def circular_difference_deg(
    angle_a: float | np.ndarray | pd.Series,
    angle_b: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return the absolute smallest circular difference between angles.

    Parameters
    ----------
    angle_a
        First angle or collection of angles in degrees.

    angle_b
        Second angle or collection of angles in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Absolute smallest angular separation in degrees,
        always in the interval [0, 180].

    Examples
    --------
    10° and 20° differ by:

        10°

    359° and 1° differ by:

        2°

    rather than 358°.
    """

    return np.abs(
        (
            angle_a
            - angle_b
            + 180.0
        )
        % FULL_CIRCLE_DEG
        - 180.0
    )

# ============================================================================
# TITHI
# ============================================================================

def tithi_number_from_phase(
    phase_angle_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert Moon-Sun phase angle into the traditional Tithi number.

    Parameters
    ----------
    phase_angle_deg
        Moon-Sun angular separation in degrees:

            (Moon longitude - Sun longitude) mod 360

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Tithi number in the range 1..30.

    Notes
    -----
    The lunar circle is divided into 30 equal Tithi sectors:

        360° / 30 = 12°

    Sector convention:

        [0°, 12°)     -> Tithi 1
        [12°, 24°)    -> Tithi 2
        ...
        [348°, 360°)  -> Tithi 30

    The input is wrapped into [0°, 360°) before sector assignment, so
    360° is treated as 0° and therefore maps to Tithi 1.
    """

    phase = wrap_degrees(phase_angle_deg)

    return np.floor(phase / TITHI_WIDTH_DEG).astype(int) + 1

def tithi_progress_from_phase(
    phase_angle_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return fractional progress through the current Tithi.

    Parameters
    ----------
    phase_angle_deg
        Moon-Sun angular separation in degrees:

            (Moon longitude - Sun longitude) mod 360

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Fractional progress through the current Tithi,
        in the interval [0, 1).

    Notes
    -----
    Each Tithi spans 12 degrees.

    Examples:

        0°   -> 0.0
        6°   -> 0.5
        12°  -> 0.0   (start of the next Tithi)
        18°  -> 0.5
    """

    phase = wrap_degrees(phase_angle_deg)

    return (phase % TITHI_WIDTH_DEG) / TITHI_WIDTH_DEG

# ============================================================================
# PAKSHA
# ============================================================================

def paksha_from_tithi(
    tithi_number: int | np.ndarray | pd.Series,
) -> str | np.ndarray | pd.Series:
    """
    Convert Tithi number into Paksha.

    Parameters
    ----------
    tithi_number
        Tithi number(s) in the range 1..30.

    Returns
    -------
    str, numpy.ndarray, or pandas.Series
        Paksha assignment:

        Tithi 1..15   -> "Shukla"
        Tithi 16..30  -> "Krishna"

    Notes
    -----
    In the convention used throughout this project:

        Tithi 15 = Purnima
        Tithi 30 = Amavasya
    """

    return np.where(
        tithi_number <= 15,
        "Shukla",
        "Krishna",
    )

def tithi_within_paksha(
    tithi_number: int | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert absolute Tithi number 1..30 into position within its Paksha.

    Parameters
    ----------
    tithi_number
        Tithi number(s) in the range 1..30.

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Tithi position within the current Paksha, in the range 1..15.

    Mapping
    -------
    Shukla Paksha:

        Tithi 1..15  -> 1..15

    Krishna Paksha:

        Tithi 16..30 -> 1..15
    """

    return ((tithi_number - 1) % 15) + 1


# ============================================================================
# KARANA
# ============================================================================

MOVABLE_KARANAS = (
    "Bava",
    "Balava",
    "Kaulava",
    "Taitila",
    "Gara",
    "Vanija",
    "Vishti",
)

KARANA_SEQUENCE = (
    ("Kimstughna",)
    + MOVABLE_KARANAS * 8
    + ("Shakuni", "Chatushpada", "Naga")
)

def karana_slot_from_phase(
    phase_angle_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert Moon-Sun phase angle into Karana slot number.

    Parameters
    ----------
    phase_angle_deg
        Moon-Sun angular separation in degrees:

            (Moon longitude - Sun longitude) mod 360

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Karana slot number in the range 1..60.

    Notes
    -----
    A lunar cycle contains 60 half-Tithi Karana slots:

        360° / 60 = 6°

    Sector convention:

        [0°, 6°)      -> slot 1
        [6°, 12°)     -> slot 2
        ...
        [354°, 360°)  -> slot 60

    The input is wrapped into [0°, 360°), so 360° maps back to slot 1.
    """

    phase = wrap_degrees(phase_angle_deg)

    return np.floor(phase / KARANA_WIDTH_DEG).astype(int) + 1

def karana_name_from_slot(
    karana_slot: int | np.ndarray | pd.Series,
) -> str | np.ndarray | pd.Series:
    """
    Convert Karana slot number 1..60 into the traditional Karana name.

    Sequence
    --------
    Slot 1:
        Kimstughna

    Slots 2..57:
        Repeating movable sequence

        Bava
        Balava
        Kaulava
        Taitila
        Gara
        Vanija
        Vishti

        repeated eight times.

    Final fixed slots:
        58 -> Shakuni
        59 -> Chatushpada
        60 -> Naga
    """

    slot_array = np.asarray(karana_slot)

    names = np.asarray(KARANA_SEQUENCE, dtype=object)

    result = names[slot_array - 1]

    if np.ndim(karana_slot) == 0:
        return str(result)

    if isinstance(karana_slot, pd.Series):
        return pd.Series(
            result,
            index=karana_slot.index,
            name=karana_slot.name,
        )

    return result

def karana_half_within_tithi(
    karana_slot: int | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Return whether a Karana occupies the first or second half of a Tithi.

    Parameters
    ----------
    karana_slot
        Karana slot number(s) in the range 1..60.

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Half position within the corresponding Tithi:

        odd-numbered Karana slot  -> 1
        even-numbered Karana slot -> 2

    Examples
    --------
    Slot 1 -> first half of Tithi 1
    Slot 2 -> second half of Tithi 1
    Slot 3 -> first half of Tithi 2
    Slot 4 -> second half of Tithi 2
    """

    return ((karana_slot - 1) % 2) + 1

def karana_progress_from_phase(
    phase_angle_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return fractional progress through the current Karana slot.

    Parameters
    ----------
    phase_angle_deg
        Moon-Sun angular separation in degrees:

            (Moon longitude - Sun longitude) mod 360

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Fractional progress through the current 6-degree Karana slot,
        in the interval [0, 1).

    Examples
    --------
    0°  -> 0.0
    3°  -> 0.5
    6°  -> 0.0   (start of the next Karana)
    9°  -> 0.5
    """

    phase = wrap_degrees(phase_angle_deg)

    return (phase % KARANA_WIDTH_DEG) / KARANA_WIDTH_DEG

# ============================================================================
# LAHIRI SIDEREAL TRANSFORMATION
# ============================================================================

def lahiri_ayanamsha_extended_deg(
    julian_day_ut: float,
) -> tuple[float, int]:
    """
    Compute the canonical extended Lahiri ayanamsha for one UT Julian day.

    Parameters
    ----------
    julian_day_ut
        Julian Day expressed in Universal Time (UT).

    Returns
    -------
    tuple[float, int]
        A two-element tuple containing:

        1. extended Lahiri ayanamsha in degrees;
        2. Swiss Ephemeris return flags.

    Scientific convention
    ---------------------
    The project uses:

        swe.SIDM_LAHIRI
        swe.get_ayanamsa_ex_ut()
        swe.FLG_SWIEPH

    without FLG_NONUT.

    Therefore the returned extended ayanamsha includes nutation.

    This is the canonical convention validated in Notebook 08 for
    transforming Skyfield apparent tropical ecliptic longitudes into
    Lahiri sidereal longitudes.

    Notes
    -----
    Swiss Ephemeris sidereal mode is global library state, so the Lahiri
    mode is explicitly set inside this function before every calculation.
    This makes the function deterministic even if another calculation
    changed the sidereal mode earlier in the Python process.
    """

    swe.set_sid_mode(LAHIRI_SIDEREAL_MODE)

    return_flags, ayanamsha_deg = swe.get_ayanamsa_ex_ut(
        float(julian_day_ut),
        LAHIRI_AYANAMSHA_FLAGS,
    )

    return float(ayanamsha_deg), int(return_flags)

def lahiri_ayanamsha_for_utc_times(
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Compute canonical extended Lahiri ayanamsha for UTC timestamps."""

    if timestamps.tz is None:
        raise ValueError(
            "Ayanamsha timestamps must be timezone-aware."
        )

    timestamps_utc = timestamps.tz_convert("UTC")

    ayanamsha_values = np.empty(
        len(timestamps_utc),
        dtype=float,
    )

    for i, timestamp in enumerate(timestamps_utc):
        hour_decimal = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
            + timestamp.microsecond / 3_600_000_000.0
        )

        julian_day_ut = swe.julday(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            hour_decimal,
            swe.GREG_CAL,
        )

        ayanamsha_values[i], _ = lahiri_ayanamsha_extended_deg(
            julian_day_ut
        )

    return ayanamsha_values

def sidereal_longitude_from_tropical(
    tropical_longitude_deg: float | np.ndarray | pd.Series,
    ayanamsha_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Convert tropical ecliptic longitude into Lahiri sidereal longitude.

    Parameters
    ----------
    tropical_longitude_deg
        Tropical ecliptic longitude in degrees.

    ayanamsha_deg
        Lahiri ayanamsha in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Sidereal longitude wrapped into [0, 360).

    Scientific convention
    ---------------------
    The project uses:

        sidereal longitude
        =
        tropical longitude
        - Lahiri ayanamsha

    followed by circular wrapping into [0, 360).

    In the canonical Notebook 08 pipeline:

        tropical longitude
        =
        Skyfield apparent tropical ecliptic longitude

        ayanamsha
        =
        extended Lahiri ayanamsha from get_ayanamsa_ex_ut()
    """

    return wrap_degrees(
        tropical_longitude_deg - ayanamsha_deg
    )

# ============================================================================
# NAKSHATRA
# ============================================================================

NAKSHATRA_NAMES = (
    "Ashwini",
    "Bharani",
    "Krittika",
    "Rohini",
    "Mrigashirsha",
    "Ardra",
    "Punarvasu",
    "Pushya",
    "Ashlesha",
    "Magha",
    "Purva Phalguni",
    "Uttara Phalguni",
    "Hasta",
    "Chitra",
    "Swati",
    "Vishakha",
    "Anuradha",
    "Jyeshtha",
    "Mula",
    "Purva Ashadha",
    "Uttara Ashadha",
    "Shravana",
    "Dhanishta",
    "Shatabhisha",
    "Purva Bhadrapada",
    "Uttara Bhadrapada",
    "Revati",
)

def nakshatra_number_from_sidereal_moon(
    moon_sidereal_longitude_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert Lahiri sidereal Moon longitude into Nakshatra number.

    Parameters
    ----------
    moon_sidereal_longitude_deg
        Moon's sidereal ecliptic longitude in degrees.

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Nakshatra number in the range 1..27.

    Notes
    -----
    The sidereal zodiac is divided into 27 equal Nakshatra sectors:

        360° / 27 = 13°20′

    Sector convention:

        [0°, 13°20′)      -> Nakshatra 1
        [13°20′, 26°40′)  -> Nakshatra 2
        ...
        final sector       -> Nakshatra 27

    The input is wrapped into [0°, 360°), so 360° maps back to
    Nakshatra 1.
    """

    longitude = wrap_degrees(moon_sidereal_longitude_deg)

    return np.floor(
        longitude / NAKSHATRA_WIDTH_DEG
    ).astype(int) + 1

def nakshatra_name_from_number(
    nakshatra_number: int | np.ndarray | pd.Series,
) -> str | np.ndarray | pd.Series:
    """
    Convert Nakshatra number 1..27 into its traditional name.

    Parameters
    ----------
    nakshatra_number
        Nakshatra number(s) in the range 1..27.

    Returns
    -------
    str, numpy.ndarray, or pandas.Series
        Corresponding Nakshatra name(s).
    """

    number_array = np.asarray(nakshatra_number)

    names = np.asarray(NAKSHATRA_NAMES, dtype=object)

    result = names[number_array - 1]

    if np.ndim(nakshatra_number) == 0:
        return str(result)

    if isinstance(nakshatra_number, pd.Series):
        return pd.Series(
            result,
            index=nakshatra_number.index,
            name=nakshatra_number.name,
        )

    return result

def nakshatra_position_deg(
    moon_sidereal_longitude_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return angular position within the current Nakshatra.

    Parameters
    ----------
    moon_sidereal_longitude_deg
        Moon's sidereal ecliptic longitude in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Position within the current Nakshatra sector, in degrees.

    Notes
    -----
    Each Nakshatra spans:

        360° / 27 = 13°20′

    Therefore the returned value lies in:

        [0°, 13°20′)

    Examples
    --------
    0°          -> 0°
    5°          -> 5°
    13°20′      -> 0°
    """

    longitude = wrap_degrees(moon_sidereal_longitude_deg)

    return longitude % NAKSHATRA_WIDTH_DEG

def nakshatra_progress_from_sidereal_moon(
    moon_sidereal_longitude_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return fractional progress through the current Nakshatra.

    Parameters
    ----------
    moon_sidereal_longitude_deg
        Moon's sidereal ecliptic longitude in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Fractional progress through the current Nakshatra,
        in the interval [0, 1).

    Notes
    -----
    Each Nakshatra spans:

        360° / 27 = 13°20′

    Examples
    --------
    0°                     -> 0.0
    half a Nakshatra       -> 0.5
    exact sector boundary  -> 0.0
    """

    longitude = wrap_degrees(moon_sidereal_longitude_deg)

    return (
        longitude % NAKSHATRA_WIDTH_DEG
    ) / NAKSHATRA_WIDTH_DEG

def nakshatra_number_from_tropical_moon(
    moon_tropical_longitude_deg: float | np.ndarray | pd.Series,
    ayanamsha_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert tropical Moon longitude directly into Nakshatra number.

    Parameters
    ----------
    moon_tropical_longitude_deg
        Moon's apparent tropical ecliptic longitude in degrees.

    ayanamsha_deg
        Canonical extended Lahiri ayanamsha in degrees.

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Nakshatra number in the range 1..27.

    Notes
    -----
    This is a convenience composition of two already-defined steps:

        tropical Moon longitude
            -> Lahiri sidereal Moon longitude
            -> Nakshatra number

    It deliberately does not calculate astronomical geometry itself.
    """

    moon_sidereal_longitude_deg = sidereal_longitude_from_tropical(
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    return nakshatra_number_from_sidereal_moon(
        moon_sidereal_longitude_deg
    )

# ============================================================================
# YOGA
# ============================================================================

YOGA_NAMES = (
    "Vishkambha",
    "Priti",
    "Ayushman",
    "Saubhagya",
    "Shobhana",
    "Atiganda",
    "Sukarma",
    "Dhriti",
    "Shula",
    "Ganda",
    "Vriddhi",
    "Dhruva",
    "Vyaghata",
    "Harshana",
    "Vajra",
    "Siddhi",
    "Vyatipata",
    "Variyana",
    "Parigha",
    "Shiva",
    "Siddha",
    "Sadhya",
    "Shubha",
    "Shukla",
    "Brahma",
    "Indra",
    "Vaidhriti",
)

def yoga_angle_from_sidereal_longitudes(
    sun_sidereal_longitude_deg: float | np.ndarray | pd.Series,
    moon_sidereal_longitude_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute the Panchang Yoga angle from sidereal Sun and Moon longitudes.

    Parameters
    ----------
    sun_sidereal_longitude_deg
        Sun's Lahiri sidereal ecliptic longitude in degrees.

    moon_sidereal_longitude_deg
        Moon's Lahiri sidereal ecliptic longitude in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Yoga angle wrapped into [0, 360).

    Notes
    -----
    The traditional Yoga angle is:

        (sidereal Sun longitude + sidereal Moon longitude) mod 360°
    """

    return wrap_degrees(
        sun_sidereal_longitude_deg
        + moon_sidereal_longitude_deg
    )

def yoga_angle_from_tropical_longitudes(
    sun_tropical_longitude_deg: float | np.ndarray | pd.Series,
    moon_tropical_longitude_deg: float | np.ndarray | pd.Series,
    ayanamsha_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute Panchang Yoga angle directly from tropical Sun and Moon longitudes.

    Parameters
    ----------
    sun_tropical_longitude_deg
        Sun's apparent tropical ecliptic longitude in degrees.

    moon_tropical_longitude_deg
        Moon's apparent tropical ecliptic longitude in degrees.

    ayanamsha_deg
        Canonical extended Lahiri ayanamsha in degrees.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Yoga angle wrapped into [0, 360).

    Notes
    -----
    The calculation is equivalent to:

        sidereal Sun = tropical Sun - ayanamsha
        sidereal Moon = tropical Moon - ayanamsha

        Yoga angle
        =
        (sidereal Sun + sidereal Moon) mod 360

    therefore:

        Yoga angle
        =
        (tropical Sun + tropical Moon - 2 * ayanamsha) mod 360

    Unlike Tithi, the ayanamsha does not cancel in Yoga.
    """

    sun_sidereal_longitude_deg = sidereal_longitude_from_tropical(
        sun_tropical_longitude_deg,
        ayanamsha_deg,
    )

    moon_sidereal_longitude_deg = sidereal_longitude_from_tropical(
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    return yoga_angle_from_sidereal_longitudes(
        sun_sidereal_longitude_deg,
        moon_sidereal_longitude_deg,
    )

def yoga_number_from_angle(
    yoga_angle_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert Panchang Yoga angle into Yoga number.

    Parameters
    ----------
    yoga_angle_deg
        Yoga angle in degrees:

            (sidereal Sun longitude + sidereal Moon longitude) mod 360

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Yoga number in the range 1..27.

    Notes
    -----
    The full circle is divided into 27 equal Yoga sectors:

        360° / 27 = 13°20′

    Sector convention:

        [0°, 13°20′)      -> Yoga 1
        [13°20′, 26°40′)  -> Yoga 2
        ...
        final sector       -> Yoga 27

    The input is wrapped into [0°, 360°), so 360° maps back to Yoga 1.
    """

    angle = wrap_degrees(yoga_angle_deg)

    return np.floor(
        angle / YOGA_WIDTH_DEG
    ).astype(int) + 1

def yoga_name_from_number(
    yoga_number: int | np.ndarray | pd.Series,
) -> str | np.ndarray | pd.Series:
    """
    Convert Yoga number 1..27 into its traditional name.

    Parameters
    ----------
    yoga_number
        Yoga number(s) in the range 1..27.

    Returns
    -------
    str, numpy.ndarray, or pandas.Series
        Corresponding Yoga name(s).
    """

    number_array = np.asarray(yoga_number)

    names = np.asarray(YOGA_NAMES, dtype=object)

    result = names[number_array - 1]

    if np.ndim(yoga_number) == 0:
        return str(result)

    if isinstance(yoga_number, pd.Series):
        return pd.Series(
            result,
            index=yoga_number.index,
            name=yoga_number.name,
        )

    return result

def yoga_position_deg(
    yoga_angle_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return angular position within the current Yoga sector.

    Parameters
    ----------
    yoga_angle_deg
        Panchang Yoga angle in degrees:

            (sidereal Sun longitude + sidereal Moon longitude) mod 360

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Position within the current Yoga sector, in degrees.

    Notes
    -----
    Each Yoga spans:

        360° / 27 = 13°20′

    Therefore the returned value lies in:

        [0°, 13°20′)

    Examples
    --------
    0°          -> 0°
    5°          -> 5°
    13°20′      -> 0°
    """

    angle = wrap_degrees(yoga_angle_deg)

    return angle % YOGA_WIDTH_DEG

def yoga_progress_from_angle(
    yoga_angle_deg: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Return fractional progress through the current Yoga.

    Parameters
    ----------
    yoga_angle_deg
        Panchang Yoga angle in degrees:

            (sidereal Sun longitude + sidereal Moon longitude) mod 360

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        Fractional progress through the current Yoga,
        in the interval [0, 1).

    Notes
    -----
    Each Yoga spans:

        360° / 27 = 13°20′

    Examples
    --------
    0°                  -> 0.0
    half a Yoga sector  -> 0.5
    exact boundary      -> 0.0
    """

    return yoga_position_deg(yoga_angle_deg) / YOGA_WIDTH_DEG

def yoga_number_from_tropical_longitudes(
    sun_tropical_longitude_deg: float | np.ndarray | pd.Series,
    moon_tropical_longitude_deg: float | np.ndarray | pd.Series,
    ayanamsha_deg: float | np.ndarray | pd.Series,
) -> int | np.ndarray | pd.Series:
    """
    Convert tropical Sun and Moon longitudes directly into Yoga number.

    Parameters
    ----------
    sun_tropical_longitude_deg
        Sun's apparent tropical ecliptic longitude in degrees.

    moon_tropical_longitude_deg
        Moon's apparent tropical ecliptic longitude in degrees.

    ayanamsha_deg
        Canonical extended Lahiri ayanamsha in degrees.

    Returns
    -------
    int, numpy.ndarray, or pandas.Series
        Yoga number in the range 1..27.

    Notes
    -----
    This is a convenience composition of:

        tropical Sun/Moon longitudes
            -> Lahiri sidereal longitudes
            -> Yoga angle
            -> Yoga number
    """

    yoga_angle_deg = yoga_angle_from_tropical_longitudes(
        sun_tropical_longitude_deg,
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    return yoga_number_from_angle(yoga_angle_deg)

def yoga_name_from_tropical_longitudes(
    sun_tropical_longitude_deg: float | np.ndarray | pd.Series,
    moon_tropical_longitude_deg: float | np.ndarray | pd.Series,
    ayanamsha_deg: float | np.ndarray | pd.Series,
) -> str | np.ndarray | pd.Series:
    """
    Convert tropical Sun and Moon longitudes directly into Yoga name.

    Parameters
    ----------
    sun_tropical_longitude_deg
        Sun's apparent tropical ecliptic longitude in degrees.

    moon_tropical_longitude_deg
        Moon's apparent tropical ecliptic longitude in degrees.

    ayanamsha_deg
        Canonical extended Lahiri ayanamsha in degrees.

    Returns
    -------
    str, numpy.ndarray, or pandas.Series
        Corresponding traditional Yoga name(s).

    Notes
    -----
    This is a convenience composition of:

        tropical Sun/Moon longitudes
            -> Yoga angle
            -> Yoga number
            -> Yoga name
    """

    yoga_number = yoga_number_from_tropical_longitudes(
        sun_tropical_longitude_deg,
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    return yoga_name_from_number(yoga_number)


# ============================================================================
# HIGH-LEVEL F4 FEATURE GENERATION
# ============================================================================

def build_panchang_f4_raw(
    sun_tropical_longitude_deg: pd.Series,
    moon_tropical_longitude_deg: pd.Series,
    ayanamsha_deg: pd.Series,
) -> pd.DataFrame:
    """
    Build the canonical raw F4 Panchang feature table.

    Parameters
    ----------
    sun_tropical_longitude_deg
        Pandas Series containing the Sun's apparent tropical ecliptic
        longitude in degrees.

    moon_tropical_longitude_deg
        Pandas Series containing the Moon's apparent tropical ecliptic
        longitude in degrees.

    ayanamsha_deg
        Pandas Series containing the canonical extended Lahiri
        ayanamsha in degrees.

    Returns
    -------
    pandas.DataFrame
        Five-column categorical Panchang feature table containing:

        - tithi
        - paksha
        - karana
        - nakshatra
        - yoga

    Scientific design
    -----------------
    This function constructs only the raw categorical F4 feature family.

    Continuous astronomical quantities and intermediate Panchang
    quantities are deliberately excluded so that the controlled
    experimental distinction remains:

        F3 = continuous astronomy
        F4 = structured Panchang categories

    Notes
    -----
    All three input Series must have exactly the same index.

    Tithi, Paksha and Karana are derived from the tropical Moon-Sun
    longitude difference. A common ayanamsha would cancel from this
    difference.

    Nakshatra and Yoga use the canonical extended Lahiri ayanamsha.
    """

    # ------------------------------------------------------------------------
    # Input alignment validation
    # ------------------------------------------------------------------------

    if not sun_tropical_longitude_deg.index.equals(
        moon_tropical_longitude_deg.index
    ):
        raise ValueError(
            "Sun and Moon longitude Series must have identical indexes."
        )

    if not sun_tropical_longitude_deg.index.equals(
        ayanamsha_deg.index
    ):
        raise ValueError(
            "Astronomical longitude and ayanamsha Series must have "
            "identical indexes."
        )

    # ------------------------------------------------------------------------
    # Moon-Sun phase: Tithi, Paksha and Karana
    # ------------------------------------------------------------------------

    phase_angle_deg = wrap_degrees(
        moon_tropical_longitude_deg
        - sun_tropical_longitude_deg
    )

    tithi_number = tithi_number_from_phase(
        phase_angle_deg
    )

    paksha = paksha_from_tithi(
        tithi_number
    )

    karana_slot = karana_slot_from_phase(
        phase_angle_deg
    )

    karana = karana_name_from_slot(
        karana_slot
    )

    # ------------------------------------------------------------------------
    # Nakshatra
    # ------------------------------------------------------------------------

    moon_sidereal_longitude_deg = sidereal_longitude_from_tropical(
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    nakshatra_number = nakshatra_number_from_sidereal_moon(
        moon_sidereal_longitude_deg
    )

    nakshatra = nakshatra_name_from_number(
        nakshatra_number
    )

    # ------------------------------------------------------------------------
    # Yoga
    # ------------------------------------------------------------------------

    yoga_angle_deg = yoga_angle_from_tropical_longitudes(
        sun_tropical_longitude_deg,
        moon_tropical_longitude_deg,
        ayanamsha_deg,
    )

    yoga_number = yoga_number_from_angle(
        yoga_angle_deg
    )

    yoga = yoga_name_from_number(
        yoga_number
    )

    # ------------------------------------------------------------------------
    # Canonical raw F4 table
    # ------------------------------------------------------------------------

    df_f4 = pd.DataFrame(
        {
            "tithi": tithi_number,
            "paksha": paksha,
            "karana": karana,
            "nakshatra": nakshatra,
            "yoga": yoga,
        },
        index=sun_tropical_longitude_deg.index,
    )

    categorical_columns = [
        "tithi",
        "paksha",
        "karana",
        "nakshatra",
        "yoga",
    ]

    for column in categorical_columns:
        df_f4[column] = df_f4[column].astype("category")

    return df_f4

