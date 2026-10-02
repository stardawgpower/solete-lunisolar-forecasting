"""
Canonical feature-family definitions for the SOLETE forecasting study.

This module centralizes the frozen F0-F4 feature dimensions and the
deterministic Panchang categorical vocabulary used by the completed
experiments.

Scientific hierarchy
--------------------
F0
    Leakage-safe historical meteorological and photovoltaic features.

F1
    F0 + Gregorian/calendar features.

F2
    F1 + deterministic solar-geometry features.

F3
    F2 + continuous lunar/Sun-Moon astronomical features.

F4
    F3 + 97 categorical Panchang indicators.

The categorical vocabularies are fixed a priori and are never inferred
from the supplied modelling population.
"""

from __future__ import annotations

import pandas as pd

import vedic_calendar

RAW_PANCHANG_COLUMNS = (
    "valid_tithi",
    "valid_paksha",
    "valid_karana",
    "valid_nakshatra",
    "valid_yoga",
)

TITHI_CATEGORIES = tuple(range(1, 31))

PAKSHA_CATEGORIES = (
    "Shukla",
    "Krishna",
)

KARANA_CATEGORIES = tuple(
    dict.fromkeys(vedic_calendar.KARANA_SEQUENCE)
)

NAKSHATRA_CATEGORIES = tuple(
    vedic_calendar.NAKSHATRA_NAMES
)

YOGA_CATEGORIES = tuple(
    vedic_calendar.YOGA_NAMES
)

PANCHANG_VOCABULARY = {
    "valid_tithi": TITHI_CATEGORIES,
    "valid_paksha": PAKSHA_CATEGORIES,
    "valid_karana": KARANA_CATEGORIES,
    "valid_nakshatra": NAKSHATRA_CATEGORIES,
    "valid_yoga": YOGA_CATEGORIES,
}

PANCHANG_GROUP_PREFIXES = {
    "tithi": "valid_tithi_",
    "paksha": "valid_paksha_",
    "karana": "valid_karana_",
    "nakshatra": "valid_nakshatra_",
    "yoga": "valid_yoga_",
}

PANCHANG_GROUP_SIZES = {
    "tithi": 30,
    "paksha": 2,
    "karana": 11,
    "nakshatra": 27,
    "yoga": 27,
}

PANCHANG_INDICATOR_COUNT = sum(
    PANCHANG_GROUP_SIZES.values()
)

FEATURE_FAMILY_COUNTS = {
    "F0": 118,
    "F1": 133,
    "F2": 138,
    "F3": 147,
    "F4": 244,
}


def encode_panchang_fixed_vocabulary(
    df_panchang: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode the five canonical Panchang variables into 97 indicators.

    Category vocabularies are fixed in advance. Therefore the returned
    table has the same dimensionality and column ordering even when a
    particular input sample does not contain every possible state.

    Parameters
    ----------
    df_panchang
        DataFrame containing exactly the five forecast-valid Panchang
        columns listed in ``RAW_PANCHANG_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        Deterministic int8 one-hot table with exactly 97 columns.
    """

    if list(df_panchang.columns) != list(
        RAW_PANCHANG_COLUMNS
    ):
        raise ValueError(
            "Unexpected raw Panchang column structure. "
            f"Expected {list(RAW_PANCHANG_COLUMNS)}, "
            f"received {list(df_panchang.columns)}."
        )

    categorical = pd.DataFrame(
        index=df_panchang.index
    )

    for column, categories in PANCHANG_VOCABULARY.items():
        observed = set(
            df_panchang[column].unique()
        )
        allowed = set(categories)

        unexpected = observed - allowed

        if unexpected:
            raise ValueError(
                f"Unexpected categories in {column}: "
                f"{sorted(unexpected)}"
            )

        categorical[column] = pd.Categorical(
            df_panchang[column],
            categories=categories,
            ordered=False,
        )

    encoded = pd.get_dummies(
        categorical,
        prefix={
            "valid_tithi": "valid_tithi",
            "valid_paksha": "valid_paksha",
            "valid_karana": "valid_karana",
            "valid_nakshatra": "valid_nakshatra",
            "valid_yoga": "valid_yoga",
        },
        prefix_sep="_",
        dtype="int8",
    )

    if encoded.shape[1] != PANCHANG_INDICATOR_COUNT:
        raise RuntimeError(
            "Unexpected Panchang indicator count: "
            f"{encoded.shape[1]}."
        )

    if encoded.isna().any().any():
        raise RuntimeError(
            "Encoded Panchang table contains missing values."
        )

    return encoded


def panchang_group_columns(
    encoded_columns,
) -> dict[str, tuple[str, ...]]:
    """
    Group canonical F4 one-hot columns by Panchang family.
    """

    columns = tuple(encoded_columns)

    groups = {
        family: tuple(
            column
            for column in columns
            if column.startswith(prefix)
        )
        for family, prefix in PANCHANG_GROUP_PREFIXES.items()
    }

    for family, expected_size in PANCHANG_GROUP_SIZES.items():
        actual_size = len(groups[family])

        if actual_size != expected_size:
            raise ValueError(
                f"{family} group has {actual_size} columns; "
                f"expected {expected_size}."
            )

    return groups
