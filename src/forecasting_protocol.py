"""
Frozen SOLETE forecasting protocol.

This module centralizes the leakage-safe forecasting mechanics defined
before model comparison in Notebook 09:

* canonical 5-minute observational timeline;
* F0 historical/autoregressive feature construction;
* 15-, 30-, and 60-minute future PV targets;
* chronological train/validation/test boundaries;
* split assignment by forecast-valid time;
* model-ready X/y/metadata bundles.

Deterministic Gregorian, solar, lunar, and Panchang calculations remain
in their dedicated feature modules. This module only coordinates their
forecasting-time alignment.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from feature_families import FEATURE_FAMILY_COUNTS

SOLETE_RESOLUTION_MINUTES = 5

TARGET_COLUMN = "P_Solar[kW]"

F0_HISTORY_SOURCES = {
    "P_Solar_kw": "P_Solar[kW]",
    "GHI_kw1m2": "GHI[kW1m2]",
    "POA_Irr_kw1m2": "POA Irr[kW1m2]",
    "temperature_degC": "TEMPERATURE[degC]",
    "humidity_clean": "Humidity_clean",
    "pressure_clean_mbar": "Pressure_clean[mbar]",
    "wind_speed_m1s": "WIND_SPEED[m1s]",
    "wind_direction_sin": "wind_direction_sin",
    "wind_direction_cos": "wind_direction_cos",
}

RECENT_LAGS = tuple(range(13))

DAILY_LAG_STEPS = 288

FORECAST_HORIZONS = {
    15: 3,
    30: 6,
    60: 12,
}

TRAIN_START = pd.Timestamp(
    "2018-06-01 00:00:00",
    tz="UTC",
)
VALID_START = pd.Timestamp(
    "2019-04-01 00:00:00",
    tz="UTC",
)
TEST_START = pd.Timestamp(
    "2019-06-01 00:00:00",
    tz="UTC",
)
TEST_END = pd.Timestamp(
    "2019-09-01 00:00:00",
    tz="UTC",
)

# Historical three-way split API; revised models must use CV_FOLDS.
SPLIT_ORDER = (
    "train",
    "validation",
    "test",
)

MODEL_FEATURE_FAMILIES = (
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
)

F0_RECENT_FEATURE_COLUMNS = tuple(
    f"{feature_name}_lag_{lag}"
    for feature_name in F0_HISTORY_SOURCES
    for lag in RECENT_LAGS
)

F0_DAILY_LAG_COLUMN = "P_Solar_kw_lag_288"

F0_FEATURE_COLUMNS = (
    F0_RECENT_FEATURE_COLUMNS
    + (
        F0_DAILY_LAG_COLUMN,
    )
)

EXPECTED_F0_FEATURES = 118


# Revised Notebook 09: expanding-window folds by FORECAST-VALID time.
# The legacy VALID_START/assign_solete_split API below remains for historical
# compatibility and MUST NOT be used for revised model selection.
CV_FOLDS = (
    {
        "fold": 1,
        "train_start": TRAIN_START,
        "train_end": pd.Timestamp("2018-12-01", tz="UTC"),
        "validation_start": pd.Timestamp("2018-12-01", tz="UTC"),
        "validation_end": pd.Timestamp("2019-02-01", tz="UTC"),
    },
    {
        "fold": 2,
        "train_start": TRAIN_START,
        "train_end": pd.Timestamp("2019-02-01", tz="UTC"),
        "validation_start": pd.Timestamp("2019-02-01", tz="UTC"),
        "validation_end": pd.Timestamp("2019-04-01", tz="UTC"),
    },
    {
        "fold": 3,
        "train_start": TRAIN_START,
        "train_end": pd.Timestamp("2019-04-01", tz="UTC"),
        "validation_start": pd.Timestamp("2019-04-01", tz="UTC"),
        "validation_end": TEST_START,
    },
)

# (train rows, validation rows), folds 1–3, from executed Notebook 09.
EXPECTED_CV_COUNTS_BY_HORIZON = {
    15: ((52413, 17856), (70269, 16992), (87261, 17568)),
    30: ((52410, 17856), (70266, 16992), (87258, 17568)),
    60: ((52404, 17856), (70260, 16992), (87252, 17568)),
}


def _validate_valid_time(valid_time: pd.Series) -> None:
    if not isinstance(valid_time, pd.Series):
        raise TypeError("valid_time must be a pandas Series.")
    if not isinstance(valid_time.dtype, pd.DatetimeTZDtype):
        raise TypeError("valid_time must contain timezone-aware datetimes.")
    if str(valid_time.dt.tz) != "UTC":
        raise ValueError("valid_time must use UTC.")
    if valid_time.isna().any():
        raise ValueError("valid_time must not contain NaT.")


def make_rolling_origin_masks(valid_time: pd.Series) -> tuple[dict, ...]:
    """Return fold-specific train/validation masks by forecast-valid time.

    Each mask keeps the Series index. No scaling or modelling occurs here.
    The June–August 2019 terminal holdout is excluded from all folds.
    """
    _validate_valid_time(valid_time)
    output = []
    for spec in CV_FOLDS:
        train = (
            valid_time.ge(spec["train_start"])
            & valid_time.lt(spec["train_end"])
        ).rename("train_mask")
        validation = (
            valid_time.ge(spec["validation_start"])
            & valid_time.lt(spec["validation_end"])
        ).rename("validation_mask")
        if (train & validation).any():
            raise RuntimeError("Fold train/validation overlap.")
        if spec["train_end"] > spec["validation_start"]:
            raise RuntimeError("Training/validation date ranges overlap.")
        if spec["validation_end"] > TEST_START:
            raise RuntimeError("Fold overlaps terminal holdout.")
        output.append({
            "fold": spec["fold"],
            "train_mask": train,
            "validation_mask": validation,
        })
    return tuple(output)


def terminal_holdout_mask(valid_time: pd.Series) -> pd.Series:
    """Return the locked June–August 2019 test mask by valid time."""
    _validate_valid_time(valid_time)
    return (
        valid_time.ge(TEST_START) & valid_time.lt(TEST_END)
    ).rename("test_mask")


def validate_rolling_origin_counts(
    valid_time: pd.Series,
    horizon_minutes: int,
) -> tuple[tuple[int, int], ...]:
    """Check exact Notebook 09 counts on the FULL eligible horizon sample."""
    if horizon_minutes not in EXPECTED_CV_COUNTS_BY_HORIZON:
        raise ValueError(f"Unsupported horizon: {horizon_minutes}.")
    folds = make_rolling_origin_masks(valid_time)
    counts = tuple(
        (int(f["train_mask"].sum()), int(f["validation_mask"].sum()))
        for f in folds
    )
    expected = EXPECTED_CV_COUNTS_BY_HORIZON[horizon_minutes]
    if counts != expected:
        raise ValueError(
            f"{horizon_minutes}-minute CV counts {counts} "
            f"do not match Notebook 09 {expected}."
        )
    if int(terminal_holdout_mask(valid_time).sum()) != 26496:
        raise ValueError("Terminal holdout must contain 26496 valid times.")
    return counts


def _require_columns(
    frame: pd.DataFrame,
    columns,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}."
        )


def validate_solete_5min_index(
    frame: pd.DataFrame,
) -> None:
    """Validate the complete chronological SOLETE 5-minute grid."""

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "SOLETE rows must use a DatetimeIndex."
        )

    if frame.index.tz is None:
        raise ValueError(
            "SOLETE timestamps must be timezone-aware."
        )

    if str(frame.index.tz) != "UTC":
        raise ValueError(
            "Frozen SOLETE forecasting timestamps must use UTC."
        )

    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "The 5-minute index is not chronologically ordered."
        )

    if frame.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps exist in the 5-minute dataset."
        )

    if len(frame.index) > 1:
        differences = (
            frame.index
            .to_series()
            .diff()
            .dropna()
        )

        expected = pd.Timedelta(
            minutes=SOLETE_RESOLUTION_MINUTES
        )

        if not (
            differences == expected
        ).all():
            raise ValueError(
                "The dataset is not a complete 5-minute grid."
            )


def build_f0_history(
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the frozen 118-column F0 historical feature block.

    F0 contains nine observational variables at issue-time lags
    0 through 12 plus the 288-step (24-hour) PV lag.
    """

    validate_solete_5min_index(
        df_clean
    )

    _require_columns(
        df_clean,
        F0_HISTORY_SOURCES.values(),
    )

    columns = {}

    for (
        feature_name,
        source_column,
    ) in F0_HISTORY_SOURCES.items():
        for lag in RECENT_LAGS:
            column_name = (
                f"{feature_name}_lag_{lag}"
            )

            columns[column_name] = (
                df_clean[
                    source_column
                ]
                .shift(lag)
            )

    columns[
        F0_DAILY_LAG_COLUMN
    ] = (
        df_clean[
            TARGET_COLUMN
        ]
        .shift(
            DAILY_LAG_STEPS
        )
    )

    history = pd.DataFrame(
        columns,
        index=df_clean.index,
    )

    if list(
        history.columns
    ) != list(
        F0_FEATURE_COLUMNS
    ):
        raise RuntimeError(
            "Unexpected F0 feature ordering."
        )

    if history.shape[1] != (
        EXPECTED_F0_FEATURES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_F0_FEATURES} F0 features; "
            f"received {history.shape[1]}."
        )

    return history


def build_forecast_targets(
    df_clean: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Construct the frozen future-PV target table for every horizon.

    Rows remain indexed by forecast issue time ``t``. For each horizon
    ``h`` the target is ``P_Solar(t + h)`` and ``valid_time`` is
    explicitly stored as ``t + h``.
    """

    validate_solete_5min_index(
        df_clean
    )

    _require_columns(
        df_clean,
        [
            TARGET_COLUMN,
        ],
    )

    target_tables = {}

    for (
        horizon_minutes,
        horizon_steps,
    ) in FORECAST_HORIZONS.items():
        target = pd.DataFrame(
            index=df_clean.index
        )

        target[
            "issue_time"
        ] = target.index

        target[
            "valid_time"
        ] = (
            target[
                "issue_time"
            ]
            + pd.Timedelta(
                minutes=horizon_minutes
            )
        )

        target[
            "target_P_Solar_kW"
        ] = (
            df_clean[
                TARGET_COLUMN
            ]
            .shift(
                -horizon_steps
            )
        )

        target[
            "horizon_minutes"
        ] = horizon_minutes

        available = target[
            "target_P_Solar_kW"
        ].notna()

        expected_rows = (
            len(df_clean)
            - horizon_steps
        )

        if int(
            available.sum()
        ) != expected_rows:
            raise ValueError(
                f"{horizon_minutes}-minute target "
                "row count mismatch."
            )

        if expected_rows > 0:
            last_issue = (
                target.index[
                    available
                ][-1]
            )

            last_valid = target.loc[
                last_issue,
                "valid_time",
            ]

            if (
                last_valid
                != df_clean.index[-1]
            ):
                raise RuntimeError(
                    f"{horizon_minutes}-minute final "
                    "target alignment failed."
                )

            expected_target = (
                df_clean.loc[
                    last_valid,
                    TARGET_COLUMN,
                ]
            )

            actual_target = (
                target.loc[
                    last_issue,
                    "target_P_Solar_kW",
                ]
            )

            if actual_target != expected_target:
                raise RuntimeError(
                    f"{horizon_minutes}-minute target "
                    "value alignment failed."
                )

        target_tables[
            horizon_minutes
        ] = target

    return target_tables


# Historical single-validation function, retained for old experiment APIs.
def assign_solete_split(
    valid_time: pd.Series,
) -> pd.Series:
    """
    Assign frozen chronological splits from forecast-valid timestamps.

    The half-open intervals are:

    train
        [2018-06-01, 2019-04-01)

    validation
        [2019-04-01, 2019-06-01)

    test
        [2019-06-01, 2019-09-01)

    Timestamps outside the frozen experiment interval remain missing.
    """

    if not isinstance(
        valid_time,
        pd.Series,
    ):
        raise TypeError(
            "valid_time must be a pandas Series."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        valid_time.dtype
    ):
        raise TypeError(
            "valid_time must contain datetime values."
        )

    labels = pd.Series(
        pd.NA,
        index=valid_time.index,
        dtype="string",
        name="split",
    )

    labels.loc[
        (
            valid_time >= TRAIN_START
        )
        & (
            valid_time < VALID_START
        )
    ] = "train"

    labels.loc[
        (
            valid_time >= VALID_START
        )
        & (
            valid_time < TEST_START
        )
    ] = "validation"

    labels.loc[
        (
            valid_time >= TEST_START
        )
        & (
            valid_time < TEST_END
        )
    ] = "test"

    return labels


def build_base_modelling_tables(
    df_clean: pd.DataFrame,
    *,
    f0_history: pd.DataFrame | None = None,
    target_tables: Mapping[
        int,
        pd.DataFrame,
    ]
    | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Build one leakage-safe base modelling population per horizon.

    Retained rows must have complete F0 history, an available future
    target, and a forecast-valid timestamp inside the frozen experiment
    period. Split assignment uses forecast-valid time, never issue time.
    """

    validate_solete_5min_index(
        df_clean
    )

    if f0_history is None:
        f0_history = build_f0_history(
            df_clean
        )

    if not f0_history.index.equals(
        df_clean.index
    ):
        raise ValueError(
            "F0 history index does not match the cleaned timeline."
        )

    if list(
        f0_history.columns
    ) != list(
        F0_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unexpected F0 feature structure."
        )

    if target_tables is None:
        target_tables = (
            build_forecast_targets(
                df_clean
            )
        )

    if set(
        target_tables
    ) != set(
        FORECAST_HORIZONS
    ):
        raise ValueError(
            "Target tables must contain exactly the frozen "
            "15-, 30-, and 60-minute horizons."
        )

    base_tables = {}

    for horizon_minutes in (
        FORECAST_HORIZONS
    ):
        target = target_tables[
            horizon_minutes
        ]

        if not target.index.equals(
            df_clean.index
        ):
            raise ValueError(
                f"{horizon_minutes}-minute target index "
                "does not match the cleaned timeline."
            )

        _require_columns(
            target,
            [
                "issue_time",
                "valid_time",
                "target_P_Solar_kW",
                "horizon_minutes",
            ],
        )

        model = (
            f0_history
            .copy()
            .join(
                target[
                    [
                        "issue_time",
                        "valid_time",
                        "target_P_Solar_kW",
                        "horizon_minutes",
                    ]
                ]
            )
        )

        usable = (
            model[
                list(
                    F0_FEATURE_COLUMNS
                )
            ]
            .notna()
            .all(
                axis=1
            )
            & model[
                "target_P_Solar_kW"
            ].notna()
            & (
                model[
                    "valid_time"
                ]
                >= TRAIN_START
            )
            & (
                model[
                    "valid_time"
                ]
                < TEST_END
            )
        )

        model = (
            model.loc[
                usable
            ]
            .copy()
        )

        model[
            "split"
        ] = assign_solete_split(
            model[
                "valid_time"
            ]
        )

        if model[
            "split"
        ].isna().any():
            raise RuntimeError(
                f"Unassigned split rows found for "
                f"{horizon_minutes}-minute horizon."
            )

        base_tables[
            horizon_minutes
        ] = model

    return base_tables


def build_modelling_bundle(
    base_table: pd.DataFrame,
    feature_tables: Mapping[
        str,
        pd.DataFrame,
    ],
    solar_feature_table: pd.DataFrame,
) -> dict:
    """
    Construct one frozen model-ready X/y/split bundle.

    No scaling, fitting, feature selection, or learned transformation is
    performed here.
    """

    _require_columns(
        base_table,
        [
            "issue_time",
            "valid_time",
            "horizon_minutes",
            "split",
            "target_P_Solar_kW",
        ],
    )

    missing_families = [
        family
        for family in (
            MODEL_FEATURE_FAMILIES
        )
        if family not in feature_tables
    ]

    if missing_families:
        raise ValueError(
            "Missing feature families: "
            f"{missing_families}."
        )

    _require_columns(
        solar_feature_table,
        [
            "valid_solar_daylight",
        ],
    )

    canonical_index = (
        base_table.index
    )

    if not solar_feature_table.index.equals(
        canonical_index
    ):
        raise ValueError(
            "Solar feature index does not match the modelling index."
        )

    y = (
        base_table[
            "target_P_Solar_kW"
        ]
        .copy()
        .rename(
            "target_P_Solar_kW"
        )
    )

    metadata = (
        base_table[
            [
                "issue_time",
                "valid_time",
                "horizon_minutes",
                "split",
            ]
        ]
        .copy()
    )

    train_mask = metadata[
        "split"
    ].eq(
        "train"
    )

    validation_mask = metadata[
        "split"
    ].eq(
        "validation"
    )

    test_mask = metadata[
        "split"
    ].eq(
        "test"
    )

    daylight_mask = (
        solar_feature_table[
            "valid_solar_daylight"
        ]
        .eq(1)
    )

    x_tables = {}

    for family in (
        MODEL_FEATURE_FAMILIES
    ):
        x = (
            feature_tables[
                family
            ]
            .copy()
        )

        if not x.index.equals(
            canonical_index
        ):
            raise ValueError(
                f"{family}: X/y index alignment failed."
            )

        expected_count = (
            FEATURE_FAMILY_COUNTS[
                family
            ]
        )

        if x.shape[1] != (
            expected_count
        ):
            raise ValueError(
                f"{family}: expected {expected_count} features; "
                f"received {x.shape[1]}."
            )

        if x.isna().any().any():
            raise ValueError(
                f"{family}: missing feature values found."
            )

        x_tables[
            family
        ] = x

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    split_membership_count = (
        train_mask.astype(int)
        + validation_mask.astype(int)
        + test_mask.astype(int)
    )

    if not (
        split_membership_count == 1
    ).all():
        raise ValueError(
            "Every modelling row must belong to exactly one split."
        )

    return {
        "X": x_tables,
        "y": y,
        "metadata": metadata,
        "train_mask": train_mask,
        "validation_mask":
            validation_mask,
        "test_mask": test_mask,
        "daylight_mask":
            daylight_mask,
    }


def build_modelling_bundles(
    base_modelling_tables: Mapping[
        int,
        pd.DataFrame,
    ],
    feature_family_tables: Mapping[
        int,
        Mapping[
            str,
            pd.DataFrame,
        ],
    ],
    solar_feature_tables: Mapping[
        int,
        pd.DataFrame,
    ],
) -> dict[int, dict]:
    """Construct model-ready bundles for all three frozen horizons."""

    bundles = {}

    for horizon_minutes in (
        FORECAST_HORIZONS
    ):
        bundles[
            horizon_minutes
        ] = build_modelling_bundle(
            base_modelling_tables[
                horizon_minutes
            ],
            feature_family_tables[
                horizon_minutes
            ],
            solar_feature_tables[
                horizon_minutes
            ],
        )

    return bundles
