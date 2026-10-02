"""
Frozen PVOD external-validation protocol.

This module extracts the reusable data and experiment mechanics from
Notebook 14 without changing the completed external-validation study.

The protocol preserves:

* ten independent PVOD stations;
* raw timestamps interpreted as UTC;
* the single frozen station00 timestamp correction;
* genuine temporal gaps without interpolation;
* nominal-capacity normalization without clipping values above 1 p.u.;
* station-specific 65/15/20 chronological splits;
* target-valid-time split assignment;
* gap-safe 15/30/60-minute forecasting populations;
* F0 history with 71 predictors;
* target-time deterministic F1/F2/F3/F4 attachment;
* the frozen external model/scaling protocol.

Astronomical and Panchang calculations themselves remain in the
dedicated astronomy and calendar modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pvlib

PVOD_CHINA_TIMEZONE = "Asia/Shanghai"

PVOD_CADENCE = pd.Timedelta(
    minutes=15
)

PVOD_HISTORY_OFFSETS_MIN = (
    60,
    45,
    30,
    15,
    0,
)

PVOD_RECENT_HISTORY_OFFSETS = tuple(
    pd.Timedelta(
        minutes=minutes
    )
    for minutes in PVOD_HISTORY_OFFSETS_MIN
)

PVOD_DAILY_POWER_LAG_MIN = 24 * 60
PVOD_DAILY_LAG = pd.Timedelta(
    hours=24
)

PVOD_HORIZONS = {
    15: pd.Timedelta(
        minutes=15
    ),
    30: pd.Timedelta(
        minutes=30
    ),
    60: pd.Timedelta(
        minutes=60
    ),
}

TRAIN_FRACTION = 0.65
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.20

PVOD_METEOROLOGICAL_COLUMNS = (
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
    "nwp_winddirection",
    "nwp_pressure",
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_winddirection",
    "lmd_windspeed",
)

PVOD_TARGET_COLUMN = "power_MW"
PVOD_NORMALIZED_TARGET_COLUMN = "power_pu"

PVOD_DYNAMIC_SOURCE_COLUMNS = (
    *PVOD_METEOROLOGICAL_COLUMNS,
    PVOD_TARGET_COLUMN,
)

PVOD_EXPECTED_DIMS = {
    "F0": 71,
    "F1": 86,
    "F2": 91,
    "F3": 100,
    "F4": 197,
}

PVOD_ASTRONOMY_REFERENCE = (
    "target_valid_time"
)
PVOD_ASTRONOMY_OFFSET_MIN = 0
PVOD_SIDEREAL_CONVENTION = "Lahiri"
PVOD_EPHEMERIS = "JPL DE440s"

PVOD_PRIMARY_TARGET = "power_pu"
PVOD_SECONDARY_TARGET = "power_MW"

PVOD_TIMESTAMP_CORRECTIONS = {
    "station00": (
        {
            "raw_row_position": 13093,
            "observed_timestamp":
                pd.Timestamp(
                    "2019-01-01 01:15:00",
                    tz="UTC",
                ),
            "corrected_timestamp":
                pd.Timestamp(
                    "2018-12-31 01:15:00",
                    tz="UTC",
                ),
            "expected_power_MW":
                0.715430,
        },
    ),
}


# ---------------------------------------------------------------------
# Frozen external model protocol
# ---------------------------------------------------------------------

PVOD_EXTERNAL_PROTOCOL_NAME = (
    "pvod_external_validation_v1"
)

PVOD_EXTERNAL_MODELS = (
    "Ridge",
    "LightGBM",
    "LSTM",
    "TCN",
)

PVOD_PRIMARY_VARIANTS = (
    "F3",
    "F4_real",
)

PVOD_EXTERNAL_HORIZONS = (
    15,
    30,
    60,
)

PVOD_MODEL_UNIT = (
    "station_specific_refit"
)

PVOD_PREDICTION_CLIPPING = False

PVOD_RIDGE_CONFIG = {
    15: {
        "alpha": 0.01,
    },
    30: {
        "alpha": 0.01,
    },
    60: {
        "alpha": 0.01,
    },
}

PVOD_LIGHTGBM_BASE_CONFIG = {
    "boosting_type":
        "gbdt",
    "colsample_bytree":
        0.8,
    "deterministic":
        True,
    "force_col_wise":
        True,
    "learning_rate":
        0.03,
    "max_depth":
        -1,
    "min_child_samples":
        20,
    "n_estimators":
        300,
    "n_jobs":
        -1,
    "num_leaves":
        15,
    "objective":
        "regression",
    "random_state":
        42,
    "reg_alpha":
        0.0,
    "reg_lambda":
        1.0,
    "subsample":
        0.8,
    "subsample_freq":
        1,
    "verbosity":
        -1,
}

PVOD_LIGHTGBM_CONFIG = {
    horizon:
        PVOD_LIGHTGBM_BASE_CONFIG.copy()
    for horizon in (
        PVOD_EXTERNAL_HORIZONS
    )
}

PVOD_LSTM_ARCHITECTURE = {
    "candidate_id":
        "RNN_C1",
    "units":
        32,
    "n_recurrent_layers":
        1,
    "dropout":
        0.0,
    "fusion_dense_units":
        32,
}

PVOD_LSTM_EPOCHS = {
    15: 7,
    30: 1,
    60: 2,
}

PVOD_TCN_ARCHITECTURE = {
    "candidate_id":
        "TCN_C1",
    "filters":
        32,
    "kernel_size":
        3,
    "dilations":
        (
            1,
            2,
            4,
        ),
    "convolutions_per_block":
        2,
    "dropout":
        0.0,
    "fusion_dense_units":
        32,
    "receptive_field_timesteps":
        29,
}

PVOD_TCN_EPOCHS = {
    15: 4,
    30: 1,
    60: 1,
}

PVOD_NEURAL_SEED = 42
PVOD_NEURAL_BATCH_SIZE = 256
PVOD_NEURAL_SHUFFLE = True
PVOD_NEURAL_CALLBACKS = None
PVOD_NEURAL_VALIDATION_DATA = None

PVOD_SEQUENCE_OFFSETS_MIN = (
    60,
    45,
    30,
    15,
    0,
)

PVOD_SEQUENCE_CHANNELS = (
    PVOD_DYNAMIC_SOURCE_COLUMNS
)

PVOD_SEQUENCE_LENGTH = len(
    PVOD_SEQUENCE_OFFSETS_MIN
)

PVOD_SEQUENCE_CHANNEL_COUNT = len(
    PVOD_SEQUENCE_CHANNELS
)

PVOD_F3_AUXILIARY_DIM = 30
PVOD_F4_AUXILIARY_DIM = 127

PVOD_VALIDATION_POLICY = {
    "hyperparameter_search":
        False,
    "architecture_selection":
        False,
    "epoch_selection":
        False,
    "neural_scaler_fit":
        False,
    "final_refit_population":
        "train_plus_validation",
}

PVOD_EVALUATION_SCOPES = (
    "all",
    "daylight",
)

PVOD_PRIMARY_COMPARISON = {
    "reference":
        "F3",
    "candidate":
        "F4_real",
    "effect":
        "MAE_F4_minus_MAE_F3",
    "lower_is_better":
        True,
}


# ---------------------------------------------------------------------
# Generic validation helpers
# ---------------------------------------------------------------------


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
            f"Missing required columns: {missing}."
        )


def _require_utc_datetime_index(
    index: pd.DatetimeIndex,
) -> None:
    if not isinstance(
        index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected a pandas DatetimeIndex."
        )

    if index.tz is None:
        raise ValueError(
            "PVOD timestamps must be timezone-aware."
        )

    if str(
        index.tz
    ) != "UTC":
        raise ValueError(
            "Frozen PVOD timestamps must use UTC."
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "PVOD timestamps must be chronological."
        )

    if index.has_duplicates:
        raise ValueError(
            "PVOD timestamps must be unique."
        )


# ---------------------------------------------------------------------
# Metadata / station cleaning
# ---------------------------------------------------------------------


def build_pvod_site_registry(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen site-geometry/capacity registry."""

    _require_columns(
        metadata,
        (
            "Station_ID",
            "Latitude",
            "Longitude",
            "Capacity",
        ),
    )

    registry = (
        metadata[
            [
                "Station_ID",
                "Latitude",
                "Longitude",
                "Capacity",
            ]
        ]
        .rename(
            columns={
                "Station_ID":
                    "station",
                "Latitude":
                    "latitude_deg",
                "Longitude":
                    "longitude_deg",
                "Capacity":
                    "capacity_kW",
            }
        )
        .copy()
    )

    registry[
        "capacity_MW"
    ] = (
        pd.to_numeric(
            registry[
                "capacity_kW"
            ],
            errors="raise",
        )
        /
        1000.0
    )

    if registry[
        "station"
    ].duplicated().any():
        raise ValueError(
            "Station metadata contains duplicate station IDs."
        )

    numeric_columns = [
        "latitude_deg",
        "longitude_deg",
        "capacity_kW",
        "capacity_MW",
    ]

    for column in numeric_columns:
        registry[
            column
        ] = pd.to_numeric(
            registry[
                column
            ],
            errors="raise",
        )

    if not np.isfinite(
        registry[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Site registry contains non-finite values."
        )

    if not (
        registry[
            "capacity_MW"
        ]
        > 0.0
    ).all():
        raise ValueError(
            "PVOD capacities must be positive."
        )

    return registry


def clean_pvod_station(
    raw: pd.DataFrame,
    *,
    station_id: str,
    capacity_kw: float,
    corrections: Mapping | None = None,
) -> pd.DataFrame:
    """
    Apply the frozen PVOD station-cleaning policy.

    Measurements are never interpolated or clipped. Genuine temporal
    gaps remain gaps. Power above nominal metadata capacity is retained.
    """

    if not isinstance(
        raw,
        pd.DataFrame,
    ):
        raise TypeError(
            "raw must be a pandas DataFrame."
        )

    _require_columns(
        raw,
        (
            "date_time",
            "power",
        ),
    )

    capacity_kw = float(
        capacity_kw
    )

    if (
        not np.isfinite(
            capacity_kw
        )
        or capacity_kw <= 0.0
    ):
        raise ValueError(
            "capacity_kw must be finite and positive."
        )

    capacity_mw = (
        capacity_kw
        /
        1000.0
    )

    df = raw.copy()

    df[
        "_raw_row_position"
    ] = np.arange(
        len(df),
        dtype=int,
    )

    timestamp_naive = pd.to_datetime(
        df[
            "date_time"
        ],
        errors="raise",
    )

    if getattr(
        timestamp_naive.dt,
        "tz",
        None,
    ) is not None:
        raise ValueError(
            "Raw PVOD date_time values are expected "
            "to be timezone-naive."
        )

    df[
        "timestamp_utc"
    ] = (
        timestamp_naive
        .dt
        .tz_localize(
            "UTC"
        )
    )

    if corrections is None:
        corrections = (
            PVOD_TIMESTAMP_CORRECTIONS
        )

    station_corrections = (
        corrections.get(
            station_id,
            (),
        )
    )

    for correction in (
        station_corrections
    ):
        row_position = int(
            correction[
                "raw_row_position"
            ]
        )

        row_mask = (
            df[
                "_raw_row_position"
            ]
            .eq(
                row_position
            )
        )

        if int(
            row_mask.sum()
        ) != 1:
            raise ValueError(
                f"{station_id}: expected exactly one "
                f"raw row {row_position}."
            )

        actual_timestamp = (
            df.loc[
                row_mask,
                "timestamp_utc",
            ]
            .iloc[
                0
            ]
        )

        expected_timestamp = (
            correction[
                "observed_timestamp"
            ]
        )

        if (
            actual_timestamp
            != expected_timestamp
        ):
            raise ValueError(
                f"{station_id}: timestamp guard failed "
                f"for raw row {row_position}."
            )

        actual_power = float(
            df.loc[
                row_mask,
                "power",
            ]
            .iloc[
                0
            ]
        )

        if not np.isclose(
            actual_power,
            float(
                correction[
                    "expected_power_MW"
                ]
            ),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{station_id}: power guard failed "
                f"for raw row {row_position}."
            )

        df.loc[
            row_mask,
            "timestamp_utc",
        ] = (
            correction[
                "corrected_timestamp"
            ]
        )

    df = (
        df
        .sort_values(
            "timestamp_utc",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    timestamp_index = pd.DatetimeIndex(
        df[
            "timestamp_utc"
        ]
    )

    _require_utc_datetime_index(
        timestamp_index
    )

    if len(
        timestamp_index
    ) > 1:
        differences = (
            df[
                "timestamp_utc"
            ]
            .diff()
            .dropna()
        )

        if not (
            differences
            >= PVOD_CADENCE
        ).all():
            raise ValueError(
                f"{station_id}: sub-15-minute "
                "timestamp step after cleaning."
            )

    df[
        "station_id"
    ] = str(
        station_id
    )

    df[
        "capacity_MW"
    ] = capacity_mw

    df[
        "power_MW"
    ] = pd.to_numeric(
        df[
            "power"
        ],
        errors="raise",
    )

    if not np.isfinite(
        df[
            "power_MW"
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"{station_id}: non-finite power values."
        )

    df[
        "power_pu"
    ] = (
        df[
            "power_MW"
        ]
        /
        capacity_mw
    )

    df[
        "timestamp_china"
    ] = (
        df[
            "timestamp_utc"
        ]
        .dt
        .tz_convert(
            PVOD_CHINA_TIMEZONE
        )
    )

    return df


# ---------------------------------------------------------------------
# Chronological boundaries
# ---------------------------------------------------------------------


def ceil_to_15min(
    timestamp: pd.Timestamp,
) -> pd.Timestamp:
    """Snap a split boundary upward to the next 15-minute point."""

    return timestamp.ceil(
        "15min"
    )


def build_pvod_split_boundaries(
    clean_stations: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """Build frozen station-specific 65/15/20 temporal boundaries."""

    if not clean_stations:
        raise ValueError(
            "clean_stations must not be empty."
        )

    rows = []

    for station_id, frame in (
        clean_stations.items()
    ):
        _require_columns(
            frame,
            (
                "timestamp_utc",
            ),
        )

        timestamps = pd.DatetimeIndex(
            frame[
                "timestamp_utc"
            ]
        )

        _require_utc_datetime_index(
            timestamps
        )

        station_start = (
            timestamps.min()
        )

        station_end = (
            timestamps.max()
        )

        span = (
            station_end
            -
            station_start
            +
            PVOD_CADENCE
        )

        train_end_candidate = (
            station_start
            +
            span
            *
            TRAIN_FRACTION
        )

        validation_end_candidate = (
            station_start
            +
            span
            *
            (
                TRAIN_FRACTION
                +
                VALIDATION_FRACTION
            )
        )

        validation_start = (
            ceil_to_15min(
                train_end_candidate
            )
        )

        test_start = (
            ceil_to_15min(
                validation_end_candidate
            )
        )

        station_stop_exclusive = (
            station_end
            +
            PVOD_CADENCE
        )

        if not (
            station_start
            <
            validation_start
            <
            test_start
            <
            station_stop_exclusive
        ):
            raise ValueError(
                f"{station_id}: invalid frozen split boundaries."
            )

        rows.append(
            {
                "station":
                    station_id,
                "station_start_utc":
                    station_start,
                "validation_start_utc":
                    validation_start,
                "test_start_utc":
                    test_start,
                "station_stop_exclusive_utc":
                    station_stop_exclusive,
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# F0 historical design
# ---------------------------------------------------------------------


def build_pvod_f0(
    station_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the frozen 71-column PVOD F0 predictor table.

    Fourteen dynamic sources are sampled at t-60, t-45, t-30,
    t-15, and t, followed by same-time previous-day PV.
    """

    required = [
        "timestamp_utc",
        *PVOD_DYNAMIC_SOURCE_COLUMNS,
    ]

    _require_columns(
        station_frame,
        required,
    )

    indexed = (
        station_frame
        .copy()
        .set_index(
            "timestamp_utc",
            drop=False,
        )
    )

    _require_utc_datetime_index(
        pd.DatetimeIndex(
            indexed.index
        )
    )

    issue_times = pd.DatetimeIndex(
        indexed.index
    )

    columns = {}

    for source_column in (
        PVOD_DYNAMIC_SOURCE_COLUMNS
    ):
        source = indexed[
            source_column
        ]

        for lag_min in (
            PVOD_HISTORY_OFFSETS_MIN
        ):
            required_times = (
                issue_times
                -
                pd.Timedelta(
                    minutes=lag_min
                )
            )

            feature_name = (
                f"{source_column}"
                f"__lag_{lag_min:04d}m"
            )

            columns[
                feature_name
            ] = (
                source
                .reindex(
                    required_times
                )
                .to_numpy(
                    dtype=np.float64
                )
            )

    daily_times = (
        issue_times
        -
        pd.Timedelta(
            minutes=(
                PVOD_DAILY_POWER_LAG_MIN
            )
        )
    )

    columns[
        "power_MW__lag_1440m"
    ] = (
        indexed[
            "power_MW"
        ]
        .reindex(
            daily_times
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    f0 = pd.DataFrame(
        columns,
        index=issue_times,
    )

    if f0.shape[
        1
    ] != PVOD_EXPECTED_DIMS[
        "F0"
    ]:
        raise RuntimeError(
            "Frozen PVOD F0 dimension mismatch."
        )

    if not f0.columns.is_unique:
        raise RuntimeError(
            "PVOD F0 columns are not unique."
        )

    return f0.astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Exact-time station solar geometry
# ---------------------------------------------------------------------


def build_pvod_solar_features(
    timestamps: pd.DatetimeIndex,
    *,
    latitude_deg: float,
    longitude_deg: float,
) -> pd.DataFrame:
    """
    Build the five frozen site-specific PVOD solar features.

    Unlike SOLETE, PVOD uses the exact supplied timestamp with no
    +2-minute offset.
    """

    timestamps = pd.DatetimeIndex(
        timestamps
    )

    _require_utc_datetime_index(
        timestamps
    )

    solar_position = (
        pvlib.solarposition
        .get_solarposition(
            time=timestamps,
            latitude=float(
                latitude_deg
            ),
            longitude=float(
                longitude_deg
            ),
            method="nrel_numpy",
        )
    )

    elevation = (
        solar_position[
            "elevation"
        ]
        .to_numpy(
            dtype=float
        )
    )

    zenith = (
        solar_position[
            "zenith"
        ]
        .to_numpy(
            dtype=float
        )
    )

    azimuth_deg = (
        solar_position[
            "azimuth"
        ]
        .to_numpy(
            dtype=float
        )
    )

    azimuth_rad = np.deg2rad(
        azimuth_deg
    )

    features = pd.DataFrame(
        {
            "solar_elevation_deg":
                elevation,
            "solar_cos_zenith":
                np.cos(
                    np.deg2rad(
                        zenith
                    )
                ),
            "solar_azimuth_sin":
                np.sin(
                    azimuth_rad
                ),
            "solar_azimuth_cos":
                np.cos(
                    azimuth_rad
                ),
            "solar_daylight":
                (
                    elevation
                    > 0.0
                ).astype(
                    np.int8
                ),
        },
        index=timestamps,
    )

    if features.shape != (
        len(
            timestamps
        ),
        5,
    ):
        raise RuntimeError(
            "PVOD F2 solar dimension mismatch."
        )

    if not np.isfinite(
        features.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "PVOD solar features contain non-finite values."
        )

    return features


# ---------------------------------------------------------------------
# Gap-safe sample population
# ---------------------------------------------------------------------


def build_pvod_sample_indices(
    station_frame: pd.DataFrame,
    f0: pd.DataFrame,
    split_boundary: Mapping,
    solar_features: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Build gap-safe horizon-specific PVOD forecasting populations.

    All F0 predictors must exist. Every future 15-minute timestamp from
    issue time through target time must physically exist. Split labels
    are determined only from target-valid time.
    """

    _require_columns(
        station_frame,
        (
            "timestamp_utc",
            "power_MW",
            "power_pu",
        ),
    )

    _require_columns(
        solar_features,
        (
            "solar_daylight",
        ),
    )

    indexed = (
        station_frame
        .copy()
        .set_index(
            "timestamp_utc",
            drop=False,
        )
    )

    timestamp_index = pd.DatetimeIndex(
        indexed.index
    )

    _require_utc_datetime_index(
        timestamp_index
    )

    if not f0.index.equals(
        timestamp_index
    ):
        raise ValueError(
            "F0 index does not match the station timeline."
        )

    if not solar_features.index.equals(
        timestamp_index
    ):
        raise ValueError(
            "Solar-feature index does not match "
            "the station timeline."
        )

    required_boundary_keys = (
        "validation_start_utc",
        "test_start_utc",
        "station_stop_exclusive_utc",
    )

    missing_boundary_keys = [
        key
        for key in required_boundary_keys
        if key not in split_boundary
    ]

    if missing_boundary_keys:
        raise ValueError(
            "Missing split-boundary keys: "
            f"{missing_boundary_keys}."
        )

    validation_start = (
        split_boundary[
            "validation_start_utc"
        ]
    )

    test_start = (
        split_boundary[
            "test_start_utc"
        ]
    )

    stop_exclusive = (
        split_boundary[
            "station_stop_exclusive_utc"
        ]
    )

    timestamp_set = set(
        timestamp_index
    )

    complete_f0_times = (
        f0.index[
            f0.notna().all(
                axis=1
            )
        ]
    )

    result = {}

    for horizon_min in sorted(
        PVOD_HORIZONS
    ):
        horizon_delta = (
            PVOD_HORIZONS[
                horizon_min
            ]
        )

        horizon_steps = (
            horizon_min
            //
            15
        )

        records = []

        for issue_time in (
            complete_f0_times
        ):
            future_times = [
                issue_time
                +
                pd.Timedelta(
                    minutes=(
                        15
                        *
                        step
                    )
                )
                for step in range(
                    1,
                    horizon_steps + 1,
                )
            ]

            if not all(
                future_time
                in timestamp_set
                for future_time
                in future_times
            ):
                continue

            target_time = (
                issue_time
                +
                horizon_delta
            )

            if (
                target_time
                <
                validation_start
            ):
                split_name = "train"

            elif (
                target_time
                <
                test_start
            ):
                split_name = (
                    "validation"
                )

            elif (
                target_time
                <
                stop_exclusive
            ):
                split_name = "test"

            else:
                continue

            target_row = (
                indexed.loc[
                    target_time
                ]
            )

            target_power_mw = float(
                target_row[
                    "power_MW"
                ]
            )

            target_power_pu = float(
                target_row[
                    "power_pu"
                ]
            )

            target_daylight = int(
                solar_features.loc[
                    target_time,
                    "solar_daylight",
                ]
            )

            records.append(
                {
                    "issue_time_utc":
                        issue_time,
                    "target_time_utc":
                        target_time,
                    "split":
                        split_name,
                    "target_power_MW":
                        target_power_mw,
                    "target_power_pu":
                        target_power_pu,
                    "target_solar_daylight":
                        target_daylight,
                }
            )

        sample_table = pd.DataFrame(
            records
        )

        if sample_table.empty:
            raise ValueError(
                f"No eligible samples for {horizon_min}-minute horizon."
            )

        if not sample_table[
            "issue_time_utc"
        ].is_unique:
            raise RuntimeError(
                "Duplicate issue times in PVOD sample index."
            )

        if not sample_table[
            "target_time_utc"
        ].is_unique:
            raise RuntimeError(
                "Duplicate target times in PVOD sample index."
            )

        if not sample_table[
            "issue_time_utc"
        ].is_monotonic_increasing:
            raise RuntimeError(
                "PVOD sample index is not chronological."
            )

        timing = (
            sample_table[
                "target_time_utc"
            ]
            -
            sample_table[
                "issue_time_utc"
            ]
        )

        if not timing.eq(
            horizon_delta
        ).all():
            raise RuntimeError(
                "PVOD target timing mismatch."
            )

        numeric_target = (
            sample_table[
                [
                    "target_power_MW",
                    "target_power_pu",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_target
        ).all():
            raise ValueError(
                "PVOD sample targets contain "
                "non-finite values."
            )

        if (
            sample_table[
                "target_power_pu"
            ].min()
            <
            0.0
        ):
            raise ValueError(
                "PVOD normalized target contains "
                "negative values."
            )

        result[
            horizon_min
        ] = sample_table

    return result


# ---------------------------------------------------------------------
# Lazy F0-F4 design materialization
# ---------------------------------------------------------------------


def materialize_pvod_design(
    sample_table: pd.DataFrame,
    *,
    f0: pd.DataFrame,
    f1: pd.DataFrame,
    f2: pd.DataFrame,
    f3: pd.DataFrame,
    f4: pd.DataFrame,
    feature_block: str,
    split: str | None = None,
) -> dict:
    """
    Materialize one frozen PVOD design matrix.

    F0 is issue-time history. F1-F4 are selected at forecast-valid time
    and then reindexed to issue time for model fitting.
    """

    if feature_block not in (
        PVOD_EXPECTED_DIMS
    ):
        raise KeyError(
            f"Unknown feature block: {feature_block}"
        )

    _require_columns(
        sample_table,
        (
            "issue_time_utc",
            "target_time_utc",
            "split",
            "target_power_pu",
            "target_power_MW",
            "target_solar_daylight",
        ),
    )

    samples = (
        sample_table.copy()
    )

    if split is not None:
        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unknown split: {split}"
            )

        samples = (
            samples.loc[
                samples[
                    "split"
                ].eq(
                    split
                )
            ]
            .copy()
        )

    samples = samples.reset_index(
        drop=True
    )

    if samples.empty:
        raise ValueError(
            "No rows remain after split filtering."
        )

    issue_times = pd.DatetimeIndex(
        samples[
            "issue_time_utc"
        ]
    )

    target_times = pd.DatetimeIndex(
        samples[
            "target_time_utc"
        ]
    )

    blocks = [
        f0.loc[
            issue_times
        ].copy()
    ]

    blocks[
        0
    ].index = issue_times

    if feature_block in {
        "F1",
        "F2",
        "F3",
        "F4",
    }:
        block = f1.loc[
            target_times
        ].copy()
        block.index = (
            issue_times
        )
        blocks.append(
            block
        )

    if feature_block in {
        "F2",
        "F3",
        "F4",
    }:
        block = f2.loc[
            target_times
        ].copy()
        block.index = (
            issue_times
        )
        blocks.append(
            block
        )

    if feature_block in {
        "F3",
        "F4",
    }:
        block = f3.loc[
            target_times
        ].copy()
        block.index = (
            issue_times
        )
        blocks.append(
            block
        )

    if feature_block == "F4":
        block = f4.loc[
            target_times
        ].copy()
        block.index = (
            issue_times
        )
        blocks.append(
            block
        )

    X = pd.concat(
        blocks,
        axis=1,
    )

    expected_dim = (
        PVOD_EXPECTED_DIMS[
            feature_block
        ]
    )

    if X.shape != (
        len(
            samples
        ),
        expected_dim,
    ):
        raise ValueError(
            f"{feature_block}: expected design shape "
            f"({len(samples)}, {expected_dim}), "
            f"found {X.shape}."
        )

    if not X.columns.is_unique:
        raise ValueError(
            "PVOD design contains duplicate columns."
        )

    if not X.index.equals(
        issue_times
    ):
        raise ValueError(
            "PVOD design index alignment failed."
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "PVOD design contains non-finite values."
        )

    X = X.astype(
        np.float32
    )

    y_pu = pd.Series(
        samples[
            "target_power_pu"
        ].to_numpy(
            dtype=np.float32
        ),
        index=issue_times,
        name="target_power_pu",
    )

    y_mw = pd.Series(
        samples[
            "target_power_MW"
        ].to_numpy(
            dtype=np.float32
        ),
        index=issue_times,
        name="target_power_MW",
    )

    meta = (
        samples
        .set_index(
            "issue_time_utc",
            drop=False,
        )
        .copy()
    )

    return {
        "X":
            X,
        "y_pu":
            y_pu,
        "y_MW":
            y_mw,
        "meta":
            meta,
    }


# ---------------------------------------------------------------------
# Neural auxiliary classification
# ---------------------------------------------------------------------


def is_pvod_binary_auxiliary_column(
    column_name: str,
) -> bool:
    """Return whether the frozen neural protocol leaves a column 0/1."""

    column_name = str(
        column_name
    )

    if column_name in {
        "utc_is_weekend",
        "valid_utc_is_weekend",
        "solar_daylight",
    }:
        return True

    panchang_prefixes = (
        "tithi_",
        "paksha_",
        "karana_",
        "nakshatra_",
        "yoga_",
    )

    return column_name.startswith(
        panchang_prefixes
    )
