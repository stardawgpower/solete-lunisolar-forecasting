import numpy as np
import pandas as pd
import pytest

from pvod import (
    PVOD_ASTRONOMY_OFFSET_MIN,
    PVOD_CADENCE,
    PVOD_CHINA_TIMEZONE,
    PVOD_DAILY_POWER_LAG_MIN,
    PVOD_DYNAMIC_SOURCE_COLUMNS,
    PVOD_EXPECTED_DIMS,
    PVOD_EXTERNAL_MODELS,
    PVOD_F3_AUXILIARY_DIM,
    PVOD_F4_AUXILIARY_DIM,
    PVOD_HISTORY_OFFSETS_MIN,
    PVOD_LIGHTGBM_BASE_CONFIG,
    PVOD_LSTM_ARCHITECTURE,
    PVOD_LSTM_EPOCHS,
    PVOD_METEOROLOGICAL_COLUMNS,
    PVOD_MODEL_UNIT,
    PVOD_PREDICTION_CLIPPING,
    PVOD_RIDGE_CONFIG,
    PVOD_TCN_ARCHITECTURE,
    PVOD_TCN_EPOCHS,
    PVOD_TIMESTAMP_CORRECTIONS,
    PVOD_VALIDATION_POLICY,
    build_pvod_f0,
    build_pvod_sample_indices,
    build_pvod_site_registry,
    build_pvod_solar_features,
    build_pvod_split_boundaries,
    ceil_to_15min,
    clean_pvod_station,
    is_pvod_binary_auxiliary_column,
    materialize_pvod_design,
)


def make_raw_station(
    *,
    start="2020-01-01 00:00:00",
    periods=500,
    capacity_mw=1.0,
):
    timestamps = pd.date_range(
        start,
        periods=periods,
        freq="15min",
    )

    data = {
        "date_time":
            timestamps.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        "power":
            np.linspace(
                0.0,
                0.9 * capacity_mw,
                periods,
            ),
    }

    for offset, column in enumerate(
        PVOD_METEOROLOGICAL_COLUMNS
    ):
        data[
            column
        ] = (
            np.arange(
                periods,
                dtype=float,
            )
            + float(
                offset + 1
            )
        )

    return pd.DataFrame(
        data
    )


def clean_synthetic(
    *,
    periods=500,
):
    raw = make_raw_station(
        periods=periods
    )

    return clean_pvod_station(
        raw,
        station_id="station99",
        capacity_kw=1000.0,
    )


def test_protocol_constants_and_dimensions():
    assert PVOD_CADENCE == pd.Timedelta(
        minutes=15
    )

    assert PVOD_HISTORY_OFFSETS_MIN == (
        60,
        45,
        30,
        15,
        0,
    )

    assert PVOD_DAILY_POWER_LAG_MIN == 1440

    assert len(
        PVOD_METEOROLOGICAL_COLUMNS
    ) == 13

    assert len(
        PVOD_DYNAMIC_SOURCE_COLUMNS
    ) == 14

    assert PVOD_EXPECTED_DIMS == {
        "F0": 71,
        "F1": 86,
        "F2": 91,
        "F3": 100,
        "F4": 197,
    }

    assert PVOD_ASTRONOMY_OFFSET_MIN == 0

    assert PVOD_CHINA_TIMEZONE == (
        "Asia/Shanghai"
    )


def test_timestamp_correction_registry_is_frozen():
    correction = (
        PVOD_TIMESTAMP_CORRECTIONS[
            "station00"
        ][
            0
        ]
    )

    assert correction[
        "raw_row_position"
    ] == 13093

    assert correction[
        "observed_timestamp"
    ] == pd.Timestamp(
        "2019-01-01 01:15:00",
        tz="UTC",
    )

    assert correction[
        "corrected_timestamp"
    ] == pd.Timestamp(
        "2018-12-31 01:15:00",
        tz="UTC",
    )

    assert correction[
        "expected_power_MW"
    ] == pytest.approx(
        0.715430
    )


def test_ceil_to_15min():
    value = pd.Timestamp(
        "2020-01-01 00:07:00",
        tz="UTC",
    )

    assert ceil_to_15min(
        value
    ) == pd.Timestamp(
        "2020-01-01 00:15:00",
        tz="UTC",
    )

    already_aligned = pd.Timestamp(
        "2020-01-01 00:30:00",
        tz="UTC",
    )

    assert ceil_to_15min(
        already_aligned
    ) == already_aligned


def test_site_registry_converts_capacity():
    metadata = pd.DataFrame(
        {
            "Station_ID":
                [
                    "station00",
                    "station01",
                ],
            "Latitude":
                [
                    30.0,
                    31.0,
                ],
            "Longitude":
                [
                    120.0,
                    121.0,
                ],
            "Capacity":
                [
                    1000.0,
                    2500.0,
                ],
        }
    )

    registry = (
        build_pvod_site_registry(
            metadata
        )
    )

    assert registry[
        "capacity_MW"
    ].tolist() == [
        1.0,
        2.5,
    ]


def test_clean_station_preserves_gaps_and_above_capacity():
    raw = make_raw_station(
        periods=120
    )

    raw.loc[
        30,
        "power",
    ] = 1.2

    raw = raw.drop(
        index=40
    ).reset_index(
        drop=True
    )

    cleaned = clean_pvod_station(
        raw,
        station_id="station99",
        capacity_kw=1000.0,
    )

    assert len(
        cleaned
    ) == 119

    assert cleaned[
        "timestamp_utc"
    ].is_unique

    assert cleaned[
        "power_pu"
    ].max() == pytest.approx(
        1.2
    )

    assert (
        cleaned[
            "timestamp_utc"
        ]
        .diff()
        .dropna()
        .max()
        ==
        pd.Timedelta(
            minutes=30
        )
    )


def test_clean_station_applies_guarded_custom_correction():
    raw = make_raw_station(
        periods=3
    )

    raw[
        "date_time"
    ] = [
        "2020-01-01 00:00:00",
        "2020-01-02 00:15:00",
        "2020-01-01 00:30:00",
    ]

    raw.loc[
        1,
        "power",
    ] = 0.5

    corrections = {
        "stationX": (
            {
                "raw_row_position":
                    1,
                "observed_timestamp":
                    pd.Timestamp(
                        "2020-01-02 00:15:00",
                        tz="UTC",
                    ),
                "corrected_timestamp":
                    pd.Timestamp(
                        "2020-01-01 00:15:00",
                        tz="UTC",
                    ),
                "expected_power_MW":
                    0.5,
            },
        )
    }

    cleaned = clean_pvod_station(
        raw,
        station_id="stationX",
        capacity_kw=1000.0,
        corrections=corrections,
    )

    assert cleaned[
        "timestamp_utc"
    ].tolist() == list(
        pd.date_range(
            "2020-01-01 00:00:00",
            periods=3,
            freq="15min",
            tz="UTC",
        )
    )


def test_split_boundaries_are_half_open_and_snapped():
    clean = clean_synthetic(
        periods=400
    )

    boundaries = (
        build_pvod_split_boundaries(
            {
                "station99":
                    clean,
            }
        )
    )

    row = boundaries.iloc[
        0
    ]

    start = pd.Timestamp(
        "2020-01-01 00:00:00",
        tz="UTC",
    )

    assert row[
        "station_start_utc"
    ] == start

    assert row[
        "validation_start_utc"
    ] == (
        start
        + pd.Timedelta(
            hours=65
        )
    )

    assert row[
        "test_start_utc"
    ] == (
        start
        + pd.Timedelta(
            hours=80
        )
    )

    assert row[
        "station_stop_exclusive_utc"
    ] == (
        start
        + pd.Timedelta(
            hours=100
        )
    )


def test_f0_has_71_columns_and_exact_lags():
    clean = clean_synthetic(
        periods=200
    )

    f0 = build_pvod_f0(
        clean
    )

    assert f0.shape == (
        200,
        71,
    )

    issue_position = 100

    assert f0[
        "power_MW__lag_0000m"
    ].iloc[
        issue_position
    ] == pytest.approx(
        clean[
            "power_MW"
        ].iloc[
            issue_position
        ]
    )

    assert f0[
        "power_MW__lag_0015m"
    ].iloc[
        issue_position
    ] == pytest.approx(
        clean[
            "power_MW"
        ].iloc[
            issue_position - 1
        ]
    )

    assert f0[
        "power_MW__lag_1440m"
    ].iloc[
        96
    ] == pytest.approx(
        clean[
            "power_MW"
        ].iloc[
            0
        ]
    )


def test_f0_preserves_gap_missingness():
    raw = make_raw_station(
        periods=200
    )

    missing_time = pd.Timestamp(
        raw.loc[
            120,
            "date_time",
        ]
    )

    raw = raw.drop(
        index=120
    ).reset_index(
        drop=True
    )

    clean = clean_pvod_station(
        raw,
        station_id="station99",
        capacity_kw=1000.0,
    )

    f0 = build_pvod_f0(
        clean
    )

    issue_time = (
        missing_time
        .tz_localize(
            "UTC"
        )
        + pd.Timedelta(
            minutes=15
        )
    )

    assert np.isnan(
        f0.loc[
            issue_time,
            "power_MW__lag_0015m",
        ]
    )


def test_sample_indices_are_gap_safe_and_valid_time_split():
    raw = make_raw_station(
        periods=500
    )

    missing_raw_index = 350

    missing_time = pd.Timestamp(
        raw.loc[
            missing_raw_index,
            "date_time",
        ]
    ).tz_localize(
        "UTC"
    )

    raw = raw.drop(
        index=missing_raw_index
    ).reset_index(
        drop=True
    )

    clean = clean_pvod_station(
        raw,
        station_id="station99",
        capacity_kw=1000.0,
    )

    f0 = build_pvod_f0(
        clean
    )

    boundaries = (
        build_pvod_split_boundaries(
            {
                "station99":
                    clean,
            }
        )
        .set_index(
            "station"
        )
        .loc[
            "station99"
        ]
    )

    solar = pd.DataFrame(
        {
            "solar_daylight":
                np.ones(
                    len(
                        clean
                    ),
                    dtype=np.int8,
                ),
        },
        index=pd.DatetimeIndex(
            clean[
                "timestamp_utc"
            ]
        ),
    )

    indices = (
        build_pvod_sample_indices(
            clean,
            f0,
            boundaries,
            solar,
        )
    )

    for horizon_min, samples in (
        indices.items()
    ):
        assert set(
            samples[
                "split"
            ].unique()
        ) == {
            "train",
            "validation",
            "test",
        }

        assert (
            samples[
                "target_time_utc"
            ]
            -
            samples[
                "issue_time_utc"
            ]
        ).eq(
            pd.Timedelta(
                minutes=horizon_min
            )
        ).all()

        timestamp_set = set(
            clean[
                "timestamp_utc"
            ]
        )

        for issue_time in samples[
            "issue_time_utc"
        ]:
            for step in range(
                1,
                horizon_min // 15 + 1,
            ):
                assert (
                    issue_time
                    + pd.Timedelta(
                        minutes=15 * step
                    )
                    in timestamp_set
                )

    assert missing_time not in set(
        indices[
            15
        ][
            "target_time_utc"
        ]
    )


def test_sample_indices_preserve_above_one_pu_targets():
    raw = make_raw_station(
        periods=500
    )

    raw.loc[
        450,
        "power",
    ] = 1.25

    clean = clean_pvod_station(
        raw,
        station_id="station99",
        capacity_kw=1000.0,
    )

    f0 = build_pvod_f0(
        clean
    )

    boundary = (
        build_pvod_split_boundaries(
            {
                "station99":
                    clean,
            }
        )
        .set_index(
            "station"
        )
        .loc[
            "station99"
        ]
    )

    solar = pd.DataFrame(
        {
            "solar_daylight":
                np.ones(
                    len(
                        clean
                    ),
                    dtype=np.int8,
                ),
        },
        index=pd.DatetimeIndex(
            clean[
                "timestamp_utc"
            ]
        ),
    )

    indices = (
        build_pvod_sample_indices(
            clean,
            f0,
            boundary,
            solar,
        )
    )

    assert (
        indices[
            15
        ][
            "target_power_pu"
        ].max()
        >= 1.25
    )


def test_solar_features_are_site_specific_and_finite():
    timestamps = pd.date_range(
        "2020-06-01 00:00:00",
        periods=8,
        freq="15min",
        tz="UTC",
    )

    solar = build_pvod_solar_features(
        timestamps,
        latitude_deg=30.0,
        longitude_deg=120.0,
    )

    assert solar.shape == (
        8,
        5,
    )

    assert np.isfinite(
        solar.to_numpy(
            dtype=float
        )
    ).all()

    assert set(
        solar[
            "solar_daylight"
        ].unique()
    ).issubset(
        {
            0,
            1,
        }
    )


def make_design_fixture():
    clean = clean_synthetic(
        periods=500
    )

    f0 = build_pvod_f0(
        clean
    )

    boundary = (
        build_pvod_split_boundaries(
            {
                "station99":
                    clean,
            }
        )
        .set_index(
            "station"
        )
        .loc[
            "station99"
        ]
    )

    timeline = pd.DatetimeIndex(
        clean[
            "timestamp_utc"
        ]
    )

    solar = pd.DataFrame(
        {
            "solar_daylight":
                np.ones(
                    len(
                        timeline
                    ),
                    dtype=np.int8,
                ),
        },
        index=timeline,
    )

    sample_table = (
        build_pvod_sample_indices(
            clean,
            f0,
            boundary,
            solar,
        )[
            15
        ]
    )

    f1 = pd.DataFrame(
        {
            f"f1_{i}":
                (
                    np.arange(
                        len(
                            timeline
                        ),
                        dtype=float,
                    )
                    if i == 0
                    else np.full(
                        len(
                            timeline
                        ),
                        i,
                        dtype=float,
                    )
                )
            for i in range(
                15
            )
        },
        index=timeline,
    )

    f2 = pd.DataFrame(
        {
            f"f2_{i}":
                np.full(
                    len(
                        timeline
                    ),
                    i + 1,
                    dtype=float,
                )
            for i in range(
                5
            )
        },
        index=timeline,
    )

    f3 = pd.DataFrame(
        {
            f"f3_{i}":
                np.full(
                    len(
                        timeline
                    ),
                    i + 1,
                    dtype=float,
                )
            for i in range(
                9
            )
        },
        index=timeline,
    )

    f4 = pd.DataFrame(
        {
            f"f4_{i}":
                np.full(
                    len(
                        timeline
                    ),
                    i % 2,
                    dtype=np.int8,
                )
            for i in range(
                97
            )
        },
        index=timeline,
    )

    return (
        sample_table,
        f0,
        f1,
        f2,
        f3,
        f4,
        timeline,
    )


def test_design_materializer_has_frozen_dimensions_and_target_alignment():
    (
        sample_table,
        f0,
        f1,
        f2,
        f3,
        f4,
        timeline,
    ) = make_design_fixture()

    for family, expected_dim in (
        PVOD_EXPECTED_DIMS.items()
    ):
        design = (
            materialize_pvod_design(
                sample_table,
                f0=f0,
                f1=f1,
                f2=f2,
                f3=f3,
                f4=f4,
                feature_block=family,
                split="test",
            )
        )

        assert design[
            "X"
        ].shape[
            1
        ] == expected_dim

        assert design[
            "X"
        ].index.equals(
            design[
                "y_pu"
            ].index
        )

    design_f1 = (
        materialize_pvod_design(
            sample_table,
            f0=f0,
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            feature_block="F1",
            split="test",
        )
    )

    first_meta = (
        design_f1[
            "meta"
        ].iloc[
            0
        ]
    )

    target_time = first_meta[
        "target_time_utc"
    ]

    target_position = (
        timeline.get_loc(
            target_time
        )
    )

    assert design_f1[
        "X"
    ][
        "f1_0"
    ].iloc[
        0
    ] == pytest.approx(
        target_position
    )


def test_design_materializer_rejects_invalid_requests():
    (
        sample_table,
        f0,
        f1,
        f2,
        f3,
        f4,
        _,
    ) = make_design_fixture()

    with pytest.raises(
        KeyError,
        match="Unknown feature block",
    ):
        materialize_pvod_design(
            sample_table,
            f0=f0,
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            feature_block="F9",
        )

    with pytest.raises(
        ValueError,
        match="Unknown split",
    ):
        materialize_pvod_design(
            sample_table,
            f0=f0,
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            feature_block="F3",
            split="future",
        )


def test_external_model_protocol_is_frozen_and_binary_detection():
    assert PVOD_EXTERNAL_MODELS == (
        "Ridge",
        "LightGBM",
        "LSTM",
        "TCN",
    )

    assert PVOD_MODEL_UNIT == (
        "station_specific_refit"
    )

    assert PVOD_PREDICTION_CLIPPING is False

    assert {
        horizon:
            config[
                "alpha"
            ]
        for horizon, config
        in PVOD_RIDGE_CONFIG.items()
    } == {
        15: 0.01,
        30: 0.01,
        60: 0.01,
    }

    assert PVOD_LIGHTGBM_BASE_CONFIG[
        "learning_rate"
    ] == pytest.approx(
        0.03
    )

    assert PVOD_LIGHTGBM_BASE_CONFIG[
        "n_estimators"
    ] == 300

    assert PVOD_LIGHTGBM_BASE_CONFIG[
        "num_leaves"
    ] == 15

    assert PVOD_LSTM_ARCHITECTURE[
        "candidate_id"
    ] == "RNN_C1"

    assert PVOD_LSTM_EPOCHS == {
        15: 7,
        30: 1,
        60: 2,
    }

    assert PVOD_TCN_ARCHITECTURE[
        "candidate_id"
    ] == "TCN_C1"

    assert PVOD_TCN_ARCHITECTURE[
        "kernel_size"
    ] == 3

    assert PVOD_TCN_ARCHITECTURE[
        "convolutions_per_block"
    ] == 2

    assert PVOD_TCN_EPOCHS == {
        15: 4,
        30: 1,
        60: 1,
    }

    assert PVOD_F3_AUXILIARY_DIM == 30
    assert PVOD_F4_AUXILIARY_DIM == 127

    assert (
        PVOD_VALIDATION_POLICY[
            "hyperparameter_search"
        ]
        is False
    )

    assert (
        PVOD_VALIDATION_POLICY[
            "epoch_selection"
        ]
        is False
    )

    assert is_pvod_binary_auxiliary_column(
        "solar_daylight"
    )

    assert is_pvod_binary_auxiliary_column(
        "tithi_1"
    )

    assert not is_pvod_binary_auxiliary_column(
        "moon_phase_sin"
    )
