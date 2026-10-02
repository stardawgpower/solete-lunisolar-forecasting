import numpy as np
import pandas as pd
import pytest

from feature_families import FEATURE_FAMILY_COUNTS
from forecasting_protocol import (
    DAILY_LAG_STEPS,
    F0_FEATURE_COLUMNS,
    F0_HISTORY_SOURCES,
    FORECAST_HORIZONS,
    MODEL_FEATURE_FAMILIES,
    RECENT_LAGS,
    TARGET_COLUMN,
    TEST_END,
    TEST_START,
    TRAIN_START,
    VALID_START,
    assign_solete_split,
    build_base_modelling_tables,
    build_f0_history,
    build_forecast_targets,
    build_modelling_bundle,
    validate_solete_5min_index,
)


def make_clean_frame(
    start="2019-01-01 00:00:00",
    periods=400,
):
    index = pd.date_range(
        start,
        periods=periods,
        freq="5min",
        tz="UTC",
    )

    data = {}

    for offset, column in enumerate(
        dict.fromkeys(
            F0_HISTORY_SOURCES.values()
        )
    ):
        data[column] = (
            np.arange(
                periods,
                dtype=float,
            )
            + float(offset)
        )

    return pd.DataFrame(
        data,
        index=index,
    )


def test_frozen_protocol_constants():
    assert RECENT_LAGS == tuple(
        range(13)
    )

    assert DAILY_LAG_STEPS == 288

    assert FORECAST_HORIZONS == {
        15: 3,
        30: 6,
        60: 12,
    }

    assert TRAIN_START == pd.Timestamp(
        "2018-06-01 00:00:00",
        tz="UTC",
    )

    assert VALID_START == pd.Timestamp(
        "2019-04-01 00:00:00",
        tz="UTC",
    )

    assert TEST_START == pd.Timestamp(
        "2019-06-01 00:00:00",
        tz="UTC",
    )

    assert TEST_END == pd.Timestamp(
        "2019-09-01 00:00:00",
        tz="UTC",
    )


def test_f0_history_matches_frozen_lag_structure():
    clean = make_clean_frame()

    history = build_f0_history(
        clean
    )

    assert history.shape == (
        len(clean),
        118,
    )

    assert tuple(
        history.columns
    ) == F0_FEATURE_COLUMNS

    assert history[
        "P_Solar_kw_lag_0"
    ].iloc[10] == pytest.approx(
        clean[
            TARGET_COLUMN
        ].iloc[10]
    )

    assert history[
        "P_Solar_kw_lag_1"
    ].iloc[10] == pytest.approx(
        clean[
            TARGET_COLUMN
        ].iloc[9]
    )

    assert history[
        "P_Solar_kw_lag_12"
    ].iloc[12] == pytest.approx(
        clean[
            TARGET_COLUMN
        ].iloc[0]
    )

    assert np.isnan(
        history[
            "P_Solar_kw_lag_288"
        ].iloc[
            DAILY_LAG_STEPS - 1
        ]
    )

    assert history[
        "P_Solar_kw_lag_288"
    ].iloc[
        DAILY_LAG_STEPS
    ] == pytest.approx(
        clean[
            TARGET_COLUMN
        ].iloc[0]
    )


def test_incomplete_five_minute_grid_is_rejected():
    clean = make_clean_frame(
        periods=20
    )

    broken = clean.drop(
        clean.index[7]
    )

    with pytest.raises(
        ValueError,
        match="complete 5-minute grid",
    ):
        validate_solete_5min_index(
            broken
        )


def test_forecast_targets_use_future_valid_time():
    clean = make_clean_frame(
        periods=20
    )

    targets = build_forecast_targets(
        clean
    )

    assert set(
        targets
    ) == {
        15,
        30,
        60,
    }

    target_15 = targets[
        15
    ]

    assert target_15[
        "issue_time"
    ].iloc[0] == clean.index[0]

    assert target_15[
        "valid_time"
    ].iloc[0] == (
        clean.index[0]
        + pd.Timedelta(
            minutes=15
        )
    )

    assert target_15[
        "target_P_Solar_kW"
    ].iloc[0] == pytest.approx(
        clean[
            TARGET_COLUMN
        ].iloc[3]
    )

    assert target_15[
        "target_P_Solar_kW"
    ].notna().sum() == 17

    assert targets[
        60
    ][
        "target_P_Solar_kW"
    ].notna().sum() == 8


def test_split_assignment_uses_frozen_valid_time_boundaries():
    valid_time = pd.Series(
        pd.DatetimeIndex(
            [
                TRAIN_START,
                VALID_START
                - pd.Timedelta(
                    minutes=5
                ),
                VALID_START,
                TEST_START
                - pd.Timedelta(
                    minutes=5
                ),
                TEST_START,
                TEST_END
                - pd.Timedelta(
                    minutes=5
                ),
                TEST_END,
            ]
        )
    )

    split = assign_solete_split(
        valid_time
    )

    assert split.tolist() == [
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
        pd.NA,
    ]


def test_base_modelling_table_uses_valid_time_and_exclusive_test_end():
    index = pd.date_range(
        "2019-08-30 00:00:00",
        "2019-09-01 01:00:00",
        freq="5min",
        tz="UTC",
    )

    clean = make_clean_frame(
        start=index[0],
        periods=len(index),
    )

    base = build_base_modelling_tables(
        clean
    )

    for (
        horizon,
        table,
    ) in base.items():
        assert not table.empty

        assert (
            table[
                "valid_time"
            ]
            < TEST_END
        ).all()

        assert table[
            "valid_time"
        ].max() == pd.Timestamp(
            "2019-08-31 23:55:00",
            tz="UTC",
        )

        assert table[
            "split"
        ].eq(
            "test"
        ).all()

        assert (
            table[
                "valid_time"
            ]
            - table[
                "issue_time"
            ]
        ).eq(
            pd.Timedelta(
                minutes=horizon
            )
        ).all()


def make_bundle_inputs():
    valid_times = pd.DatetimeIndex(
        [
            pd.Timestamp(
                "2019-03-31 23:55:00",
                tz="UTC",
            ),
            pd.Timestamp(
                "2019-04-01 00:00:00",
                tz="UTC",
            ),
            pd.Timestamp(
                "2019-06-01 00:00:00",
                tz="UTC",
            ),
        ]
    )

    index = (
        valid_times
        - pd.Timedelta(
            minutes=15
        )
    )

    base = pd.DataFrame(
        {
            "issue_time":
                index,
            "valid_time":
                valid_times,
            "horizon_minutes":
                15,
            "split":
                [
                    "train",
                    "validation",
                    "test",
                ],
            "target_P_Solar_kW":
                [
                    1.0,
                    2.0,
                    3.0,
                ],
        },
        index=index,
    )

    feature_tables = {}

    for family in (
        MODEL_FEATURE_FAMILIES
    ):
        count = FEATURE_FAMILY_COUNTS[
            family
        ]

        feature_tables[
            family
        ] = pd.DataFrame(
            np.zeros(
                (
                    len(index),
                    count,
                )
            ),
            index=index,
            columns=[
                f"{family}_{i}"
                for i in range(
                    count
                )
            ],
        )

    solar = pd.DataFrame(
        {
            "valid_solar_daylight":
                [
                    0,
                    1,
                    1,
                ],
        },
        index=index,
    )

    return (
        base,
        feature_tables,
        solar,
    )


def test_modelling_bundle_preserves_alignment_and_masks():
    (
        base,
        feature_tables,
        solar,
    ) = make_bundle_inputs()

    bundle = build_modelling_bundle(
        base,
        feature_tables,
        solar,
    )

    assert bundle[
        "y"
    ].index.equals(
        base.index
    )

    assert bundle[
        "train_mask"
    ].tolist() == [
        True,
        False,
        False,
    ]

    assert bundle[
        "validation_mask"
    ].tolist() == [
        False,
        True,
        False,
    ]

    assert bundle[
        "test_mask"
    ].tolist() == [
        False,
        False,
        True,
    ]

    assert bundle[
        "daylight_mask"
    ].tolist() == [
        False,
        True,
        True,
    ]

    for family in (
        MODEL_FEATURE_FAMILIES
    ):
        assert bundle[
            "X"
        ][
            family
        ].shape[1] == (
            FEATURE_FAMILY_COUNTS[
                family
            ]
        )


def test_modelling_bundle_rejects_misaligned_features():
    (
        base,
        feature_tables,
        solar,
    ) = make_bundle_inputs()

    feature_tables = dict(
        feature_tables
    )

    feature_tables[
        "F3"
    ] = (
        feature_tables[
            "F3"
        ]
        .iloc[
            ::-1
        ]
    )

    with pytest.raises(
        ValueError,
        match="F3.*alignment",
    ):
        build_modelling_bundle(
            base,
            feature_tables,
            solar,
        )


def test_modelling_bundle_rejects_wrong_feature_dimension():
    (
        base,
        feature_tables,
        solar,
    ) = make_bundle_inputs()

    feature_tables = dict(
        feature_tables
    )

    feature_tables[
        "F4"
    ] = (
        feature_tables[
            "F4"
        ]
        .iloc[
            :,
            :-1
        ]
    )

    with pytest.raises(
        ValueError,
        match="F4: expected 244",
    ):
        build_modelling_bundle(
            base,
            feature_tables,
            solar,
        )
