import numpy as np
import pandas as pd
import pytest

from bootstrap import (
    BOOTSTRAP_CI_PERCENTILES,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PVOD_EXPECTED_STATIONS,
    SOLETE_EXPECTED_TEST_DAYS,
    pvod_primary_hierarchical_bootstrap,
    pvod_secondary_hierarchical_bootstrap,
    solete_paired_day_block_bootstrap,
)


def test_frozen_bootstrap_constants():
    assert BOOTSTRAP_REPLICATES == 10_000
    assert BOOTSTRAP_SEED == 42
    assert BOOTSTRAP_CI_PERCENTILES == (
        2.5,
        97.5,
    )
    assert SOLETE_EXPECTED_TEST_DAYS == 92
    assert PVOD_EXPECTED_STATIONS == 10


def make_solete_table():
    valid_time = pd.date_range(
        "2019-06-01",
        periods=6,
        freq="12h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "valid_time":
                valid_time,
            "target_P_Solar_kW":
                [2.0] * 6,
            "daylight":
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ],
            "comparator":
                [4.0] * 6,
            "F4_real":
                [3.0] * 6,
        }
    )


def test_solete_bootstrap_sign_convention():
    table = make_solete_table()

    draws = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )

    result = solete_paired_day_block_bootstrap(
        table,
        "comparator",
        "F4_real",
        "all",
        draws,
        expected_daily_blocks=3,
    )

    assert result[
        "comparator_MAE_kW"
    ] == pytest.approx(2.0)

    assert result[
        "F4_real_MAE_kW"
    ] == pytest.approx(1.0)

    assert result[
        "comparator_minus_F4_real_MAE_kW"
    ] == pytest.approx(1.0)

    assert result[
        "F4_real_improvement_percent"
    ] == pytest.approx(50.0)

    assert result[
        "CI_zero_status"
    ] == "above_zero"


def test_solete_daylight_scope_is_applied_before_bootstrap():
    table = make_solete_table()

    table.loc[
        ~table["daylight"],
        "comparator",
    ] = 20.0

    draws = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=int,
    )

    result = solete_paired_day_block_bootstrap(
        table,
        "comparator",
        "F4_real",
        "daylight",
        draws,
        expected_daily_blocks=3,
    )

    assert result["n_rows"] == 3
    assert result[
        "comparator_MAE_kW"
    ] == pytest.approx(2.0)


def test_solete_invalid_scope_is_rejected():
    table = make_solete_table()

    with pytest.raises(
        ValueError,
        match="Unsupported scope",
    ):
        solete_paired_day_block_bootstrap(
            table,
            "comparator",
            "F4_real",
            "night",
            np.zeros(
                (2, 3),
                dtype=int,
            ),
            expected_daily_blocks=3,
        )


def test_solete_out_of_range_draw_is_rejected():
    table = make_solete_table()

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        solete_paired_day_block_bootstrap(
            table,
            "comparator",
            "F4_real",
            "all",
            np.array(
                [
                    [0, 1, 3],
                ],
                dtype=int,
            ),
            expected_daily_blocks=3,
        )


def make_pvod_primary_daily():
    rows = []

    for station_index in range(10):
        station = f"{station_index:02d}"

        rows.extend(
            [
                {
                    "station":
                        station,
                    "bootstrap_china_date":
                        "2020-01-01",
                    "error_sum_F3":
                        2.0,
                    "error_sum_F4_real":
                        1.0,
                    "paired_row_count":
                        2.0,
                },
                {
                    "station":
                        station,
                    "bootstrap_china_date":
                        "2020-01-02",
                    "error_sum_F3":
                        4.0,
                    "error_sum_F4_real":
                        2.0,
                    "paired_row_count":
                        2.0,
                },
            ]
        )

    return pd.DataFrame(rows)


def test_pvod_primary_point_estimate_uses_f4_minus_f3():
    daily = make_pvod_primary_daily()

    result = pvod_primary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=5,
    )

    assert result[
        "point_station_deltas"
    ].shape == (10,)

    assert np.allclose(
        result[
            "point_station_deltas"
        ],
        -0.75,
    )

    assert result[
        "point_estimate"
    ] == pytest.approx(
        -0.75
    )


def test_pvod_primary_bootstrap_is_seed_reproducible():
    daily = make_pvod_primary_daily()

    first = pvod_primary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=20,
    )

    second = pvod_primary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=20,
    )

    np.testing.assert_array_equal(
        first["bootstrap_values"],
        second["bootstrap_values"],
    )


def make_pvod_secondary_daily():
    rows = []

    for station_index in range(10):
        station = f"{station_index:02d}"

        rows.extend(
            [
                {
                    "station":
                        station,
                    "target_china_date":
                        "2020-01-01",
                    "n_rows":
                        2.0,
                    "sum_abs_error_pu_real":
                        1.0,
                    "sum_abs_error_pu_control":
                        2.0,
                },
                {
                    "station":
                        station,
                    "target_china_date":
                        "2020-01-02",
                    "n_rows":
                        2.0,
                    "sum_abs_error_pu_real":
                        2.0,
                    "sum_abs_error_pu_control":
                        4.0,
                },
            ]
        )

    return pd.DataFrame(rows)


def test_pvod_secondary_point_estimate_uses_real_minus_control():
    daily = make_pvod_secondary_daily()

    result = pvod_secondary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=5,
    )

    assert np.allclose(
        result[
            "point_station_deltas"
        ],
        -0.75,
    )

    assert result[
        "point_estimate"
    ] == pytest.approx(
        -0.75
    )


def test_pvod_secondary_bootstrap_is_seed_reproducible():
    daily = make_pvod_secondary_daily()

    first = pvod_secondary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=20,
    )

    second = pvod_secondary_hierarchical_bootstrap(
        daily,
        np.random.default_rng(42),
        n_replicates=20,
    )

    np.testing.assert_array_equal(
        first["bootstrap_values"],
        second["bootstrap_values"],
    )


@pytest.mark.parametrize(
    "bootstrap_function,daily_factory",
    [
        (
            pvod_primary_hierarchical_bootstrap,
            make_pvod_primary_daily,
        ),
        (
            pvod_secondary_hierarchical_bootstrap,
            make_pvod_secondary_daily,
        ),
    ],
)
def test_pvod_bootstrap_requires_all_ten_stations(
    bootstrap_function,
    daily_factory,
):
    daily = daily_factory()

    daily = daily.loc[
        ~daily[
            "station"
        ].eq("09")
    ]

    with pytest.raises(
        ValueError,
        match="Expected 10 stations",
    ):
        bootstrap_function(
            daily,
            np.random.default_rng(42),
            n_replicates=5,
        )
