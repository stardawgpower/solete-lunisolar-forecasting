"""
Bootstrap uncertainty procedures used in the frozen forecasting study.

The completed experiments use two related but distinct resampling
designs.

SOLETE
------
A paired UTC-calendar-day block bootstrap. The frozen contrast is

    MAE_comparator - MAE_F4_real

so positive values indicate lower MAE for correctly aligned F4.

PVOD primary external validation
--------------------------------
A two-level hierarchical bootstrap:

1. resample stations with replacement;
2. independently resample China-local test days within every sampled
   station occurrence.

Stations receive equal weight. The frozen contrast is

    MAE_F4_real - MAE_F3

so negative values indicate lower MAE for F4.

PVOD secondary representation controls
---------------------------------------
The same station/day hierarchy is used, with the contrast

    MAE_real - MAE_control

so negative values indicate lower MAE for the correctly aligned real
representation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI_PERCENTILES = (
    2.5,
    97.5,
)

SOLETE_EXPECTED_TEST_DAYS = 92
PVOD_EXPECTED_STATIONS = 10


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


def solete_paired_day_block_bootstrap(
    prediction_table: pd.DataFrame,
    comparator_prediction_column: str,
    real_f4_prediction_column: str,
    scope: str,
    bootstrap_draws: np.ndarray,
    *,
    expected_daily_blocks: int | None = (
        SOLETE_EXPECTED_TEST_DAYS
    ),
) -> dict:
    """
    Apply the frozen SOLETE paired UTC-day block bootstrap.

    Parameters
    ----------
    prediction_table
        Table containing valid time, target, daylight flag, comparator
        prediction, and correctly aligned F4 prediction.

    comparator_prediction_column
        Prediction column for F3 or another frozen comparator.

    real_f4_prediction_column
        Prediction column for the correctly aligned F4 representation.

    scope
        Either ``"all"`` or ``"daylight"``.

    bootstrap_draws
        Integer matrix of shape ``(n_replicates, n_days)`` containing
        zero-based day-block indices.

    expected_daily_blocks
        Frozen SOLETE test data contain 92 UTC calendar-day blocks.
        Set to ``None`` only for generic/testing use.

    Returns
    -------
    dict
        Frozen point estimates, 95% percentile intervals, and CI
        classification.

    Notes
    -----
    The contrast is

        comparator MAE - F4_real MAE

    Positive values therefore mean F4_real has lower MAE.
    """

    if scope not in {
        "all",
        "daylight",
    }:
        raise ValueError(
            f"Unsupported scope: {scope}"
        )

    _require_columns(
        prediction_table,
        [
            "valid_time",
            "target_P_Solar_kW",
            "daylight",
            comparator_prediction_column,
            real_f4_prediction_column,
        ],
    )

    valid_time = prediction_table[
        "valid_time"
    ]

    if not pd.api.types.is_datetime64_any_dtype(
        valid_time
    ):
        raise ValueError(
            "valid_time must be datetime-like."
        )

    if scope == "all":
        scope_mask = np.ones(
            len(prediction_table),
            dtype=bool,
        )
    else:
        scope_mask = (
            prediction_table[
                "daylight"
            ]
            .to_numpy(
                dtype=bool
            )
        )

    scoped = (
        prediction_table
        .loc[
            scope_mask
        ]
        .copy()
    )

    if scoped.empty:
        raise ValueError(
            "Selected bootstrap scope is empty."
        )

    scoped[
        "_valid_utc_day"
    ] = (
        scoped[
            "valid_time"
        ]
        .dt.floor("D")
    )

    y_true = (
        scoped[
            "target_P_Solar_kW"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    comparator_pred = (
        scoped[
            comparator_prediction_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    real_pred = (
        scoped[
            real_f4_prediction_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not (
        np.isfinite(y_true).all()
        and np.isfinite(comparator_pred).all()
        and np.isfinite(real_pred).all()
    ):
        raise ValueError(
            "Bootstrap inputs contain non-finite values."
        )

    comparator_abs_error = np.abs(
        y_true - comparator_pred
    )

    real_abs_error = np.abs(
        y_true - real_pred
    )

    working = pd.DataFrame(
        {
            "day": scoped[
                "_valid_utc_day"
            ].to_numpy(),
            "comparator_abs_error":
                comparator_abs_error,
            "real_abs_error":
                real_abs_error,
        }
    )

    daily = (
        working
        .groupby(
            "day",
            sort=True,
        )
        .agg(
            comparator_error_sum=(
                "comparator_abs_error",
                "sum",
            ),
            real_error_sum=(
                "real_abs_error",
                "sum",
            ),
            row_count=(
                "comparator_abs_error",
                "size",
            ),
        )
    )

    n_days = len(daily)

    if (
        expected_daily_blocks is not None
        and n_days != expected_daily_blocks
    ):
        raise ValueError(
            f"Expected {expected_daily_blocks} daily blocks; "
            f"received {n_days}."
        )

    draws = np.asarray(
        bootstrap_draws
    )

    if draws.ndim != 2:
        raise ValueError(
            "bootstrap_draws must be a two-dimensional array."
        )

    if draws.shape[1] != n_days:
        raise ValueError(
            "Each bootstrap replicate must draw exactly "
            f"{n_days} day blocks."
        )

    if not np.issubdtype(
        draws.dtype,
        np.integer,
    ):
        raise TypeError(
            "bootstrap_draws must contain integer indices."
        )

    if (
        draws.size
        and (
            draws.min() < 0
            or draws.max() >= n_days
        )
    ):
        raise ValueError(
            "bootstrap_draws contains an out-of-range day index."
        )

    comparator_daily_sum = (
        daily[
            "comparator_error_sum"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    real_daily_sum = (
        daily[
            "real_error_sum"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    daily_count = (
        daily[
            "row_count"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    comparator_mae = float(
        comparator_abs_error.mean()
    )

    real_mae = float(
        real_abs_error.mean()
    )

    delta_mae = (
        comparator_mae
        - real_mae
    )

    improvement_percent = float(
        100.0
        * delta_mae
        / comparator_mae
    )

    sampled_counts = (
        daily_count[
            draws
        ]
        .sum(
            axis=1
        )
    )

    sampled_comparator_mae = (
        comparator_daily_sum[
            draws
        ]
        .sum(
            axis=1
        )
        / sampled_counts
    )

    sampled_real_mae = (
        real_daily_sum[
            draws
        ]
        .sum(
            axis=1
        )
        / sampled_counts
    )

    bootstrap_delta = (
        sampled_comparator_mae
        - sampled_real_mae
    )

    bootstrap_percent = (
        100.0
        * bootstrap_delta
        / sampled_comparator_mae
    )

    ci_low_kw, ci_high_kw = np.percentile(
        bootstrap_delta,
        BOOTSTRAP_CI_PERCENTILES,
    )

    (
        ci_low_percent,
        ci_high_percent,
    ) = np.percentile(
        bootstrap_percent,
        BOOTSTRAP_CI_PERCENTILES,
    )

    if ci_low_kw > 0:
        ci_zero_status = "above_zero"
    elif ci_high_kw < 0:
        ci_zero_status = "below_zero"
    else:
        ci_zero_status = "crosses_zero"

    return {
        "n_rows": int(
            len(scoped)
        ),
        "daily_blocks": int(
            n_days
        ),
        "comparator_MAE_kW":
            comparator_mae,
        "F4_real_MAE_kW":
            real_mae,
        "comparator_minus_F4_real_MAE_kW":
            float(delta_mae),
        "F4_real_improvement_percent":
            improvement_percent,
        "CI95_low_kW":
            float(ci_low_kw),
        "CI95_high_kW":
            float(ci_high_kw),
        "CI95_low_percent":
            float(ci_low_percent),
        "CI95_high_percent":
            float(ci_high_percent),
        "CI_zero_status":
            ci_zero_status,
    }


def pvod_primary_hierarchical_bootstrap(
    condition_daily: pd.DataFrame,
    rng: np.random.Generator,
    n_replicates: int,
    *,
    expected_stations: int = PVOD_EXPECTED_STATIONS,
) -> dict:
    """
    Apply the frozen PVOD primary hierarchical bootstrap.

    ``condition_daily`` must already be restricted to one
    model × horizon × evaluation-scope condition.

    The contrast is

        MAE_F4_real - MAE_F3

    and stations are equally weighted.
    """

    _require_columns(
        condition_daily,
        [
            "station",
            "bootstrap_china_date",
            "error_sum_F3",
            "error_sum_F4_real",
            "paired_row_count",
        ],
    )

    if n_replicates <= 0:
        raise ValueError(
            "n_replicates must be positive."
        )

    station_names = sorted(
        condition_daily[
            "station"
        ]
        .unique()
        .tolist()
    )

    if len(station_names) != expected_stations:
        raise ValueError(
            f"Expected {expected_stations} stations; "
            f"received {len(station_names)}."
        )

    payload = []

    for station_id in station_names:
        station_days = (
            condition_daily
            .loc[
                condition_daily[
                    "station"
                ].eq(
                    station_id
                )
            ]
            .sort_values(
                "bootstrap_china_date"
            )
            .reset_index(
                drop=True
            )
        )

        error_f3 = (
            station_days[
                "error_sum_F3"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        error_f4 = (
            station_days[
                "error_sum_F4_real"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        counts = (
            station_days[
                "paired_row_count"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not (
            len(error_f3)
            == len(error_f4)
            == len(counts)
        ):
            raise ValueError(
                "Station bootstrap arrays have unequal lengths."
            )

        if len(counts) == 0:
            raise ValueError(
                f"Station {station_id} has no test days."
            )

        if not np.all(
            counts > 0
        ):
            raise ValueError(
                "paired_row_count must be positive."
            )

        mae_f3 = float(
            error_f3.sum()
            / counts.sum()
        )

        mae_f4 = float(
            error_f4.sum()
            / counts.sum()
        )

        payload.append(
            {
                "station":
                    station_id,
                "error_f3":
                    error_f3,
                "error_f4":
                    error_f4,
                "counts":
                    counts,
                "n_days":
                    len(counts),
                "point_delta":
                    float(
                        mae_f4
                        - mae_f3
                    ),
            }
        )

    n_stations = len(payload)

    point_station_deltas = np.array(
        [
            station_data[
                "point_delta"
            ]
            for station_data in payload
        ],
        dtype=np.float64,
    )

    point_estimate = float(
        point_station_deltas.mean()
    )

    bootstrap_values = np.empty(
        n_replicates,
        dtype=np.float64,
    )

    for replicate_index in range(
        n_replicates
    ):
        station_draw = rng.integers(
            low=0,
            high=n_stations,
            size=n_stations,
        )

        station_delta_draw = np.empty(
            n_stations,
            dtype=np.float64,
        )

        for (
            draw_position,
            station_index,
        ) in enumerate(
            station_draw
        ):
            station_data = payload[
                int(
                    station_index
                )
            ]

            n_days = int(
                station_data[
                    "n_days"
                ]
            )

            day_draw = rng.integers(
                low=0,
                high=n_days,
                size=n_days,
            )

            sampled_counts = (
                station_data[
                    "counts"
                ][
                    day_draw
                ]
            )

            sampled_error_f3 = (
                station_data[
                    "error_f3"
                ][
                    day_draw
                ]
            )

            sampled_error_f4 = (
                station_data[
                    "error_f4"
                ][
                    day_draw
                ]
            )

            denominator = float(
                sampled_counts.sum()
            )

            if denominator <= 0.0:
                raise RuntimeError(
                    "Bootstrap denominator must be positive."
                )

            sampled_mae_f3 = float(
                sampled_error_f3.sum()
                / denominator
            )

            sampled_mae_f4 = float(
                sampled_error_f4.sum()
                / denominator
            )

            station_delta_draw[
                draw_position
            ] = (
                sampled_mae_f4
                - sampled_mae_f3
            )

        bootstrap_values[
            replicate_index
        ] = float(
            station_delta_draw.mean()
        )

    return {
        "point_estimate":
            point_estimate,
        "point_station_deltas":
            point_station_deltas,
        "bootstrap_values":
            bootstrap_values,
    }


def pvod_secondary_hierarchical_bootstrap(
    condition_daily: pd.DataFrame,
    rng: np.random.Generator,
    n_replicates: int,
    *,
    expected_stations: int = PVOD_EXPECTED_STATIONS,
) -> dict:
    """
    Apply the frozen PVOD secondary-control hierarchical bootstrap.

    The contrast is

        MAE_real - MAE_control

    so negative values indicate lower MAE for the correctly aligned
    real Panchang representation.
    """

    _require_columns(
        condition_daily,
        [
            "station",
            "target_china_date",
            "n_rows",
            "sum_abs_error_pu_real",
            "sum_abs_error_pu_control",
        ],
    )

    if n_replicates <= 0:
        raise ValueError(
            "n_replicates must be positive."
        )

    station_names = sorted(
        condition_daily[
            "station"
        ]
        .unique()
    )

    if len(station_names) != expected_stations:
        raise ValueError(
            f"Expected {expected_stations} stations; "
            f"received {len(station_names)}."
        )

    payload = []
    point_station_deltas = []

    for station_id in station_names:
        station = (
            condition_daily
            .loc[
                condition_daily[
                    "station"
                ].eq(
                    station_id
                )
            ]
            .sort_values(
                "target_china_date"
            )
            .reset_index(
                drop=True
            )
        )

        counts = (
            station[
                "n_rows"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        error_real = (
            station[
                "sum_abs_error_pu_real"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        error_control = (
            station[
                "sum_abs_error_pu_control"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(counts) == 0:
            raise ValueError(
                f"Station {station_id} has no test days."
            )

        if not np.all(
            counts > 0
        ):
            raise ValueError(
                "n_rows must be positive."
            )

        denominator = float(
            counts.sum()
        )

        point_delta = float(
            error_real.sum()
            / denominator
            - error_control.sum()
            / denominator
        )

        payload.append(
            {
                "counts":
                    counts,
                "error_real":
                    error_real,
                "error_control":
                    error_control,
                "n_days":
                    len(counts),
            }
        )

        point_station_deltas.append(
            point_delta
        )

    point_station_deltas = np.asarray(
        point_station_deltas,
        dtype=np.float64,
    )

    point_estimate = float(
        point_station_deltas.mean()
    )

    n_stations = len(
        station_names
    )

    station_draw = rng.integers(
        low=0,
        high=n_stations,
        size=(
            n_replicates,
            n_stations,
        ),
    )

    bootstrap_values = np.zeros(
        n_replicates,
        dtype=np.float64,
    )

    for occurrence_position in range(
        n_stations
    ):
        selected_station = station_draw[
            :,
            occurrence_position,
        ]

        occurrence_delta = np.empty(
            n_replicates,
            dtype=np.float64,
        )

        for (
            station_index,
            station_data,
        ) in enumerate(
            payload
        ):
            replicate_positions = np.flatnonzero(
                selected_station
                == station_index
            )

            if len(
                replicate_positions
            ) == 0:
                continue

            n_days = int(
                station_data[
                    "n_days"
                ]
            )

            day_draw = rng.integers(
                low=0,
                high=n_days,
                size=(
                    len(
                        replicate_positions
                    ),
                    n_days,
                ),
            )

            sampled_counts = (
                station_data[
                    "counts"
                ][
                    day_draw
                ]
            )

            denominator = (
                sampled_counts
                .sum(
                    axis=1
                )
            )

            if not np.all(
                denominator > 0.0
            ):
                raise RuntimeError(
                    "Bootstrap denominator must be positive."
                )

            sampled_real = (
                station_data[
                    "error_real"
                ][
                    day_draw
                ]
                .sum(
                    axis=1
                )
                / denominator
            )

            sampled_control = (
                station_data[
                    "error_control"
                ][
                    day_draw
                ]
                .sum(
                    axis=1
                )
                / denominator
            )

            occurrence_delta[
                replicate_positions
            ] = (
                sampled_real
                - sampled_control
            )

        bootstrap_values += (
            occurrence_delta
            / n_stations
        )

    return {
        "point_estimate":
            point_estimate,
        "point_station_deltas":
            point_station_deltas,
        "bootstrap_values":
            bootstrap_values,
    }
