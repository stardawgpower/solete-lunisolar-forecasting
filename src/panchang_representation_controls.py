"""
Frozen Panchang representation controls used in SOLETE Phase 2.

The controls operate on the complete 97-column canonical Panchang
one-hot block and are applied independently within the chronological
train, validation, and test partitions.

Two null-control families are implemented:

1. Joint row permutation
   Preserves dimensionality, row-wise Panchang combinations, and
   split-specific empirical category frequencies while destroying
   timestamp alignment and temporal persistence.

2. Circular shift
   Preserves dimensionality, row structure, cross-family combinations,
   and local temporal persistence while breaking correct temporal /
   astronomical alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_families import PANCHANG_INDICATOR_COUNT

SPLIT_ORDER = (
    "train",
    "validation",
    "test",
)

SHIFT_DAYS = (
    17,
    37,
    61,
)

PERMUTATION_SEEDS = (
    1301,
    1302,
    1303,
)

CORE_244_VARIANTS = (
    "F4_real",
    "F4_shift_17d",
    "F4_shift_37d",
    "F4_shift_61d",
    "F4_perm_1301",
    "F4_perm_1302",
    "F4_perm_1303",
)

EXPECTED_COMPONENT_COUNTS = {
    "F3_plus_tithi": 177,
    "F3_plus_paksha": 149,
    "F3_plus_karana": 158,
    "F3_plus_nakshatra": 174,
    "F3_plus_yoga": 174,
}


def _validate_control_inputs(
    block: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Validate the canonical Phase-2 control input contract."""

    if not isinstance(block, pd.DataFrame):
        raise TypeError("block must be a pandas DataFrame.")

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")

    if not block.index.equals(metadata.index):
        raise ValueError(
            "block and metadata must have identical indices."
        )

    if not block.index.is_unique:
        raise ValueError(
            "Control input index must be unique."
        )

    if "split" not in metadata.columns:
        raise ValueError(
            "metadata must contain a 'split' column."
        )

    if block.shape[1] != PANCHANG_INDICATOR_COUNT:
        raise ValueError(
            "Canonical Panchang control block must contain "
            f"{PANCHANG_INDICATOR_COUNT} columns; "
            f"received {block.shape[1]}."
        )

    if not block.columns.is_unique:
        raise ValueError(
            "Panchang control columns must be unique."
        )

    if metadata["split"].isna().any():
        raise ValueError(
            "metadata['split'] contains missing values."
        )

    observed_splits = set(
        metadata["split"].unique()
    )
    expected_splits = set(SPLIT_ORDER)

    if observed_splits != expected_splits:
        raise ValueError(
            "Expected exactly the chronological splits "
            f"{SPLIT_ORDER}; received "
            f"{tuple(sorted(observed_splits))}."
        )

    values = block.to_numpy()

    if not np.isin(values, (0, 1)).all():
        raise ValueError(
            "Canonical Panchang control block must be binary."
        )

    active_per_row = values.sum(axis=1)

    if not np.all(active_per_row == 5):
        raise ValueError(
            "Every canonical Panchang row must contain "
            "exactly five active indicators."
        )


def splitwise_joint_permutation(
    block: pd.DataFrame,
    metadata: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    Jointly permute complete Panchang rows within each split.

    The random-number generator is reinitialized independently for
    train, validation, and test using ``seed + split_offset``, matching
    the frozen Phase-2 implementation.
    """

    _validate_control_inputs(
        block,
        metadata,
    )

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    out = pd.DataFrame(
        index=block.index,
        columns=block.columns,
        dtype="int8",
    )

    for split_offset, split_name in enumerate(
        SPLIT_ORDER
    ):
        split_mask = metadata["split"].eq(
            split_name
        )
        split_index = metadata.index[
            split_mask
        ]

        values = block.loc[
            split_index
        ].to_numpy(
            copy=True
        )

        rng = np.random.default_rng(
            int(seed) + split_offset
        )
        order = rng.permutation(
            len(values)
        )

        out.loc[
            split_index,
            :,
        ] = values[
            order
        ]

    return out.astype("int8")


def splitwise_circular_shift(
    block: pd.DataFrame,
    metadata: pd.DataFrame,
    shift_days: int,
    resolution_minutes: int = 5,
) -> pd.DataFrame:
    """
    Circularly shift complete Panchang rows within each split.

    The shift is expressed in calendar days and converted to rows using
    the supplied regular sampling resolution.
    """

    _validate_control_inputs(
        block,
        metadata,
    )

    if not isinstance(
        shift_days,
        (int, np.integer),
    ):
        raise TypeError(
            "shift_days must be an integer."
        )

    if not isinstance(
        resolution_minutes,
        (int, np.integer),
    ):
        raise TypeError(
            "resolution_minutes must be an integer."
        )

    resolution_minutes = int(
        resolution_minutes
    )

    if resolution_minutes <= 0:
        raise ValueError(
            "resolution_minutes must be positive."
        )

    if (24 * 60) % resolution_minutes != 0:
        raise ValueError(
            "resolution_minutes must divide 1440 exactly."
        )

    steps_per_day = (
        24 * 60
    ) // resolution_minutes

    shift_steps = (
        int(shift_days)
        * steps_per_day
    )

    out = pd.DataFrame(
        index=block.index,
        columns=block.columns,
        dtype="int8",
    )

    for split_name in SPLIT_ORDER:
        split_mask = metadata["split"].eq(
            split_name
        )
        split_index = metadata.index[
            split_mask
        ]

        values = block.loc[
            split_index
        ].to_numpy(
            copy=True
        )

        effective_shift = (
            shift_steps
            % len(values)
        )

        shifted = np.roll(
            values,
            shift=effective_shift,
            axis=0,
        )

        out.loc[
            split_index,
            :,
        ] = shifted

    return out.astype("int8")
