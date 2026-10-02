import numpy as np
import pandas as pd
import pytest

from feature_families import (
    KARANA_CATEGORIES,
    NAKSHATRA_CATEGORIES,
    YOGA_CATEGORIES,
    encode_panchang_fixed_vocabulary,
)
from representation_controls import (
    CORE_244_VARIANTS,
    EXPECTED_COMPONENT_COUNTS,
    PERMUTATION_SEEDS,
    SHIFT_DAYS,
    SPLIT_ORDER,
    splitwise_circular_shift,
    splitwise_joint_permutation,
)


def canonical_control_data():
    index = pd.date_range(
        "2019-01-01",
        periods=9,
        freq="5min",
        tz="UTC",
    )

    raw = pd.DataFrame(
        {
            "valid_tithi": list(
                range(1, 10)
            ),
            "valid_paksha": [
                "Shukla"
            ] * 9,
            "valid_karana": [
                KARANA_CATEGORIES[
                    i % len(KARANA_CATEGORIES)
                ]
                for i in range(9)
            ],
            "valid_nakshatra": [
                NAKSHATRA_CATEGORIES[
                    i % len(NAKSHATRA_CATEGORIES)
                ]
                for i in range(9)
            ],
            "valid_yoga": [
                YOGA_CATEGORIES[
                    i % len(YOGA_CATEGORIES)
                ]
                for i in range(9)
            ],
        },
        index=index,
    )

    block = encode_panchang_fixed_vocabulary(
        raw
    )

    metadata = pd.DataFrame(
        {
            "split": (
                ["train"] * 3
                + ["validation"] * 3
                + ["test"] * 3
            )
        },
        index=index,
    )

    return block, metadata


def test_frozen_control_constants():
    assert SHIFT_DAYS == (
        17,
        37,
        61,
    )

    assert PERMUTATION_SEEDS == (
        1301,
        1302,
        1303,
    )

    assert CORE_244_VARIANTS == (
        "F4_real",
        "F4_shift_17d",
        "F4_shift_37d",
        "F4_shift_61d",
        "F4_perm_1301",
        "F4_perm_1302",
        "F4_perm_1303",
    )


def test_frozen_component_dimensions():
    assert EXPECTED_COMPONENT_COUNTS == {
        "F3_plus_tithi": 177,
        "F3_plus_paksha": 149,
        "F3_plus_karana": 158,
        "F3_plus_nakshatra": 174,
        "F3_plus_yoga": 174,
    }


def test_joint_permutation_matches_frozen_rng_protocol():
    block, metadata = canonical_control_data()

    result = splitwise_joint_permutation(
        block,
        metadata,
        seed=1301,
    )

    for offset, split_name in enumerate(
        SPLIT_ORDER
    ):
        index = metadata.index[
            metadata["split"].eq(split_name)
        ]

        original = block.loc[
            index
        ].to_numpy()

        expected_order = (
            np.random.default_rng(
                1301 + offset
            )
            .permutation(
                len(original)
            )
        )

        expected = original[
            expected_order
        ]

        np.testing.assert_array_equal(
            result.loc[
                index
            ].to_numpy(),
            expected,
        )


def test_joint_permutation_is_deterministic():
    block, metadata = canonical_control_data()

    first = splitwise_joint_permutation(
        block,
        metadata,
        seed=1301,
    )

    second = splitwise_joint_permutation(
        block,
        metadata,
        seed=1301,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_joint_permutation_never_crosses_split_boundaries():
    block, metadata = canonical_control_data()

    result = splitwise_joint_permutation(
        block,
        metadata,
        seed=1301,
    )

    for split_name in SPLIT_ORDER:
        index = metadata.index[
            metadata["split"].eq(split_name)
        ]

        original_rows = sorted(
            map(
                tuple,
                block.loc[
                    index
                ].to_numpy(),
            )
        )

        result_rows = sorted(
            map(
                tuple,
                result.loc[
                    index
                ].to_numpy(),
            )
        )

        assert result_rows == original_rows


def test_circular_shift_matches_frozen_splitwise_protocol():
    block, metadata = canonical_control_data()

    result = splitwise_circular_shift(
        block,
        metadata,
        shift_days=1,
        resolution_minutes=720,
    )

    for split_name in SPLIT_ORDER:
        index = metadata.index[
            metadata["split"].eq(split_name)
        ]

        original = block.loc[
            index
        ].to_numpy()

        expected = np.roll(
            original,
            shift=2,
            axis=0,
        )

        np.testing.assert_array_equal(
            result.loc[
                index
            ].to_numpy(),
            expected,
        )


@pytest.mark.parametrize(
    "control",
    [
        lambda block, metadata:
            splitwise_joint_permutation(
                block,
                metadata,
                seed=1301,
            ),
        lambda block, metadata:
            splitwise_circular_shift(
                block,
                metadata,
                shift_days=17,
            ),
    ],
)
def test_controls_preserve_canonical_row_structure(
    control,
):
    block, metadata = canonical_control_data()

    result = control(
        block,
        metadata,
    )

    assert result.shape == block.shape
    assert result.index.equals(block.index)
    assert result.columns.equals(block.columns)
    assert result.dtypes.eq("int8").all()
    assert (result.sum(axis=1) == 5).all()


def test_misaligned_metadata_is_rejected():
    block, metadata = canonical_control_data()

    metadata = metadata.iloc[::-1]

    with pytest.raises(
        ValueError,
        match="identical indices",
    ):
        splitwise_joint_permutation(
            block,
            metadata,
            seed=1301,
        )


def test_noncanonical_split_labels_are_rejected():
    block, metadata = canonical_control_data()

    metadata = metadata.copy()
    metadata.loc[
        metadata.index[-1],
        "split",
    ] = "future"

    with pytest.raises(
        ValueError,
        match="Expected exactly",
    ):
        splitwise_joint_permutation(
            block,
            metadata,
            seed=1301,
        )


def test_noncanonical_block_dimension_is_rejected():
    block, metadata = canonical_control_data()

    bad = block.iloc[:, :-1]

    with pytest.raises(
        ValueError,
        match="97 columns",
    ):
        splitwise_circular_shift(
            bad,
            metadata,
            shift_days=17,
        )


def test_invalid_resolution_is_rejected():
    block, metadata = canonical_control_data()

    with pytest.raises(
        ValueError,
        match="divide 1440",
    ):
        splitwise_circular_shift(
            block,
            metadata,
            shift_days=17,
            resolution_minutes=7,
        )
