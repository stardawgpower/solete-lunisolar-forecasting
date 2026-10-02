import pandas as pd
import pytest

from feature_families import (
    FEATURE_FAMILY_COUNTS,
    PANCHANG_GROUP_SIZES,
    PANCHANG_INDICATOR_COUNT,
    RAW_PANCHANG_COLUMNS,
    encode_panchang_fixed_vocabulary,
    panchang_group_columns,
)


def canonical_example():
    return pd.DataFrame(
        {
            "valid_tithi": [1, 15, 16, 30],
            "valid_paksha": [
                "Shukla",
                "Shukla",
                "Krishna",
                "Krishna",
            ],
            "valid_karana": [
                "Kimstughna",
                "Bava",
                "Balava",
                "Naga",
            ],
            "valid_nakshatra": [
                "Ashwini",
                "Bharani",
                "Krittika",
                "Revati",
            ],
            "valid_yoga": [
                "Vishkambha",
                "Priti",
                "Ayushman",
                "Vaidhriti",
            ],
        }
    )


def test_frozen_feature_family_dimensions():
    assert FEATURE_FAMILY_COUNTS == {
        "F0": 118,
        "F1": 133,
        "F2": 138,
        "F3": 147,
        "F4": 244,
    }

    assert PANCHANG_INDICATOR_COUNT == 97


def test_fixed_vocabulary_encoder_always_returns_97_columns():
    encoded = encode_panchang_fixed_vocabulary(
        canonical_example()
    )

    assert encoded.shape == (4, 97)
    assert encoded.dtypes.eq("int8").all()
    assert not encoded.isna().any().any()


def test_each_encoded_row_has_exactly_five_active_indicators():
    encoded = encode_panchang_fixed_vocabulary(
        canonical_example()
    )

    assert (encoded.sum(axis=1) == 5).all()


def test_absent_categories_still_have_columns():
    one_row = canonical_example().iloc[[0]]

    encoded = encode_panchang_fixed_vocabulary(
        one_row
    )

    assert encoded.shape == (1, 97)

    assert "valid_tithi_30" in encoded.columns
    assert "valid_nakshatra_Revati" in encoded.columns
    assert "valid_yoga_Vaidhriti" in encoded.columns


def test_column_order_is_deterministic():
    first = encode_panchang_fixed_vocabulary(
        canonical_example()
    )

    second = encode_panchang_fixed_vocabulary(
        canonical_example().iloc[::-1]
    )

    assert list(first.columns) == list(second.columns)


def test_unrecognized_category_is_rejected():
    bad = canonical_example()
    bad.loc[0, "valid_paksha"] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Unexpected categories",
    ):
        encode_panchang_fixed_vocabulary(bad)


def test_wrong_column_structure_is_rejected():
    bad = canonical_example().rename(
        columns={
            "valid_tithi": "tithi",
        }
    )

    with pytest.raises(
        ValueError,
        match="Unexpected raw Panchang",
    ):
        encode_panchang_fixed_vocabulary(bad)


def test_panchang_group_column_sizes():
    encoded = encode_panchang_fixed_vocabulary(
        canonical_example()
    )

    groups = panchang_group_columns(
        encoded.columns
    )

    assert {
        family: len(columns)
        for family, columns in groups.items()
    } == PANCHANG_GROUP_SIZES


def test_raw_panchang_column_contract():
    assert RAW_PANCHANG_COLUMNS == (
        "valid_tithi",
        "valid_paksha",
        "valid_karana",
        "valid_nakshatra",
        "valid_yoga",
    )
