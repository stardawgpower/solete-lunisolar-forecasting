import numpy as np

from vedic_calendar import (
    circular_difference_deg,
    karana_name_from_slot,
    karana_slot_from_phase,
    paksha_from_tithi,
    sidereal_longitude_from_tropical,
    tithi_number_from_phase,
    tithi_progress_from_phase,
    tithi_within_paksha,
    wrap_degrees,
)


def test_wrap_degrees_boundary_cases():
    values = np.array([-1.0, 0.0, 360.0, 721.0])

    result = wrap_degrees(values)

    np.testing.assert_allclose(result, [359.0, 0.0, 0.0, 1.0])


def test_circular_difference_handles_zero_crossing():
    result = circular_difference_deg(
        np.array([359.0, 10.0]),
        np.array([1.0, 20.0]),
    )

    np.testing.assert_allclose(result, [2.0, 10.0])


def test_tithi_boundaries_and_wrapping():
    phases = np.array(
        [-1.0, 0.0, 11.999999, 12.0, 359.999999, 360.0]
    )

    result = tithi_number_from_phase(phases)

    np.testing.assert_array_equal(result, [30, 1, 1, 2, 30, 1])


def test_tithi_progress_resets_at_sector_boundary():
    phases = np.array([0.0, 6.0, 12.0, 18.0])

    result = tithi_progress_from_phase(phases)

    np.testing.assert_allclose(result, [0.0, 0.5, 0.0, 0.5])


def test_paksha_boundary_and_within_paksha_numbering():
    tithi = np.array([1, 15, 16, 30])

    paksha = paksha_from_tithi(tithi)
    within = tithi_within_paksha(tithi)

    np.testing.assert_array_equal(
        paksha,
        ["Shukla", "Shukla", "Krishna", "Krishna"],
    )
    np.testing.assert_array_equal(within, [1, 15, 1, 15])


def test_karana_slot_boundaries_and_wrapping():
    phases = np.array(
        [0.0, 5.999999, 6.0, 353.999999, 354.0, 360.0]
    )

    result = karana_slot_from_phase(phases)

    np.testing.assert_array_equal(result, [1, 1, 2, 59, 60, 1])


def test_karana_fixed_end_sequence():
    slots = np.array([1, 2, 58, 59, 60])

    names = karana_name_from_slot(slots)

    np.testing.assert_array_equal(
        names,
        [
            "Kimstughna",
            "Bava",
            "Shakuni",
            "Chatushpada",
            "Naga",
        ],
    )


def test_sidereal_longitude_wraps_after_ayanamsha_subtraction():
    tropical = np.array([10.0, 350.0])
    ayanamsha = np.array([24.0, 24.0])

    result = sidereal_longitude_from_tropical(
        tropical,
        ayanamsha,
    )

    np.testing.assert_allclose(result, [346.0, 326.0])
