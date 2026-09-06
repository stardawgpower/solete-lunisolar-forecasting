# ============================================================
# SOLETE DATA LOADER
# ============================================================
#
# This module contains reusable functions for loading the
# SOLETE renewable-energy dataset stored in HDF5 format.
#
# The same loader can be used for different SOLETE resolutions:
#   - 60-minute
#   - 5-minute
#   - 1-minute
#   - 1-second
#
# The resolution is determined by the input file.
# ============================================================

from pathlib import Path

import h5py
import pandas as pd


def load_solete(file_path: str | Path) -> pd.DataFrame:
    """
    Load a SOLETE HDF5 dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the SOLETE HDF5 file.

    Returns
    -------
    pandas.DataFrame
        Dataset with:
        - timestamps as the DatetimeIndex
        - measurement variables as columns
    """

    # Open the HDF5 file in read-only mode.
    with h5py.File(file_path, "r") as f:

        # The actual measurements are stored inside
        # the DATA group.
        data = f["DATA"]

        # Read the column names from the HDF5 metadata.
        # HDF5 stores these as byte strings, so we decode
        # them into normal Python strings.
        columns = [
            name.decode("utf-8")
            for name in data["axis0"][:]
        ]

        # Read timestamps.
        #
        # SOLETE timestamps are stored as nanoseconds since
        # Unix epoch. We convert them to pandas timestamps
        # and explicitly keep them timezone-aware in UTC.
        timestamps = pd.to_datetime(
            data["axis1"][:],
            unit="ns",
            utc=True,
        )

        # Read the actual measurement values.
        values = data["block0_values"][:]

    # Construct the DataFrame.
    df = pd.DataFrame(
        values,
        columns=columns,
        index=timestamps,
    )

    # Give the index a meaningful name.
    df.index.name = "Timestamp"

    return df