from unittest import mock

import pytest
import numpy as np

from ecephys_etl.modules.align_timestamps.barcode_sync_dataset import (
    BarcodeSyncDataset
)


@pytest.mark.parametrize(
    "line_labels,expected", [[["barcode"], 0], [["barcodes"], 0], [[], None]]
)
def test_barcode_line(line_labels, expected):

    dataset = BarcodeSyncDataset()
    dataset.line_labels = line_labels

    if expected is None:
        with pytest.raises(ValueError):
            obtained = dataset.barcode_line

    else:
        obtained = dataset.barcode_line
        assert obtained == expected


@pytest.mark.parametrize(
    "sample_frequency, rising_edges, falling_edges, times_exp, codes_exp",
    [
        [
            1,
            np.array([30, 50, 50.08]),
            np.array([31, 50.04, 50.12]),
            [50],
            [3]
        ]
    ],
)
def test_extract_barcodes(
    sample_frequency, rising_edges, falling_edges, times_exp, codes_exp
):

    dataset = BarcodeSyncDataset()
    dataset.sample_frequency = sample_frequency
    dataset.line_labels = ["barcode"]

    with mock.patch(
        "ecephys_etl.data_extractors.sync_dataset.Dataset.get_rising_edges",
        return_value=rising_edges,
    ):
        with mock.patch(
            "ecephys_etl.data_extractors.sync_dataset"
            ".Dataset.get_falling_edges",
            return_value=falling_edges,
        ):

            times, codes = dataset.extract_barcodes()

            assert np.allclose(times, times_exp)
            assert np.allclose(codes, codes_exp)
