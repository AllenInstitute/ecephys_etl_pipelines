import json
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt


from ecephys_etl.data_extractors import Dataset


@pytest.fixture(scope="module")
def mock_meta():
    labels = ['']*32
    labels[0] = "zero"
    labels[5] = "five"
    meta = {'total_samples': 100,
            'sampling_type': 'frequency',
            'timeouts': [],
            'start_time': '2018-03-30 13:29:06.993000',
            'ni_daq': {'device': 'Dev1',
                       'event_bits': 32,
                       'counter_bits': 32,
                       'sample_rate': 100000.0,
                       'counter_output_freq': 100000.0},
            'version': "1.4.0",
            'stop_time': '2018-03-30 13:29:06.997000',
            'line_labels': labels}
    return meta


@pytest.fixture(scope="module")
def mock_data():
    data = np.zeros(shape=(100, 2), dtype="<u4")
    data[2:-1, 1] = 1
    for i in range(1, 100):
        data[i, 0] = i
        if i % 2 == 0:
            data[i, 1] |= (1 << 5)
    return data


@pytest.fixture(scope="module")
def sync_file(tmpdir_factory, mock_data, mock_meta):
    filename = str(tmpdir_factory.mktemp("test").join("data.h5"))
    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=mock_data)
        f.create_dataset("meta", data=json.dumps(mock_meta))
    return filename


def test_dataset(sync_file, mock_meta):
    dset = Dataset(sync_file)
    assert(dset.meta_data == mock_meta)
    assert(dset.line_labels == mock_meta["line_labels"])
    assert(dset.sample_freq == mock_meta["ni_daq"]["sample_rate"])


def test_line_bit_conversions(sync_file, mock_meta):
    dset = Dataset(sync_file)
    for i in range(32):
        assert(dset._line_to_bit(i) == i)
        assert(dset._bit_to_line(i) == mock_meta["line_labels"][i])
    assert(dset._line_to_bit("zero") == 0)
    assert(dset._line_to_bit("five") == 5)
    with pytest.raises(ValueError):
        dset._line_to_bit("foo")
    with pytest.raises(TypeError):
        dset._line_to_bit(("tuple",))


@pytest.mark.parametrize("bit,auto_show", [
    (0, False),
    (6, True)
])
@patch("matplotlib.pyplot.show")
def test_plot_bit(mock_show, bit, auto_show, sync_file):
    dset = Dataset(sync_file)
    dset.plot_bit(bit, auto_show=auto_show)
    if auto_show:
        mock_show.assert_called_once()
    else:
        assert(mock_show.call_count == 0)
    f = plt.figure()
    ax = f.add_subplot(111)
    with patch("matplotlib.pyplot.figure") as mock_figure:
        dset.plot_bit(bit, axes=ax)
        assert(mock_figure.call_count == 0)
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_line(mock_show, sync_file):
    dset = Dataset(sync_file)
    dset.plot_line("zero")
    mock_show.assert_called_once()
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_plot_lines(mock_show, sync_file):
    dset = Dataset(sync_file)
    dset.plot_lines(["zero", "five"])
    mock_show.assert_called_once()
    plt.close("all")
