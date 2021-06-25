import numpy as np

from ecephys_etl.modules.align_timestamps import barcode


def extract_barcodes_from_states(
    channel_states: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float,
    inter_barcode_interval: float = 10.0,
    bar_duration: float = 0.03,
    barcode_duration_ceiling: float = 2.0,
    nbits: int = 32,
):
    """Obtain barcodes from timestamped rising/falling edges.

    Parameters
    ----------
    channel_states : numpy.ndarray
        Rising and falling edges, denoted 1 and -1
    timestamps : numpy.ndarray
        Sample index of each event.
    sampling_rate : numeric
        Samples / second
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode
    """
    on_events = np.where(channel_states == 1)
    off_events = np.where(channel_states == -1)

    T_on = timestamps[on_events] / float(sampling_rate)
    T_off = timestamps[off_events] / float(sampling_rate)

    return barcode.extract_barcodes_from_times(
        on_times=T_on,
        off_times=T_off,
        inter_barcode_interval=inter_barcode_interval,
        bar_duration=bar_duration,
        barcode_duration_ceiling=barcode_duration_ceiling,
        nbits=nbits
    )


def extract_splits_from_states(
    channel_states: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float
) -> np.ndarray:
    """Obtain barcodes from timestamped rising/falling edges.

    Parameters
    ----------
    channel_states : numpy.ndarray
        Rising and falling edges, denoted 1 and -1
    timestamps : numpy.ndarray
        Sample index of each event.
    sampling_rate : numeric
        Samples / second
    """
    split_events = np.where(channel_states == 0)

    T_split = timestamps[split_events] / float(sampling_rate)

    if len(T_split) == 0:
        T_split = np.array([0])

    return T_split
