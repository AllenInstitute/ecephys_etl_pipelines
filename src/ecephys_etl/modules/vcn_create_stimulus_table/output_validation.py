import numpy as np
import pandas as pd
import warnings


def validate_epoch_durations(
    table, start_key="Start", end_key="End", fail_on_negative_durations=False
):
    durations = table[end_key] - table[start_key]
    min_duration_index = durations.idxmin()
    min_duration = durations[min_duration_index]

    if min_duration == 0:
        warnings.warn(
            f"There is an epoch in this stimulus table "
            f"(index: {min_duration_index}) with duration = {min_duration}",
            UserWarning,
        )
    if min_duration < 0:
        msg = (
            f"there is an epoch with negative duration "
            f"(index: {min_duration_index})"
        )
        if fail_on_negative_durations:
            raise ValueError(msg)
        warnings.warn(msg)


def validate_epoch_order(table, time_keys=("Start", "End")):
    for time_key in time_keys:
        change = np.diff(table[time_key].values)
        assert np.amin(change) > 0


def validate_max_spontaneous_epoch_duration(
    table,
    max_duration,
    get_spontaneous_epochs=None,
    index_key="stimulus_index",
    start_key="Start",
    end_key="End",
):
    if get_spontaneous_epochs is None:

        def get_spontaneous_epochs(table: pd.DataFrame) -> pd.DataFrame:
            return table[np.isnan(table[index_key])]

    spontaneous_epochs = get_spontaneous_epochs(table)
    durations = (
        spontaneous_epochs[end_key].values
        - spontaneous_epochs[start_key].values
    )
    if np.amax(durations) > max_duration:
        warnings.warn(
            f"There is a spontaneous activity duration longer "
            f"than {max_duration}",
            UserWarning,
        )
