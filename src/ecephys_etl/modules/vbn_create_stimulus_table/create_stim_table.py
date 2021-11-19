from functools import partial
from typing import List, Tuple, Union
import logging

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance

from ecephys_etl.data_extractors.sync_dataset import Dataset
from ecephys_etl.data_extractors.stim_file import (
    CamStimOnePickleStimFile,
    BehaviorPickleFile,
    ReplayPickleFile
)
from ecephys_etl.modules.vcn_create_stimulus_table.ephys_pre_spikes import (
    build_stimuluswise_table,
    create_stim_table,
    make_spontaneous_activity_tables
)
from ecephys_etl.data_transformers.visual_behavior_stimulus_processing import (
    get_stimulus_presentations
)


def get_vsyncs(
    sync_dataset: Dataset,
    fallback_line: Union[int, str] = 2
) -> np.ndarray:
    """Get vsync times (when camstim is flipping display frame buffer) from a
    loaded session *.sync dataset.

    Parameters
    ----------
    sync_dataset : Dataset
        A loaded *.sync file for a session (contains events from
        different data streams logged on a global time basis)
    fallback_line : Union[int, str], optional
        The sync dataset line label to use if named line labels could not
        be found, by default 2.

        For more details about line labels see:
        https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH  # noqa: E501

    Returns
    -------
    np.ndarray
        An array of times (in seconds) when the display frame buffer
        finished being flipped.
    """

    # Look for vsync line in sync dataset line labels
    vsync_line: Union[int, str] = fallback_line
    for line in sync_dataset.line_labels:
        if line == 'vsync_stim':
            vsync_line = line
            break
        if line == 'stim_vsync':
            vsync_line = line
            break

    if vsync_line == fallback_line:
        logging.warning(
            f"Could not find 'vsync_stim' nor 'stim_vsync' line labels in "
            f"sync dataset ({sync_dataset.dfile.filename}). Defaulting to "
            f"using fallback line label index ({fallback_line}) which "
            f"is not guaranteed to be correct!"
        )

    falling_edges = sync_dataset.get_falling_edges(vsync_line, units='seconds')

    return falling_edges


def get_stim_starts_and_ends(
    sync_dataset: Dataset, fallback_line: Union[int, str] = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Get stimulus presentation start and end times from a loaded session
    *.sync datset.

    Parameters
    ----------
    sync_dataset : Dataset
        A loaded *.sync file for a session (contains events from
        different data streams logged on a global time basis)
    fallback_line : Union[int, str], optional
        The sync dataset line label to use if named line labels could not
        be found, by default 5.

        For more details about line labels see:
        https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH  # noqa: E501

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays containing
        (stimulus_start_times, stimulus_end_times) in seconds.
    """

    # Look for 'stim_running' line in sync dataset line labels
    stim_line: Union[int, str] = fallback_line
    for line in sync_dataset.line_labels:
        if line == 'stim_running':
            stim_line = line
            break
        if line == 'sweep':
            stim_line = line
            break

    if stim_line == fallback_line:
        logging.warning(
            f"Could not find 'stim_running' nor 'sweep' line labels in "
            f"sync dataset ({sync_dataset.dfile.filename}). Defaulting to "
            f"using fallback line label index ({fallback_line}) which "
            f"is not guaranteed to be correct!"
        )

    # 'stim_running'/'sweep' line is high while visual stimulus is being
    # displayed and low otherwise
    stim_starts = sync_dataset.get_rising_edges(stim_line, units='seconds')
    stim_ends = sync_dataset.get_falling_edges(stim_line, units='seconds')

    return stim_starts, stim_ends


def get_frame_offsets(
    sync_dataset: Dataset, frame_counts: List[int], tolerance: int = 0
) -> List[int]:
    """List of the inferred start frames for each of the stimuli

    Parameters
    ----------
    sync_dataset : Dataset
        Sync data from an ecephys session (a 'Dataset' object that has loaded
        a *.sync H5 file).
    frame_counts : List[int]
        List of expected frame counts (taken from behavior, mapping, and replay
        pkl files) for each stimuli category. The list of counts should be
        ordered by the actual display sequence of the stimuli categories.

        WARNING: The order of frame counts should match the actual order of
        presentations.
    tolerance : int
        Percent by which frame counts are allowed to deviate from,
        by default 0.

    Returns
    -------
    List[float]
        List of the inferred start frames for each stimuli category presented
        during session.
    """
    frame_count_arr = np.array(frame_counts)
    tolerance_pct = tolerance / 100.

    # Get vsyncs and stim_running signals from sync data
    vsync_times = get_vsyncs(sync_dataset)
    stim_starts, stim_ends = get_stim_starts_and_ends(sync_dataset)

    # Get vsync frame lengths and indices for all stimuli
    epoch_frame_counts = []
    epoch_start_frames = []
    for start, end in zip(stim_starts, stim_ends):
        # Inner expression returns a bool array where conditions are True
        # np.where evaluates bool array to return indices where bool array True
        epoch_frames = np.where((vsync_times > start) & (vsync_times < end))[0]
        epoch_frame_counts.append(len(epoch_frames))
        epoch_start_frames.append(epoch_frames[0])

    if len(epoch_frame_counts) == len(frame_count_arr):
        if not np.allclose(frame_count_arr, epoch_frame_counts):
            logging.warning(
                f"Number of frames derived from sync file "
                f"({epoch_frame_counts})for each epoch not matching up with "
                f"frame counts derived from pkl files ({frame_count_arr})!"
            )
        start_frames = epoch_start_frames
    elif len(epoch_frame_counts) > len(frame_count_arr):
        logging.warning(
            f"Number of stim presentations obtained from sync "
            f"({len(epoch_frame_counts)}) higher than number expected "
            f"({len(frame_count_arr)}). Inferring start frames."
        )

        start_frames = []
        for stim_idx, fc in enumerate(frame_count_arr):

            logging.info(f"Finding stim start for stim with index: {stim_idx}")
            # Get index of stimulus whose frame counts most closely match
            # the expected number of frames
            best_match = int(
                np.argmin([np.abs(efc - fc) for efc in epoch_frame_counts])
            )
            lower_tol = fc * (1 - tolerance_pct)
            upper_tol = fc * (1 + tolerance_pct)
            if lower_tol <= epoch_frame_counts[best_match] <= upper_tol:
                _ = epoch_frame_counts.pop(best_match)
                start_frame = epoch_start_frames.pop(best_match)
                start_frames.append(start_frame)
                logging.info(
                    f"Found stim start for stim with index ({stim_idx})"
                    f"at vsync ({start_frame})"
                )
            else:
                raise RuntimeError(
                    f"Could not find matching sync frames for stim: {stim_idx}"
                )
    else:
        raise RuntimeError(
            f"Do not know how to handle more pkl frame count entries "
            f"({frame_count_arr}) than sync derived epoch frame count "
            f"entries ({epoch_frame_counts})!"
        )

    return start_frames, stim_starts, stim_ends


def determine_behavior_stimulus_properties(
    raw_behavior_stimulus_df: pd.DataFrame,
    reward_times: np.ndarray
) -> dict:
    """Given a raw behavior stimulus dataframe, determine properties of
    the stimuli presented.

    Parameters
    ----------
    raw_behavior_stimulus_df : pd.DataFrame
        A visual behavior stimulus presentation dataframe obtained from
        `get_stimulus_presentations()`.
    reward_times : np.ndarray
        An array of times (in seconds) when subject was rewarded.

    Returns
    -------
    dict
        A dictionary with the following key, values:
        "flashes_since_change": int array counting number of stimulus
            presentations since an image change occurred. It gets reset to 0
            when stimuli has changed and increments for non-change stimuli
            presentations. 'Omitted' stimuli will not increment the counter.
        "is_change": bool array that is True, when stimulus presentation
            was a 'change'd image. False otherwise.
        "rewarded": bool array that is True, when subject was rewarded for
            the stimulus presentation. False otherwise.
    """

    stim_df = raw_behavior_stimulus_df
    # Iterate through raw behavior stimulus df and determine:
    # 1) The number of presentations (flashes) since an image change
    # 2) Whether the image presentation is 'change' or not
    # 3) Whether the mouse was rewarded for the image presentation
    is_change = np.zeros(len(stim_df), dtype=bool)
    rewarded = np.zeros(len(stim_df), dtype=bool)
    flashes_since_change = np.zeros(len(stim_df), dtype=int)
    current_image = None
    for index, row in stim_df.iterrows():
        if (row['image_name'] == 'omitted') or (row['omitted']):
            # An omitted stimulus shouldn't increment flashes_since_change
            flashes_since_change[index] = flashes_since_change[index-1]
        else:
            if row['image_name'] != current_image:
                is_change[index] = True
                flashes_since_change[index] = 0
                current_image = row['image_name']
                if np.min(np.abs(row['start_time'] - reward_times)) < 1:
                    rewarded[index] = True
            else:
                flashes_since_change[index] = flashes_since_change[index-1] + 1

    # First encountered change should not be considered a 'change' (in stimuli)
    is_change[np.where(is_change)[0][0]] = False

    return {
        "flashes_since_change": flashes_since_change,
        "is_change": is_change,
        "rewarded": rewarded
    }


def estimate_frame_duration(pd_times, cycle=60):
    return trimmed_stats(np.diff(pd_times))[0] / cycle


def generate_behavior_stim_table(
    behavior_pkl: BehaviorPickleFile,
    sync_dataset: Dataset,
    stim_start: float,
    stim_end: float,
    frame_offset: int = 0,
    block_offset: int = 0,
    photodiode_cycle: int = 60
) -> pd.DataFrame:
    """Generate a stimulus table for the behavior portion of a visual behavior
    neuropixels session.

    Parameters
    ----------
    behavior_pkl : BehaviorPickleFile
        A BehaviorPickleFile object, that allows easier access to key
        behavior pickle file data and metadata.
    sync_dataset : Dataset
        A sync Datset object that allows pythonic access of *.sync file data
        containing global timing information for events and presented stimuli.
    frame_offset : int, optional
        Used to give correct 'start_frame' and 'end_frame' values when
        combining with DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 0
    block_offset : int, optional
        Used to give correct 'stimulus_block' values when combining with
        DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 0

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing only presentations pertaining to the
        visual behavior portion of a VBN session. Has the following columns:
            stimulus_presentations_id (index): Unique identifier for a
                stimulus presentation
            duration: Duration (in seconds) of a stimulus presentation.
            end_frame: The last frame that the stimulus was present on screen.
            image_name: The name of the image.
            omitted: Whether the presentation was omitted or not for the
                specific presentation.
            orientation: The orientation of the image.
            start_frame: The first frame that the stimulus was present on
                screen.
            start_time: The time in seconds when stimulus first appeared on
                screen.
            stop_time: The last time in seconds that stimulus appeared on
                screen.
            stimulus_block: A full VBN session can be broken into several
                blocks, this column denotes the block index.
            stimulus_name: The name of the stimulus set which images were
                drawn from.
            is_change: Whether a presentation was a 'changed' image (True)
                or a repeated presentation of an image (False).
            rewarded: Whether subject was rewarded during presentation.
            flashes_since_change: The number of image presentations since
                a different image was presented.
            active: Whether the stimulus presentation was during a portion of
                the session where the subject was actively performing a
                behavior task.
    """
    image_set = behavior_pkl.image_set
    num_frames = behavior_pkl.num_frames
    reward_frames = behavior_pkl.reward_frames

    frame_timestamps = get_vsyncs(sync_dataset)

    partitioned_vsync_times = frame_timestamps[
        frame_offset:frame_offset + num_frames
    ]

    partitioned_photodiode_times = partition_photodiode_times(
        sync_dataset,
        partitioned_vsync_times,
        stim_start, stim_end
    )

    expected_vsync_duration = estimate_frame_duration(
        partitioned_photodiode_times, cycle=photodiode_cycle
    )

    frame_timestamps = compute_vbn_block_frame_times(
        partitioned_vsync_times,
        partitioned_photodiode_times,
        expected_vsync_duration,
        photodiode_cycle
    )

    # sync_dataset.plot_monitor_lag(
    #     partitioned_photodiode_times,
    #     frame_timestamps,
    #     photodiode_cycle
    # )

    reward_times = frame_timestamps[reward_frames.astype(int)]
    epoch_timestamps = frame_timestamps[frame_offset:frame_offset + num_frames]

    stim_df = get_stimulus_presentations(behavior_pkl.data, epoch_timestamps)
    stim_df['stimulus_block'] = block_offset
    stim_df['stimulus_name'] = image_set

    stim_properties = determine_behavior_stimulus_properties(
        raw_behavior_stimulus_df=stim_df, reward_times=reward_times
    )
    stim_df["is_change"] = stim_properties["is_change"]
    stim_df["rewarded"] = stim_properties["rewarded"]
    stim_df["flashes_since_change"] = stim_properties["flashes_since_change"]
    stim_df['active'] = True

    # Fill in 'end_frame' for omitted stimuli
    median_stim_frame_duration = np.nanmedian(
        stim_df["end_frame"] - stim_df["start_frame"]
    )
    omitted_end_frames = (
        stim_df[stim_df['omitted']]['start_frame'] + median_stim_frame_duration
    )
    stim_df.loc[stim_df['omitted'], 'end_frame'] = omitted_end_frames

    # Now fill in 'stop_time's for omitted stimuli
    omitted_end_times = (
        epoch_timestamps[stim_df[stim_df['omitted']]['end_frame'].astype(int)]
    )
    stim_df.loc[stim_df['omitted'], 'stop_time'] = omitted_end_times

    # Drop index column
    stim_df.drop(columns=["index"], inplace=True)

    column_dtypes = {
        "duration": float,
        "end_frame": int,
        "image_name": object,
        "omitted": bool,
        "orientation": float,
        "start_frame": int,
        "start_time": float,
        "stop_time": float,
        "stimulus_block": int,
        "stimulus_name": object,
        "is_change": bool,
        "rewarded": bool,
        "flashes_since_change": int,
        "active": bool
    }

    return stim_df.astype(column_dtypes)


def check_behavior_and_replay_pkl_match(
    replay_pkl: ReplayPickleFile,
    behavior_stim_table: pd.DataFrame
):
    """Check that the number of stimulus presentations and order/identity
    of stimulus presentations match between behavior.pkl and replay.pkl files.

    Parameters
    ----------
    replay_pkl : ReplayPickleFile
        A ReplayPickleFile object, that allows easier access to key
        replay pickle file data and metadata.
    behavior_stim_table : pd.DataFrame
        A behavior stimulus table created by the generate_behavior_stim_table()
        function.
    """
    ims = replay_pkl.image_presentations

    # Check that replay pkl matches behavior
    im_ons = []
    im_offs = []
    im_names = []
    for ind, im in enumerate(ims):
        if ind == 0:
            continue
        elif ind < len(ims) - 1:
            if ims[ind - 1] is None and ims[ind] is not None:
                im_ons.append(ind)
                im_names.append(im)
            elif ims[ind] is not None and ims[ind + 1] is None:
                im_offs.append(ind)

    inter_flash_interval = np.diff(im_ons)
    putative_omitted = np.where(inter_flash_interval > 70)[0]

    # Must specify dtype=object otherwise numpy will try to over-optimize
    # by using a length limited character type (e.g. <U1, <U2, etc...)
    # This causes problems when trying to insert "omitted" as it will may be
    # accidentally truncated to "o", "om", etc...
    im_names_arr = np.array(im_names, dtype=object)
    im_names_with_omitted = np.insert(
        im_names_arr, putative_omitted + 1, 'omitted'
    )

    # Handle omitted flash edge cases
    behavior_df_image_names = behavior_stim_table['image_name']

    # Check if the first flash was omitted
    first_flash_omitted = behavior_df_image_names.iloc[0] == 'omitted'
    if first_flash_omitted:
        im_names_with_omitted = np.insert(im_names_with_omitted, 0, 'omitted')

    # Check if last flash was omitted
    last_flash_omitted = behavior_df_image_names.iloc[-1] == 'omitted'
    if last_flash_omitted:
        im_names_with_omitted = np.insert(
            im_names_with_omitted,
            len(im_names_with_omitted),
            'omitted'
        )

    # Verify that the image list for replay is identical to behavior
    expected_num_images = len(behavior_stim_table['image_name'])
    obtained_num_images = len(im_names_with_omitted)
    assert expected_num_images == obtained_num_images, (
        f"Number of expected images ({expected_num_images}) presented in "
        f"replay pickle does not match up with actual number presented in "
        f"replay pickle ({obtained_num_images})!"
    )
    assert all(behavior_stim_table['image_name'] == im_names_with_omitted), (
        f"The identity and/or order of expected images presentations in the "
        f"replay pickle ({behavior_stim_table['image_name']}) does not match "
        f"the actual identity and/or order of image presentations in the "
        f"replay pickle ({obtained_num_images}!"
    )


def trim_border_pulses(pd_times, vs_times, frame_interval=1/60, num_frames=3):
    pd_times = np.array(pd_times)
    pd_times = pd_times[np.logical_and(
        pd_times >= vs_times[0],
        pd_times <= vs_times[-1] + num_frames * frame_interval
    )]

    return pd_times


def correct_on_off_effects(pd_times):
    """
    Notes
    -----
    This cannot (without additional info) determine whether an assymmetric
    offset is odd-long or even-long.
    """

    pd_diff = np.diff(pd_times)
    odd_diff_mean, odd_diff_std = trimmed_stats(pd_diff[1::2])
    even_diff_mean, even_diff_std = trimmed_stats(pd_diff[0::2])

    half_diff = np.diff(pd_times[0::2])
    full_period_mean, full_period_std = trimmed_stats(half_diff)
    half_period_mean = full_period_mean / 2

    odd_offset = odd_diff_mean - half_period_mean
    even_offset = even_diff_mean - half_period_mean

    pd_times[::2] -= odd_offset / 2
    pd_times[1::2] -= even_offset / 2

    return pd_times


def fix_unexpected_edges(pd_times, ndevs=10, cycle=60, max_frame_offset=4):
    pd_times = np.array(pd_times)
    expected_duration_mask = flag_unexpected_edges(pd_times, ndevs=ndevs)
    diff_mean, diff_std = trimmed_stats(np.diff(pd_times))
    frame_interval = diff_mean / cycle

    bad_edges = np.where(expected_duration_mask == 0)[0]
    bad_blocks = np.sort(np.unique(np.concatenate([
        [0],
        np.where(np.diff(bad_edges) > 1)[0] + 1,
        [len(bad_edges)]
    ])))

    output_edges = []
    for low, high in zip(bad_blocks[:-1], bad_blocks[1:]):
        current_bad_edge_indices = bad_edges[low: high-1]
        current_bad_edges = pd_times[current_bad_edge_indices]
        low_bound = pd_times[current_bad_edge_indices[0]]
        high_bound = pd_times[current_bad_edge_indices[-1] + 1]

        edges_missing = int(np.around((high_bound - low_bound) / diff_mean))
        expected = np.linspace(low_bound, high_bound, edges_missing + 1)

        distances = distance.cdist(
            current_bad_edges[:, None], expected[:, None]
        )
        distances = np.around(distances / frame_interval).astype(int)

        min_offsets = np.amin(distances, axis=0)
        min_offset_indices = np.argmin(distances, axis=0)
        output_edges = np.concatenate([
            output_edges,
            expected[min_offsets > max_frame_offset],
            current_bad_edges[
                min_offset_indices[min_offsets <= max_frame_offset]
            ]
        ])

    return np.sort(
        np.concatenate([output_edges, pd_times[expected_duration_mask > 0]])
    )


def trimmed_stats(data, pctiles=(10, 90)):
    low = np.percentile(data, pctiles[0])
    high = np.percentile(data, pctiles[1])

    trimmed = data[np.logical_and(
        data <= high,
        data >= low
    )]

    return np.mean(trimmed), np.std(trimmed)


def flag_unexpected_edges(pd_times, ndevs=10):
    pd_diff = np.diff(pd_times)
    diff_mean, diff_std = trimmed_stats(pd_diff)

    expected_duration_mask = np.ones(pd_diff.size)
    expected_duration_mask[np.logical_or(
        pd_diff < diff_mean - ndevs * diff_std,
        pd_diff > diff_mean + ndevs * diff_std
    )] = 0
    expected_duration_mask[1:] = np.logical_and(
        expected_duration_mask[:-1], expected_duration_mask[1:]
    )
    expected_duration_mask = np.concatenate(
        [expected_duration_mask, [expected_duration_mask[-1]]]
    )

    return expected_duration_mask


def partition_photodiode_times(
        sync_dataset,
        partitioned_vsync_times,
        stim_start, stim_end
):

    photodiode_times = sync_dataset.get_edges('all', [4], units='seconds')

    photodiode_times = photodiode_times[
        (photodiode_times >= stim_start) & (photodiode_times <= stim_end)
    ]

    photodiode_times = trim_border_pulses(
        photodiode_times, partitioned_vsync_times
    )

    photodiode_times = correct_on_off_effects(
        photodiode_times
    )

    photodiode_times = fix_unexpected_edges(
        photodiode_times
    )

    return photodiode_times


def set_corrected_times(
    corrected_frame_times,
    vsync_slice,
    start_time,
    corrected_relevant_vsyncs
):

    if vsync_slice.stop < len(corrected_frame_times):
        corrected_frame_times[
            vsync_slice
        ] = start_time + corrected_relevant_vsyncs
    else:
        # TODO - is this correct? The lengths and shapes do not always line up

        corrected_frame_times[
            vsync_slice.start:
            len(corrected_frame_times)
        ] = start_time + corrected_relevant_vsyncs[
            0:
            len(corrected_frame_times) - vsync_slice.start
        ]


def compute_vbn_block_frame_times(
    partitioned_vsync_times: np.ndarray,
    partitioned_photodiode_times: np.ndarray,
    expected_vsync_duration: float,
    num_vsyncs_per_diode_toggle: int = 60
) -> np.ndarray:

    num_vsyncs = len(partitioned_vsync_times)
    corrected_frame_times = np.zeros(num_vsyncs, dtype=float)
    vsync_durations = np.diff(partitioned_vsync_times)

    cycle = num_vsyncs_per_diode_toggle

    pd_intervals = zip(
        partitioned_photodiode_times[:-1],
        partitioned_photodiode_times[1:]
    )

    for pd_interval_ind, (start_time, end_time) in enumerate(pd_intervals):

        # Get duration of the current on->off/off->on photodiode interval
        pd_interval_duration = end_time - start_time

        # Get only vsync event times and vsync interval durations
        # associated with current photodiode on/off interval
        vsync_slice = slice(
            pd_interval_ind * num_vsyncs_per_diode_toggle,
            (pd_interval_ind + 1) * num_vsyncs_per_diode_toggle
        )
        relevant_vsyncs = partitioned_vsync_times[vsync_slice]
        relevant_vsync_durations = vsync_durations[vsync_slice]

        # Determine number of "long" vsyncs
        # (vsyncs that are double the duration of normal vsyncs)
        expected_pd_interval_duration = (
            num_vsyncs_per_diode_toggle * expected_vsync_duration
        )

        excess_pd_interval_duration = (
            np.sum(relevant_vsync_durations) - expected_pd_interval_duration
        )

        # We should only be long by multiples of vsync duration
        num_long_vsyncs = int(
            np.around(
                excess_pd_interval_duration / expected_vsync_duration
            )
        )

        # Determine total delay (sum of all sources of delay)
        # in units of 'vsyncs'
        # Total delay changes can only happen in whole 'vsyncs',
        # never in fractions of 'vsyncs' (hence rounding)
        total_delay = (
            int(
                np.around(
                    (pd_interval_duration / expected_vsync_duration)
                )
            ) - num_vsyncs_per_diode_toggle
        )

        # If our total_delay is more than we would expect from just long vsyncs
        # then extra frame to monitor delay occurred
        extra_frame_to_monitor_delay = 0

        if total_delay > num_long_vsyncs:
            print(
                """Extra delay between frame time
                 and monitor display time detected"""
            )

            # Delay attributed to hardware/software factors that delay time
            # to monitor display (in units of 'vsyncs') must then be:
            extra_frame_to_monitor_delay = total_delay - num_long_vsyncs

        # Number of actual frames/vsyncs that would fit
        # in a photodiode switch interval
        local_expected_vsync_duration = (
            pd_interval_duration / (num_vsyncs_per_diode_toggle + total_delay)
        )

        if total_delay > 0:
            # Correct for variability in vsync times
            variance_reduced_frame_diffs = (
                np.round(
                    np.diff(relevant_vsyncs) / local_expected_vsync_duration
                )
            )

            # NJM - Want first vsync to happen at diode transition
            # NJM - Is the 0th vsync happening before or after first
            # photodiode transition? There could be 1-off error (double check)
            # Will need to check empirically when implementing
            result = (
                variance_reduced_frame_diffs * local_expected_vsync_duration
            )

            corrected_relevant_vsyncs = np.insert(
                np.cumsum(result),
                0,
                0
            )

            # Then correct for extra_frame_to_monitor_delay if there was any
            # Assume that if there was a change
            # in monitor lag, it was after the long frame
            longest_ind = np.argmax(relevant_vsync_durations) + 1
            corrected_relevant_vsyncs[longest_ind:] += (
                extra_frame_to_monitor_delay * local_expected_vsync_duration
            )

            set_corrected_times(
                corrected_frame_times,
                vsync_slice,
                start_time,
                corrected_relevant_vsyncs
            )

        else:

            frame_diffs = np.ones(cycle-1)
            corrected_relevant_vsyncs = np.insert(
                np.cumsum(frame_diffs * local_expected_vsync_duration),
                0,
                0
            )

            set_corrected_times(
                corrected_frame_times,
                vsync_slice,
                start_time,
                corrected_relevant_vsyncs
            )

    # Now deal with leftover vsyncs that occur after the last diode transition
    # Just take the global frame duration for these
    leftover_vsyncs_start_ind = (
        len(partitioned_vsync_times)
        - np.mod(len(partitioned_vsync_times), num_vsyncs_per_diode_toggle)
    )
    relevant_vsyncs = partitioned_vsync_times[leftover_vsyncs_start_ind:]
    frame_diffs = np.round(
        np.diff(relevant_vsyncs) / expected_vsync_duration
    )

    corrected_relevant_vsyncs = np.insert(
        np.cumsum(frame_diffs * expected_vsync_duration),
        0,
        0
    )

    corrected_frame_times[leftover_vsyncs_start_ind:] = (
        partitioned_photodiode_times[-1] + corrected_relevant_vsyncs
    )

    return corrected_frame_times


def generate_replay_stim_table(
    replay_pkl: ReplayPickleFile,
    sync_dataset: Dataset,
    stim_start: float,
    stim_end: float,
    behavior_stim_table: pd.DataFrame,
    block_offset: int = 5,
    frame_offset: int = 0,
    photodiode_cycle: int = 60
) -> pd.DataFrame:
    """Generate a stimulus table for the replay portion of a visual behavior
    neuropixels session.

    Parameters
    ----------
    replay_pkl : ReplayPickleFile
        A ReplayPickleFile object, that allows easier access to key
        replay pickle file data and metadata.
    sync_dataset : Dataset
        A sync Datset object that allows pythonic access of *.sync file data
        containing global timing information for events and presented stimuli.
    behavior_stim_table : pd.DataFrame
        A behavior stimulus table created by the generate_behavior_stim_table()
        function.
    block_offset : int, optional
        Used to give correct 'stimulus_block' values when combining with
        DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 5
    frame_offset : int, optional
        Used to give correct 'start_frame' and 'end_frame' values when
        combining with DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 0

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the same columns as those created by the
        generate_behavior_stim_table() function.
    """

    num_frames = replay_pkl.num_frames
    frame_timestamps = get_vsyncs(sync_dataset)

    partitioned_vsync_times = (
        frame_timestamps[frame_offset:frame_offset + num_frames]
    )

    partitioned_photodiode_times = partition_photodiode_times(
        sync_dataset,
        partitioned_vsync_times,
        stim_start,
        stim_end
    )

    expected_vsync_duration = estimate_frame_duration(
        partitioned_photodiode_times,
        cycle=photodiode_cycle
    )

    frame_timestamps = compute_vbn_block_frame_times(
        partitioned_vsync_times,
        partitioned_photodiode_times,
        expected_vsync_duration,
        photodiode_cycle
    )

    # sync_dataset.plot_monitor_lag(
    #     partitioned_photodiode_times,
    #     frame_timestamps,
    #     photodiode_cycle
    # )

    check_behavior_and_replay_pkl_match(
        replay_pkl=replay_pkl, behavior_stim_table=behavior_stim_table
    )

    # If replay pkl data and behavior pkl data match, use the existing
    # behavior stim table but adjust the times/frames
    stim_table = behavior_stim_table.copy(deep=True)
    stim_table['stimulus_block'] = block_offset
    stim_table['start_time'] = frame_timestamps[stim_table['start_frame']]
    stim_table['stop_time'] = frame_timestamps[stim_table['end_frame']]
    stim_table['duration'] = stim_table['stop_time'] - stim_table['start_time']
    stim_table['start_frame'] = stim_table['start_frame'] + frame_offset
    stim_table['end_frame'] = stim_table['end_frame'] + frame_offset
    stim_table['active'] = False

    column_dtypes = {
        "duration": float,
        "end_frame": int,
        "start_frame": int,
        "start_time": float,
        "stop_time": float,
        "stimulus_block": int,
        "active": bool
    }

    return stim_table.astype(column_dtypes)


def generate_mapping_stim_table(
    mapping_pkl: CamStimOnePickleStimFile,
    sync_dataset: Dataset,
    stim_start: float,
    stim_end: float,
    frame_offset: int = 0,
    block_offset: int = 1,
    photodiode_cycle: int = 60
) -> pd.DataFrame:
    """Generate a stimulus table for the mapping portion of a visual behavior
    neuropixels session.

    Parameters
    ----------
    mapping_pkl : CamStimOnePickleStimFile
        A CamStimOnePickleStimFile object, that allows easier access to key
        mapping pickle file data and metadata.
    sync_dataset : Dataset
        A sync Datset object that allows pythonic access of *.sync file data
        containing global timing information for events and presented stimuli.
    frame_offset : int, optional
        Used to give correct 'start_frame' and 'end_frame' values when
        combining with DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 0
    block_offset : int, optional
        Used to give correct 'stimulus_block' values when combining with
        DataFrames from other portions of a VBN session
        (e.g. behavior, mapping, replay), by default 1

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing only presentations pertaining to the
        mapping portion of a VBN session. Has the following columns:
            duration: Duration (in seconds) of a stimulus presentation.
            end_frame: The last frame that the stimulus was present on screen.
            omitted: Whether the presentation was omitted or not for the
                specific presentation.
            orientation: The orientation of the image.
            start_frame: The first frame that the stimulus was present on
                screen.
            start_time: The time in seconds when stimulus first appeared on
                screen.
            stop_time: The last time in seconds that stimulus appeared on
                screen.
            stimulus_block: A full VBN session can be broken into several
                blocks, this column denotes the block index.
            stimulus_name: The name of the stimulus set which images were
                drawn from.
            active: Whether the stimulus presentation was during a portion of
                the session where the subject was actively performing a
                behavior task.
            temporal_frequency: Temporal frequency of stimulus in Hz.
            spatial_frequency: Spatial frequency of stimulus in units of
                cycles per degree.
            orientation: Orientation of stimulus, 0 indicates vertical
                orientation, 90 indicates clockwise 90 degree rotation.
            contrast:
                Contrast of stimulus defined as Michelson contrast.
            position_x: Horizontal position of stimulus on screen.
            position_y: Vertical position of stimulus on screen.
            stimulus_index: Index of mapping stimuli type. This column must
                be kept in order to maintain compatibility with CSD
                calculation module.
            color:
                "Color" of flash stimuli. 1 indicated white flash, -1
                indicates black (background is normally gray).
    """

    def seconds_to_frames(
        seconds: List[Union[int, float]]
    ) -> List[Union[int, float]]:
        offset_times = np.array(seconds) + mapping_pkl.pre_blank_sec
        return offset_times * mapping_pkl.frames_per_second

    stim_tabler = partial(
        build_stimuluswise_table, seconds_to_frames=seconds_to_frames
    )
    stim_df = create_stim_table(
        mapping_pkl.stimuli, stim_tabler, make_spontaneous_activity_tables
    )

    frame_timestamps = get_vsyncs(sync_dataset)

    num_frames = mapping_pkl.num_frames

    partitioned_vsync_times = (
        frame_timestamps[frame_offset:frame_offset + num_frames]
    )

    partitioned_photodiode_times = partition_photodiode_times(
        sync_dataset,
        partitioned_vsync_times,
        stim_start,
        stim_end
    )

    expected_vsync_duration = estimate_frame_duration(
        partitioned_photodiode_times,
        cycle=photodiode_cycle
    )

    frame_timestamps = compute_vbn_block_frame_times(
        partitioned_vsync_times,
        partitioned_photodiode_times,
        expected_vsync_duration,
        photodiode_cycle
    )

    # sync_dataset.plot_monitor_lag(
    #     partitioned_photodiode_times,
    #     frame_timestamps,
    #     photodiode_cycle
    # )

    stim_df = stim_df.rename(
        columns={
            "Start": "start_frame",
            "End": "end_frame",
            "TF": "temporal_frequency",
            "SF": "spatial_frequency",
            "Ori": "orientation",
            "Contrast": "contrast",
            "Pos_x": "position_x",
            "Pos_y": "position_y",
            "Color": "color"
        }
    )

    # Fill in "stimulus_name" for 'spontaneous' stimuli
    # Characterized by NaN for "stimulus_name" and "stimulus_block"
    spont_rows = (
        stim_df["stimulus_name"].isnull() & stim_df["stimulus_block"].isnull()
    )
    stim_df.loc[spont_rows, "stimulus_name"] = 'spontaneous'

    # Fill in "stimulus_block" column values
    # "stimulus_block" column should start at "block_offset" value and
    # increment by one every time the "stimulus_name" changes
    shifted_stim_names = stim_df["stimulus_name"].shift(
        1, fill_value=stim_df["stimulus_name"].iloc[0]
    )
    is_change = shifted_stim_names != stim_df["stimulus_name"]
    change_indices = np.where(is_change)[0]
    change_indices = np.append(change_indices, len(stim_df["stimulus_name"]))

    stimulus_block = []
    prev_change_idx = 0
    for change_idx in change_indices:
        stimulus_block.extend([block_offset] * (change_idx - prev_change_idx))
        prev_change_idx = change_idx
        block_offset += 1

    stim_df["stimulus_block"] = stimulus_block

    # Now fill in other columns
    stim_df['start_frame'] = stim_df['start_frame'].astype(int) + frame_offset
    stim_df['end_frame'] = stim_df['end_frame'].astype(int) + frame_offset
    stim_df['start_time'] = (
        frame_timestamps[stim_df['start_frame'] - frame_offset]
    )

    stim_df['stop_time'] = (
        frame_timestamps[stim_df['end_frame'] - frame_offset]
    )

    stim_df['duration'] = stim_df['stop_time'] - stim_df['start_time']
    stim_df['active'] = False

    column_dtypes = {
        "duration": float,
        "end_frame": int,
        "orientation": float,
        "start_frame": int,
        "start_time": float,
        "stop_time": float,
        "stimulus_block": int,
        "stimulus_name": object,
        "active": bool,
        "temporal_frequency": float,
        "spatial_frequency": float,
        "contrast": float,
        "position_x": float,
        "position_y": float,
        "color": float
    }

    return stim_df.astype(column_dtypes)


def create_vbn_stimulus_table(
    sync_dataset: Dataset,
    behavior_pkl: BehaviorPickleFile,
    mapping_pkl: CamStimOnePickleStimFile,
    replay_pkl: ReplayPickleFile
) -> pd.DataFrame:
    """Create a stimulus table that encompasses all 'blocks' of stimuli
    presented during a visual behavior neuropixels session
    (behavior stimuli, gabor stimuli, spontaneous stimlus,
    full field flash stimuli, replay stimuli).

    Parameters
    ----------
    sync_dataset : Dataset
        A sync Datset object that allows pythonic access of *.sync file data
        containing global timing information for events and presented stimuli.
    behavior_pkl : BehaviorPickleFile
        A BehaviorPickleFile object, that allows easier access to key
        behavior pickle file data and metadata.
    mapping_pkl : CamStimOnePickleStimFile
        A CamStimOnePickleStimFile object, that allows easier access to key
        mapping pickle file data and metadata.
    replay_pkl : ReplayPickleFile
        A ReplayPickleFile object, that allows easier access to key
        replay pickle file data and metadata.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe. For more details about columns and sources
        of data for each column please consult:
        http://confluence.corp.alleninstitute.org/display/IT/Visual+Behavior+Neuropixels+-+Stimulus+Table+Design+Document
    """

    frame_counts = [
        pkl.num_frames for pkl in (behavior_pkl, mapping_pkl, replay_pkl)
    ]

    frame_offsets, stim_starts, stim_ends = (
        get_frame_offsets(sync_dataset, frame_counts)
    )

    # Generate stim tables for the 3 different stimulus pkl types
    behavior_df = generate_behavior_stim_table(
        behavior_pkl,
        sync_dataset,
        stim_starts[0],
        stim_ends[0],
        frame_offset=frame_offsets[0]
    )
    mapping_df = generate_mapping_stim_table(
        mapping_pkl,
        sync_dataset,
        stim_starts[1],
        stim_ends[1],
        frame_offset=frame_offsets[1]
    )
    replay_df = generate_replay_stim_table(
        replay_pkl,
        sync_dataset,
        stim_starts[2],
        stim_ends[2],
        behavior_df,
        frame_offset=frame_offsets[2]
    )

    # Rearrange columns to make a bit more readable;
    key_col_order = [
        "stimulus_block", "active", "stimulus_name", "start_time",
        "stop_time", "duration", "start_frame", "end_frame"
    ]
    # Rest of the cols can just be in alphabetical order
    other_col_order = sorted(
        [c for c in behavior_df.columns if c not in key_col_order]
    )
    behavior_df = behavior_df[key_col_order + other_col_order]

    full_stim_df = pd.concat([behavior_df, mapping_df, replay_df], sort=False)
    full_stim_df.reset_index(drop=True, inplace=True)

    return full_stim_df
