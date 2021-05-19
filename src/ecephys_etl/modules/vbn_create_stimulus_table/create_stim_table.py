from functools import partial
from typing import List, Tuple, Union
import logging

import numpy as np
import pandas as pd

from ecephys_etl.data_extractors.sync_dataset import Dataset
from ecephys_etl.data_extractors.stim_file import CamStimOnePickleStimFile
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
            f"Could not find 'stim_runing' nor 'sweep' line labels in "
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

    if len(epoch_frame_counts) > len(frame_count_arr):
        logging.warning(
            f"Number of stim presentations obtains from sync "
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
        start_frames = epoch_start_frames

    return start_frames


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
    current_image = stim_df.iloc[0]['stimulus_name']
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


def generate_behavior_stim_table(
    pkl_data: dict,
    sync_dataset: Dataset,
    frame_offset: int = 0,
    block_offset: int = 0
) -> pd.DataFrame:

    pkl_behavior = pkl_data['items']['behavior']
    image_set = pkl_behavior['params']['stimulus']['params']['image_set']
    image_set = image_set.split('/')[-1].split('.')[0]
    num_frames = pkl_behavior['intervalsms'].size + 1
    reward_frames = pkl_behavior['rewards'][0]['reward_times'][:, 1]

    frame_timestamps = get_vsyncs(sync_dataset)
    reward_times = frame_timestamps[reward_frames.astype(int)]
    epoch_timestamps = frame_timestamps[frame_offset:frame_offset + num_frames]

    stim_df = get_stimulus_presentations(pkl_data, epoch_timestamps)
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
    stim_df.loc[stim_df['omitted'], 'End'] = omitted_end_times
    stim_df['common_name'] = 'behavior'

    return stim_df


def check_behavior_and_replay_pkl_match(
    replay_pkl_data: dict,
    behavior_stim_table: pd.DataFrame
):
    ims = replay_pkl_data['stimuli'][0]['sweep_params']['ReplaceImage'][0]
    im_names = np.unique([img for img in ims if img is not None])

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
    im_names_with_omitted = np.insert(
        im_names, putative_omitted + 1, 'omitted'
    )

    # Handle omitted flash edge cases ###
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
    assert all(behavior_stim_table['image_name'] == im_names_with_omitted)


def generate_replay_stim_table(
    pkl_data: dict,
    sync_dataset: Dataset,
    behavior_stim_table: pd.DataFrame,
    block_offset: int = 3,
    frame_offset: int = 0
) -> pd.DataFrame:

    num_frames = pkl_data['intervalsms'].size + 1

    frame_timestamps = get_vsyncs(sync_dataset)
    frame_timestamps = frame_timestamps[frame_offset:frame_offset + num_frames]

    check_behavior_and_replay_pkl_match(
        replay_pkl_data=pkl_data, behavior_stim_table=behavior_stim_table
    )

    # If replay pkl data and behavior pkl data match, use the existing
    # behavior stim table but adjust the times/frames
    stim_table = behavior_stim_table.copy(deep=True)
    stim_table['stimulus_block'] = block_offset
    stim_table['start_time'] = frame_timestamps[stim_table['start_frame']]
    stop_times = frame_timestamps[stim_table['end_frame'].dropna().astype(int)]
    stim_table.loc[:, 'stop_time'] = stop_times
    stim_table['start_frame'] = stim_table['start_frame'] + frame_offset
    stim_table.loc[:, 'end_frame'] = stim_table['end_frame'] + frame_offset
    stim_table['active'] = False
    stim_table['common_name'] = 'replay'

    return stim_table


def generate_mapping_stim_table(
    mapping_pkl_data: dict,
    sync_dataset: Dataset,
    block_offset: int = 1,
    frame_offset: int = 0
) -> pd.DataFrame:

    stim_file = CamStimOnePickleStimFile(data=mapping_pkl_data)

    def seconds_to_frames(
        seconds: List[Union[int, float]]
    ) -> List[Union[int, float]]:
        offset_times = np.array(seconds) + stim_file.pre_blank_sec
        return offset_times * stim_file.frames_per_second

    stim_tabler = partial(
        build_stimuluswise_table, seconds_to_frames=seconds_to_frames
    )
    stim_df = create_stim_table(
        stim_file.stimuli, stim_tabler, make_spontaneous_activity_tables
    )

    frame_timestamps = get_vsyncs(sync_dataset)

    stim_df = stim_df.rename(
        columns={
            "Start": "start_frame",
            "End": "end_frame"
        }
    )

    start_frames = np.array(stim_df['start_frame']).astype(int) + frame_offset
    stim_df['start_frame'] = start_frames
    end_frames = np.array(stim_df['end_frame']).astype(int) + frame_offset
    stim_df['end_frame'] = end_frames
    stim_df['start_time'] = frame_timestamps[stim_df['start_frame']]
    stim_df['stop_time'] = frame_timestamps[stim_df['end_frame']]
    stim_df['stimulus_block'] = stim_df['stimulus_block'] + block_offset
    stim_df['active'] = False
    stim_df['common_name'] = 'mapping'

    return stim_df


def create_vbn_stimulus_table(
    sync_dataset: Dataset,
    behavior_data: dict,
    mapping_data: dict,
    replay_data: dict
) -> pd.DataFrame:

    frame_counts = []
    for data in [behavior_data, mapping_data, replay_data]:
        if "intervalsms" in data:
            total_frames = len(data["intervalsms"]) + 1
        else:
            total_frames = len(data["items"]["behavior"]["intervalsms"]) + 1
        frame_counts.append(total_frames)

    frame_offsets = get_frame_offsets(sync_dataset, frame_counts)

    # Generate stim tables for the 3 different stimulus pkl types
    behavior_df = generate_behavior_stim_table(
        behavior_data, sync_dataset, frame_offset=frame_offsets[0]
    )
    mapping_df = generate_mapping_stim_table(
        mapping_data, sync_dataset, frame_offset=frame_offsets[1]
    )
    replay_df = generate_replay_stim_table(
        replay_data, sync_dataset, behavior_df,
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
    stim_durations = full_stim_df['stop_time'] - full_stim_df['start_time']
    full_stim_df.loc[:, 'duration'] = stim_durations
    full_stim_df.loc[full_stim_df['stimulus_name'].isnull(), 'stimulus_name'] = 'spontaneous'  # noqa: E501
    full_stim_df['presentation_index'] = np.arange(len(full_stim_df))

    return full_stim_df
