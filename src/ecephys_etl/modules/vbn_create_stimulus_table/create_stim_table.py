from functools import partial
from typing import List, Tuple, Union
import logging

import numpy as np
import pandas as pd

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


def generate_behavior_stim_table(
    behavior_pkl: BehaviorPickleFile,
    sync_dataset: Dataset,
    frame_offset: int = 0,
    block_offset: int = 0
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


def generate_replay_stim_table(
    replay_pkl: ReplayPickleFile,
    sync_dataset: Dataset,
    behavior_stim_table: pd.DataFrame,
    block_offset: int = 4,
    frame_offset: int = 0
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
        (e.g. behavior, mapping, replay), by default 4
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
    frame_timestamps = frame_timestamps[frame_offset:frame_offset + num_frames]

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
    frame_offset: int = 0
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

    # Fill in table row(s) for 'spontaneous' stimuli
    # Characterized by NaN for "stimulus_name" and "stimulus_block"
    spont_rows = (
        stim_df["stimulus_name"].isnull() & stim_df["stimulus_block"].isnull()
    )
    stim_df.loc[spont_rows, "stimulus_name"] = 'spontaneous'
    stim_df.loc[spont_rows, "stimulus_block"] = 2

    # Update "stimulus_block" values for gabor and flash stimuli
    # Gabor "stimulus_block" should take on value of 1
    gabor_rows = stim_df["stimulus_name"].str.contains("gabor")
    stim_df.loc[gabor_rows, "stimulus_block"] = 1
    # Flash "stimulus_block" should take on value of 3
    flash_rows = stim_df["stimulus_name"].str.contains("flash")
    stim_df.loc[flash_rows, "stimulus_block"] = 3

    stim_df['start_frame'] = stim_df['start_frame'].astype(int) + frame_offset
    stim_df['end_frame'] = stim_df['end_frame'].astype(int) + frame_offset
    stim_df['start_time'] = frame_timestamps[stim_df['start_frame']]
    stim_df['stop_time'] = frame_timestamps[stim_df['end_frame']]
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
    frame_offsets = get_frame_offsets(sync_dataset, frame_counts)

    # Generate stim tables for the 3 different stimulus pkl types
    behavior_df = generate_behavior_stim_table(
        behavior_pkl, sync_dataset, frame_offset=frame_offsets[0]
    )
    mapping_df = generate_mapping_stim_table(
        mapping_pkl, sync_dataset, frame_offset=frame_offsets[1]
    )
    replay_df = generate_replay_stim_table(
        replay_pkl, sync_dataset, behavior_df,
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
