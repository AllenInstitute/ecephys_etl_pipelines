from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def get_stimulus_presentations(data, stimulus_timestamps) -> pd.DataFrame:
    """
    This function retrieves the stimulus presentation dataframe and
    renames the columns, adds a stop_time column, and set's index to
    stimulus_presentation_id before sorting and returning the dataframe.
    :param data: stimulus file associated with experiment id
    :param stimulus_timestamps: timestamps indicating when stimuli switched
                                during experiment
    :return: stimulus_table: dataframe containing the stimuli metadata as well
                             as what stimuli was presented
    """
    stimulus_table = get_visual_stimuli_df(data, stimulus_timestamps)
    # workaround to rename columns to harmonize with visual
    # coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number',
                          value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(
        columns={'frame': 'start_frame',
                 'time': 'start_time',
                 'flash_number': 'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[int(start_frame)]
                                 for start_frame in
                                 stimulus_table.start_frame.values]
    end_time = []
    for end_frame in stimulus_table.end_frame.values:
        if not np.isnan(end_frame):
            end_time.append(stimulus_timestamps[int(end_frame)])
        else:
            end_time.append(float('nan'))

    stimulus_table.insert(loc=4, column='stop_time', value=end_time)
    stimulus_table.set_index('stimulus_presentations_id', inplace=True)
    stimulus_table = stimulus_table[sorted(stimulus_table.columns)]
    return stimulus_table


def _get_stimulus_epoch(set_log: List[Tuple[str, Union[str, int], int, int]],
                        current_set_index: int, start_frame: int,
                        n_frames: int) -> Tuple[int, int]:
    """
    Gets the frame range for which a stimuli was presented and the transition
    to the next stimuli was ongoing. Returns this in the form of a tuple.
    Parameters
    ----------
    set_log: List[Tuple[str, Union[str, int], int, int
        The List of Tuples in the form of
        (stimuli_type ('Image' or 'Grating'),
         stimuli_descriptor (image_name or orientation of grating in degrees),
         nonsynced_time_of_display (not sure, it's never used),
         display_frame (frame that stimuli was displayed))
    current_set_index: int
        Index of stimuli set to calculate window
    start_frame: int
        frame where stimuli was set, set_log[current_set_index][3]
    n_frames: int
        number of frames for which stimuli were displayed

    Returns
    -------
    Tuple[int, int]:
        A tuple where index 0 is start frame of stimulus window and index 1 is
        end frame of stimulus window

    """
    try:
        next_set_event = set_log[current_set_index + 1]
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames,)

    return start_frame, next_set_event[3]  # end frame isn't inclusive


def _get_draw_epochs(draw_log: List[int], start_frame: int,
                     stop_frame: int) -> List[Tuple[int, int]]:
    """
    Gets the frame numbers of the active frames within a stimulus window.
    Stimulus epochs come in the form [0, 0, 1, 1, 0, 0] where the stimulus is
    active for some amount of time in the window indicated by int 1 at that
    frame. This function returns the ranges for which the set_log is 1 within
    the draw_log window.
    Parameters
    ----------
    draw_log: List[int]
        A list of ints indicating for what frames stimuli were active
    start_frame: int
        The start frame to search within the draw_log for active values
    stop_frame: int
        The end frame to search within the draw_log for active values

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples indicating the start and end frames of every
        contiguous set of active values within the specified window
        of the draw log.
    """
    draw_epochs = []
    current_frame = start_frame

    while current_frame <= stop_frame:
        epoch_length = 0
        while current_frame < stop_frame and draw_log[current_frame] == 1:
            epoch_length += 1
            current_frame += 1
        else:
            current_frame += 1

        if epoch_length:
            draw_epochs.append(
                (current_frame - epoch_length - 1, current_frame - 1,)
            )

    return draw_epochs


def get_visual_stimuli_df(data, time) -> pd.DataFrame:
    """
    This function loads the stimuli and the omitted stimuli into a dataframe.
    These stimuli are loaded from the input data, where the set_log and
    draw_log contained within are used to calculate the epochs. These epochs
    are used as start_frame and end_frame and converted to times by input
    stimulus timestamps. The omitted stimuli do not have a end_frame by design
    though there duration is always 250ms.
    :param data: the behavior data file
    :param time: the stimulus timestamps indicating when each stimuli is
                 displayed
    :return: df: a pandas dataframe containing the stimuli and omitted stimuli
                 that were displayed with their frame, end_frame, start_time,
                 and duration
    """

    stimuli = data['items']['behavior']['stimuli']
    n_frames = len(time)
    visual_stimuli_data = []
    for stimuli_group_name, stim_dict in stimuli.items():
        for idx, (attr_name, attr_value, _time, frame,) in \
                enumerate(stim_dict["set_log"]):
            orientation = attr_value if attr_name.lower() == "ori" else np.nan
            image_name = attr_value if attr_name.lower() == "image" else np.nan

            stimulus_epoch = _get_stimulus_epoch(
                stim_dict["set_log"],
                idx,
                frame,
                n_frames,
            )
            draw_epochs = _get_draw_epochs(
                stim_dict["draw_log"],
                *stimulus_epoch
            )

            for idx, (epoch_start, epoch_end,) in enumerate(draw_epochs):

                visual_stimuli_data.append({
                    "orientation": orientation,
                    "image_name": image_name,
                    "frame": epoch_start,
                    "end_frame": epoch_end,
                    "time": time[epoch_start],
                    "duration": time[epoch_end] - time[epoch_start],
                    # this will always work because an epoch
                    # will never occur near the end of time
                    "omitted": False,
                })

    visual_stimuli_df = pd.DataFrame(data=visual_stimuli_data)

    # Add omitted flash info:
    try:
        omitted_flash_frame_log = \
            data['items']['behavior']['omitted_flash_frame_log']
    except KeyError:
        # For sessions for which there were no omitted flashes
        omitted_flash_frame_log = dict()

    omitted_flash_list = []
    for _, omitted_flash_frames in omitted_flash_frame_log.items():
        stim_frames = visual_stimuli_df['frame'].values
        omitted_flash_frames = np.array(omitted_flash_frames)

        # Test offsets of omitted flash frames
        # to see if they are in the stim log
        offsets = np.arange(-3, 4)
        offset_arr = np.add(
            np.repeat(omitted_flash_frames[:, np.newaxis],
                      offsets.shape[0], axis=1),
            offsets)
        matched_any_offset = np.any(np.isin(offset_arr, stim_frames), axis=1)

        #  Remove omitted flashes that also exist in the stimulus log
        was_true_omitted = np.logical_not(matched_any_offset)  # bool
        omitted_flash_frames_to_keep = omitted_flash_frames[was_true_omitted]

        # Have to remove frames that are double-counted in omitted log
        omitted_flash_list += list(np.unique(omitted_flash_frames_to_keep))

    omitted = np.ones_like(omitted_flash_list).astype(bool)
    time = [time[fi] for fi in omitted_flash_list]
    omitted_df = pd.DataFrame({'omitted': omitted,
                               'frame': omitted_flash_list,
                               'time': time,
                               'image_name': 'omitted'})

    df = pd.concat((visual_stimuli_df, omitted_df),
                   sort=False).sort_values('frame').reset_index()
    return df
