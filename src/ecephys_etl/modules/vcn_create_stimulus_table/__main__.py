import functools

import numpy as np
from argschema import ArgSchemaParser

from ecephys_etl.data_extractors.ecephys_sync_dataset import (
    EcephysSyncDataset,
)
from ecephys_etl.data_extractors.stim_file import CamStimOnePickleStimFile
from ecephys_etl.modules.vcn_create_stimulus_table import ephys_pre_spikes
from ecephys_etl.modules.vcn_create_stimulus_table import naming_utilities
from ecephys_etl.modules.vcn_create_stimulus_table import output_validation
from ecephys_etl.modules.vcn_create_stimulus_table._schemas import (
    InputParameters, OutputSchema
)


def build_stimulus_table(
        stimulus_pkl_path,
        sync_h5_path,
        frame_time_strategy,
        minimum_spontaneous_activity_duration,
        extract_const_params_from_repr,
        drop_const_params,
        maximum_expected_spontaneous_activity_duration,
        stimulus_name_map,
        column_name_map,
        output_stimulus_table_path,
        output_frame_times_path,
        fail_on_negative_duration,
        **kwargs
):
    stim_file = CamStimOnePickleStimFile.factory(stimulus_pkl_path)

    sync_dataset = EcephysSyncDataset.factory(sync_h5_path)
    frame_times = sync_dataset.extract_frame_times(
        strategy=frame_time_strategy)

    def seconds_to_frames(seconds):
        return  \
            (np.array(seconds) + stim_file.pre_blank_sec) * \
            stim_file.frames_per_second

    minimum_spontaneous_activity_duration = (
            minimum_spontaneous_activity_duration / stim_file.frames_per_second
    )

    stimulus_tabler = functools.partial(
        ephys_pre_spikes.build_stimuluswise_table,
        seconds_to_frames=seconds_to_frames,
        extract_const_params_from_repr=extract_const_params_from_repr,
        drop_const_params=drop_const_params,
    )
    spon_tabler = functools.partial(
        ephys_pre_spikes.make_spontaneous_activity_tables,
        duration_threshold=minimum_spontaneous_activity_duration,
    )

    stim_table_full = ephys_pre_spikes.create_stim_table(
        stim_file.stimuli, stimulus_tabler, spon_tabler
    )
    stim_table_full = ephys_pre_spikes.apply_frame_times(
        stim_table_full, frame_times, stim_file.frames_per_second, True
    )

    output_validation.validate_epoch_durations(
        stim_table_full, fail_on_negative_durations=fail_on_negative_duration)
    output_validation.validate_max_spontaneous_epoch_duration(
        stim_table_full, maximum_expected_spontaneous_activity_duration
    )

    stim_table_full = naming_utilities.collapse_columns(stim_table_full)
    stim_table_full = naming_utilities.drop_empty_columns(stim_table_full)
    stim_table_full = naming_utilities.standardize_movie_numbers(
        stim_table_full)
    stim_table_full = naming_utilities.add_number_to_shuffled_movie(
        stim_table_full)
    stim_table_full = naming_utilities.map_stimulus_names(
        stim_table_full, stimulus_name_map
    )
    stim_table_full = naming_utilities.map_column_names(stim_table_full,
                                                        column_name_map)

    stim_table_full.to_csv(output_stimulus_table_path, index=False)
    np.save(output_frame_times_path, frame_times, allow_pickle=False)
    return {
        "output_path": output_stimulus_table_path,
        "output_frame_times_path": output_frame_times_path,
    }


if __name__ == "__main__":
    parser = ArgSchemaParser(
        schema_type=InputParameters, output_schema_type=OutputSchema
    )
    output = build_stimulus_table(**parser.args)

    output.update({"input_parameters": parser.args})
    if 'output_json' in parser.args:
        parser.output(output, indent=2)
    else:
        print(parser.get_output_json(output))
