from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
    Nested, String, Float, Dict, Int, InputFile, OutputFile
)


class ProbeMappable(DefaultSchema):
    name = String(
        required=True,
        description=(
            "The name of the data modality to have timestamps remapped. "
            "e.g. 'spikes_timestamps', 'lfp_timestamps'"
        )
    )
    input_path = InputFile(
        required=True,
        description=(
            "Input path for this file. Should point to a file containing "
            "a 1D timestamps array with values in probe samples."
        ),
    )
    output_path = OutputFile(
        required=True,
        description=(
            "Output path for the mapped version of this file. Will write a "
            "1D timestamps array with values in seconds on the master clock."
        ),
    )


class ProbeInputParameters(DefaultSchema):
    name = String(required=True, description="Identifier for this probe")
    sampling_rate = Float(
        required=True,
        description=(
            "The sampling rate of the probe, in Hz, assessed on the probe "
            "clock."
        ),
    )
    lfp_sampling_rate = Float(
        required=True,
        description="The sampling rate of the LFP collected on this probe."
    )
    start_index = Int(
        default=0,
        description=(
            "Sample index of probe recording start time. Defaults to 0."
        )
    )
    barcode_channel_states_path = InputFile(
        required=True,
        description=(
            "Path to the channel states file. This file contains a "
            "1-dimensional array whose axis is events and whose "
            "values indicate the state of the channel line "
            "(rising or falling) at that event."
        ),
    )
    barcode_timestamps_path = InputFile(
        required=True,
        description=(
            "Path to the timestamps file. This file contains a "
            "1-dimensional array whose axis is events and whose "
            "values indicate the sample on which each event was detected."
        ),
    )
    mappable_timestamp_files = Nested(
        ProbeMappable,
        many=True,
        description=(
            "Timestamps files for this probe. Describe the times "
            "(in probe samples) when e.g. lfp samples were taken or "
            "spike events occured"
        )
    )


class AlignTimestampsInputParameters(ArgSchema):
    probes = Nested(
        ProbeInputParameters,
        many=True,
        description="Probes whose data will be aligned to the master clock.",
    )
    sync_h5_path = InputFile(
        required=True,
        description="path to h5 file containing syncronization information"
    )


class ProbeOutputParameters(DefaultSchema):
    name = String(required=True, description="Identifier for this probe")
    output_paths = Dict(
        required=True,
        description=(
            "Paths of each mappable file written by this run of the module."
        )
    )
    total_time_shift = Float(
        required=True,
        description=(
            "Translation (in seconds) from master->probe times computed "
            "for this probe."
        ),
    )
    global_probe_sampling_rate = Float(
        required=True,
        description=(
            "The sampling rate of this probe in Hz, assessed on the "
            "master clock."
        ),
    )
    global_probe_lfp_sampling_rate = Float(
        required=True,
        description=(
            "The sampling rate of LFP collected on this probe in Hz, "
            "assessed on the master clock."
        ),
    )


class OutputSchema(DefaultSchema):
    input_parameters = Nested(
        AlignTimestampsInputParameters,
        description="Input parameters the module was run with",
        required=True,
    )


class AlignTimestampsOutputParameters(OutputSchema):
    probe_outputs = Nested(
        ProbeOutputParameters, many="True", description="Probewise outputs."
    )
