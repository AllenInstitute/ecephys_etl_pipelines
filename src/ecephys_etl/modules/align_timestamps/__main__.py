import numpy as np

from argschema import ArgSchemaParser

from ecephys_etl.modules.align_timestamps._schemas import (
    InputParameters, OutputParameters
)
from ecephys_etl.modules.align_timestamps.barcode_sync_dataset import (
    BarcodeSyncDataset
)
from ecephys_etl.modules.align_timestamps.channel_states import (
    extract_barcodes_from_states, extract_splits_from_states
)
from ecephys_etl.modules.align_timestamps.probe_synchronizer import (
    ProbeSynchronizer
)


def align_timestamps(sync_h5_path: str, probes: dict) -> dict:

    sync_dataset = BarcodeSyncDataset.factory(args["sync_h5_path"])
    sync_times, sync_codes = sync_dataset.extract_barcodes()

    probe_output_info = []
    for probe in args["probes"]:
        print(probe["name"])
        this_probe_output_info = {}

        channel_states = np.load(probe["barcode_channel_states_path"])
        timestamps = np.load(probe["barcode_timestamps_path"])

        probe_barcode_times, probe_barcodes = extract_barcodes_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )
        probe_split_times = extract_splits_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )

        print("Split times:")
        print(probe_split_times)

        synchronizers = []

        for idx, split_time in enumerate(probe_split_times):

            min_time = probe_split_times[idx]

            if idx == (len(probe_split_times) - 1):
                max_time = np.Inf
            else:
                max_time = probe_split_times[idx + 1]

            synchronizer = ProbeSynchronizer.compute(
                sync_times,
                sync_codes,
                probe_barcode_times,
                probe_barcodes,
                min_time,
                max_time,
                probe["start_index"],
                probe["sampling_rate"],
            )

            synchronizers.append(synchronizer)

        mapped_files = {}

        for timestamp_file in probe["mappable_timestamp_files"]:
            # print(timestamp_file["name"])
            timestamps = np.load(timestamp_file["input_path"])
            aligned_timestamps = np.copy(timestamps).astype("float64")

            for synchronizer in synchronizers:
                aligned_timestamps = synchronizer(aligned_timestamps)
                print(
                    "total time shift: " + str(synchronizer.total_time_shift))
                print(
                    "actual sampling rate: "
                    + str(synchronizer.global_probe_sampling_rate)
                )

            np.save(
                timestamp_file["output_path"], aligned_timestamps,
                allow_pickle=False
            )
            mapped_files[timestamp_file["name"]] = timestamp_file[
                "output_path"]

        lfp_sampling_rate = (
                probe["lfp_sampling_rate"] * synchronizer.sampling_rate_scale
        )

        this_probe_output_info[
            "total_time_shift"] = synchronizer.total_time_shift
        this_probe_output_info[
            "global_probe_sampling_rate"
        ] = synchronizer.global_probe_sampling_rate
        this_probe_output_info[
            "global_probe_lfp_sampling_rate"] = lfp_sampling_rate
        this_probe_output_info["output_paths"] = mapped_files
        this_probe_output_info["name"] = probe["name"]

        probe_output_info.append(this_probe_output_info)

    return {"probe_outputs": probe_output_info}



if __name__ == "__main__":
    parser = ArgSchemaParser(
        schema_type=InputParameters, output_schema_type=OutputSchema
    )
    output = align_timestamps(**parser.args)

    output.update({"input_parameters": parser.args})
    if 'output_json' in parser.args:
        parser.output(output, indent=2)
    else:
        print(parser.get_output_json(output))