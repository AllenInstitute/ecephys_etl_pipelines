from typing import Dict, List, TypedDict
import logging

import numpy as np

from argschema import ArgSchemaParser

from ecephys_etl.modules.align_timestamps._schemas import (
    AlignTimestampsInputParameters, AlignTimestampsOutputParameters
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


class SingleProbeInfo(TypedDict):
    total_time_shift: float
    global_probe_sampling_rate: float
    global_probe_lfp_sampling_rate: float
    output_paths: Dict[str, str]
    name: str


def align_timestamps(
    sync_h5_path: str, probes: List[dict]
) -> Dict[str, List[SingleProbeInfo]]:
    """Perform alignment of neuropixels probe data to session timing
    information (in *.sync file).

    Parameters
    ----------
    sync_h5_path : str
        Path to a session h5 *.sync file.
    probes : List[dict]
        A list of dictionaries each containing metadata for each
        neuropixels probe. The fields in each dictionary are described
        in `ProbeInputParameters` in the align_timestamps module _schemas.py.

    Returns
    -------
    Dict[str, List[SingleProbeInfo]]
        Returns a dictionary where the top level key is "probe_outputs"
        and the value is a list of SingleProbeInfo TypedDicts.
    """
    logger = logging.getLogger("Ecephys_Align_Timestamps_Module")

    sync_dataset = BarcodeSyncDataset.factory(sync_h5_path)
    sync_times, sync_codes = sync_dataset.extract_barcodes()

    probe_output_info = []
    for probe in probes:
        logger.info(f"Aligning timestamps for: {probe['name']}")

        channel_states = np.load(probe["barcode_channel_states_path"])
        timestamps = np.load(probe["barcode_timestamps_path"])

        probe_barcode_times, probe_barcodes = extract_barcodes_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )
        probe_split_times = extract_splits_from_states(
            channel_states, timestamps, probe["sampling_rate"]
        )

        logger.info(f"Split times: {probe_split_times}")

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
            timestamps = np.load(timestamp_file["input_path"])
            aligned_timestamps = np.copy(timestamps).astype("float64")

            logger.info(
                f"Synchronization details for input file: "
                f"{timestamp_file['input_path']}"
            )

            for synchronizer in synchronizers:
                aligned_timestamps = synchronizer(aligned_timestamps)
                logger.info(
                    f"total time shift: {synchronizer.total_time_shift}"
                )
                logger.info(
                    f"actual sampling rate: "
                    f"{synchronizer.global_probe_sampling_rate}"
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

        probe_info: SingleProbeInfo = {
            "total_time_shift": synchronizer.total_time_shift,
            "global_probe_sampling_rate": (
                synchronizer.global_probe_sampling_rate
            ),
            "global_probe_lfp_sampling_rate": lfp_sampling_rate,
            "output_paths": mapped_files,
            "name": probe["name"]
        }
        probe_output_info.append(probe_info)

    return {"probe_outputs": probe_output_info}


if __name__ == "__main__":
    parser = ArgSchemaParser(
        schema_type=AlignTimestampsInputParameters,
        output_schema_type=AlignTimestampsOutputParameters
    )
    logging_level = parser.args.get('log_level', logging.DEBUG)

    # Need to remove root log handler instantiated by argschema
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format='%(asctime)-15s : %(name)-20s : %(levelname)-8s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging_level
    )

    logger = logging.getLogger("Ecephys_Align_Timestamps_Module")
    logger.setLevel(logging_level)

    output = align_timestamps(
        sync_h5_path=parser.args["sync_h5_path"],
        probes=parser.args["probes"]
    )

    output.update({"input_parameters": parser.args})
    if 'output_json' in parser.args:
        parser.output(output, indent=2)
    else:
        logging.info(parser.get_output_json(output))
