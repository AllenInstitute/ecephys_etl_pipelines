import json
from typing import List
from pathlib import Path

import pytest
import numpy as np

from ecephys_etl.modules.align_timestamps.__main__ import align_timestamps


@pytest.fixture
def align_timestamps_test_case_fixture(request, tmp_path) -> tuple:
    """This fixture creates an input dict as would be parsed by argparse
    for the align_timestamps module.

    Test data is stored as a *.json file

    Static test data for each test case is stored under
    tests/resources/align_timestamps
    """
    resource_dir = Path(__file__).resolve().parent.parent.parent / "resources"
    align_timestamps_res_dir = resource_dir / "align_timestamps"
    test_case_name = request.param["test_case_name"]

    test_dir = tmp_path / test_case_name
    test_dir_expected = test_dir / "expected"
    test_dir.mkdir()
    test_dir_expected.mkdir()

    test_case_json = (
        align_timestamps_res_dir / test_case_name / f"{test_case_name}.json"
    )
    with test_case_json.open('r') as f:
        test_data = json.load(f)

    input_probes = []
    expected_aligned = []
    for probe in test_data["probes"]:
        test_input_probe = dict()
        pname = probe["name"]
        test_input_probe["name"] = pname
        test_input_probe["sampling_rate"] = probe["sampling_rate"]
        test_input_probe["lfp_sampling_rate"] = probe["sampling_rate"]
        test_input_probe["start_index"] = 0

        # Need to convert json test data into expected *.npy file inputs
        barcode_timestamps = np.array(probe["barcode_timestamps"])
        barcode_ts_path = test_dir / f"{pname}_barcode_timestamps.npy"
        np.save(barcode_ts_path, barcode_timestamps)
        test_input_probe["barcode_timestamps_path"] = barcode_ts_path

        channel_states = np.array(probe["channel_states"])
        channel_states_path = test_dir / f"{pname}_channel_states.npy"
        np.save(channel_states_path, channel_states)
        test_input_probe["barcode_channel_states_path"] = channel_states_path

        raw_lfp_timestamps = np.array(probe["raw_lfp_timestamps"])
        lfp_path = test_dir / f"{pname}_lfp_timestamps.npy"
        np.save(lfp_path, raw_lfp_timestamps)
        lfp_output_path = test_dir / f"{pname}_lfp_aligned.npy"
        lfp_ts_entry = {
            "name": "lfp_timestamps",
            "input_path": str(lfp_path),
            "output_path": str(lfp_output_path)
        }

        raw_spike_timestamps = np.array(probe["raw_spike_timestamps"])
        spikes_path = test_dir / f"{pname}_spike_timestamps.npy"
        np.save(spikes_path, raw_spike_timestamps)
        spikes_output_path = test_dir / f"{pname}_spikes_aligned.npy"
        spikes_ts_entry = {
            "name": "spikes_timestamps",
            "input_path": str(spikes_path),
            "output_path": str(spikes_output_path)
        }
        test_input_probe["mappable_timestamp_files"] = [
            lfp_ts_entry, spikes_ts_entry
        ]
        input_probes.append(test_input_probe)

        # Convert json test data expected results to *.npy file outputs
        expected_lfp_path = test_dir_expected / f"{pname}_lfp_expected.npy"
        expected_lfp = np.array(probe["aligned_lfp_timestamps"])
        np.save(expected_lfp_path, expected_lfp)

        expected_spk_path = test_dir_expected / f"{pname}_spikes_expected.npy"
        expected_spk = np.array(probe["aligned_spike_timestamps"])
        np.save(expected_spk_path, expected_spk)

        expected_aligned.append({
            "expected_aligned_lfp": str(expected_lfp_path),
            "expected_aligned_spikes": str(expected_spk_path)
        })

    module_input = {
        "sync_h5_path": str(
            align_timestamps_res_dir
            / test_case_name
            / test_data["sync_file_fname"]
        ),
        "probes": input_probes
    }

    return module_input, expected_aligned


@pytest.mark.parametrize("align_timestamps_test_case_fixture", [
    {"test_case_name": "vcn_test_case_1"},
    {"test_case_name": "vbn_test_case_1"}


], indirect=["align_timestamps_test_case_fixture"])
def test_align_timestamps(align_timestamps_test_case_fixture):
    test_input, test_expected = align_timestamps_test_case_fixture

    print(test_input)
    print(test_expected)

    probe_outputs: List[dict] = align_timestamps(
        sync_h5_path=test_input["sync_h5_path"],
        probes=test_input["probes"]
    )["probe_outputs"]

    for indx, obt in enumerate(probe_outputs):
        exp_outputs = test_expected[indx]
        exp_aligned_lfp = np.load(exp_outputs["expected_aligned_lfp"])
        exp_aligned_spikes = np.load(exp_outputs["expected_aligned_spikes"])

        obt_aligned_lfp = np.load(obt["output_paths"]["lfp_timestamps"])
        obt_aligned_spikes = np.load(obt["output_paths"]["spikes_timestamps"])

        assert np.allclose(exp_aligned_lfp, obt_aligned_lfp)
        assert np.allclose(exp_aligned_spikes, obt_aligned_spikes)
