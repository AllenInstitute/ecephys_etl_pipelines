from unittest.mock import create_autospec

import pytest
import pandas as pd

from ecephys_etl.data_extractors.sync_dataset import Dataset
from ecephys_etl.data_extractors.stim_file import (
    CamStimOnePickleStimFile,
    BehaviorPickleFile,
    ReplayPickleFile
)
from ecephys_etl.modules.vbn_create_stimulus_table.create_stim_table import (
    create_vbn_stimulus_table
)
from ecephys_etl.modules.vbn_create_stimulus_table.__main__ import (
    VbnCreateStimulusTable
)


@pytest.mark.parametrize(
    "mock_full_stim_table, sync_fname, behavior_pkl_fname, "
    "mapping_pkl_fname, replay_pkl_fname, output_csv_fname",
    [
        (
            # mock_full_stim_table
            pd.DataFrame({
                "test_column_1": [1, 2, 3, 4, 5],
                "test_column_2": [5, 6, 7, 8, 9]
            }),
            # sync_fname
            "test_sync.h5",
            # behavior_pkl_fname
            "test.behavior.pkl",
            # mapping_pkl_fname
            "test.mapping.pkl",
            # replay_pkl_fname
            "test.replay.pkl",
            # output_csv_fname
            "vbn_test_stim_table.csv"
        )
    ]
)
def test_vbn_create_stimulus_table(
    monkeypatch, tmp_path, mock_full_stim_table, sync_fname,
    behavior_pkl_fname, mapping_pkl_fname, replay_pkl_fname, output_csv_fname
):
    # Create mock input files (that can pass argschema)
    sync_path = tmp_path / sync_fname
    sync_path.touch()
    behavior_pkl_path = tmp_path / behavior_pkl_fname
    behavior_pkl_path.touch()
    mapping_pkl_path = tmp_path / mapping_pkl_fname
    mapping_pkl_path.touch()
    replay_pkl_path = tmp_path / replay_pkl_fname
    replay_pkl_path.touch()
    output_path = tmp_path / output_csv_fname
    output_json_path = tmp_path / "module_output.json"

    # Create mocks to be patched in
    MockDataset = create_autospec(Dataset)
    MockCamStimOnePickleStimFile = create_autospec(CamStimOnePickleStimFile)
    MockBehaviorPickleFile = create_autospec(BehaviorPickleFile)
    MockReplayPickleFile = create_autospec(ReplayPickleFile)

    mock_create_vbn_stimulus_table = create_autospec(create_vbn_stimulus_table)
    mock_create_vbn_stimulus_table.return_value = mock_full_stim_table

    with monkeypatch.context() as m:
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".__main__.Dataset",
            MockDataset
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".__main__.CamStimOnePickleStimFile",
            MockCamStimOnePickleStimFile
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".__main__.BehaviorPickleFile",
            MockBehaviorPickleFile
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".__main__.ReplayPickleFile",
            MockReplayPickleFile
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".__main__.create_vbn_stimulus_table",
            mock_create_vbn_stimulus_table
        )

        stim_table_maker = VbnCreateStimulusTable(
            input_data={
                "sync_h5_path": str(sync_path),
                "behavior_pkl_path": str(behavior_pkl_path),
                "mapping_pkl_path": str(mapping_pkl_path),
                "replay_pkl_path": str(replay_pkl_path),
                "output_stimulus_table_path": str(output_path),
                "output_json": str(output_json_path)
            },
            args=[]
        )

        stim_table_maker.run()

    MockDataset.assert_called_with(str(sync_path))
    MockBehaviorPickleFile.factory.assert_called_with(str(behavior_pkl_path))
    MockCamStimOnePickleStimFile.factory.assert_called_with(
        str(mapping_pkl_path)
    )
    MockReplayPickleFile.factory.assert_called_with(str(replay_pkl_path))

    mock_create_vbn_stimulus_table.assert_called_with(
        sync_dataset=MockDataset.return_value,
        behavior_pkl=MockBehaviorPickleFile.factory.return_value,
        mapping_pkl=MockCamStimOnePickleStimFile.factory.return_value,
        replay_pkl=MockReplayPickleFile.factory.return_value
    )

    assert output_path.exists()
