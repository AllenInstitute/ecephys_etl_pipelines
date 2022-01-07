from unittest.mock import create_autospec, MagicMock
import logging

import pytest
import numpy as np
import pandas as pd

from ecephys_etl.data_extractors.sync_dataset import Dataset
from ecephys_etl.data_extractors.stim_file import (
    CamStimOnePickleStimFile,
    BehaviorPickleFile,
    ReplayPickleFile
)
from ecephys_etl.modules.vbn_create_stimulus_table import create_stim_table

INDEX_TO_BEHAVIOR = 0
INDEX_TO_MAPPING = 1
INDEX_TO_REPLAY = 2


@pytest.mark.parametrize(
    "mock_line_labels, expected_line, falling_edges, rising_edges, warning",
    [
        # Basic tests
        (
            ["vsync_stim", "unrelated_line"],
            "vsync_stim",
            np.array([0.1, 1, 2, 3, 4]),
            np.array([0, 1.5, 2.5, 3.5, 4.5]),
            None
        ),
        (
            ["stim_vsync", "unrelated_line"],
            "stim_vsync",
            np.array([5, 6, 7, 8, 9]),
            np.array([0, 1.5, 2.5, 3.5, 4.5]),
            None
        ),
        # Test that get_vsyncs will grab first relevant line label
        (
            ["vsync_stim", "stim_vsync", "unrelated_line"],
            "vsync_stim",
            np.array([0.1, 1, 2, 3, 4]),
            np.array([0, 1.5, 2.5, 3.5, 4.5]),
            None
        ),
        (
            ["stim_vsync", "vsync_stim", "unrelated_line"],
            "stim_vsync",
            np.array([0.1, 1, 2, 3, 4]),
            np.array([0, 1.5, 2.5, 3.5, 4.5]),
            None
        ),
        # Test that fallback is used and warning is raised if no relevant
        # line label found.
        (
            ["unrelated_line"],
            2,
            np.array([10, 11, 12, 13, 14]),
            np.array([0, 1.5, 2.5, 3.5, 4.5]),
            "Could not find 'vsync_stim' nor 'stim_vsync' line labels"
        )
    ]
)
def test_get_vsyncs(
    caplog, mock_line_labels, expected_line,
    falling_edges, rising_edges, warning
):
    mock_dataset = MagicMock()
    mock_dataset.line_labels = mock_line_labels
    mock_dataset.dfile.filename = "/tmp/dummy_sync_path.sync"
    mock_dataset.get_falling_edges.return_value = falling_edges
    mock_dataset.get_rising_edges.return_value = rising_edges

    with caplog.at_level(logging.WARNING):
        obt = create_stim_table.get_vsyncs(sync_dataset=mock_dataset)

    if warning:
        assert warning in caplog.text

    mock_dataset.get_falling_edges.assert_called_with(
        expected_line, units="seconds"
    )
    assert np.allclose(falling_edges, obt)


@pytest.mark.parametrize(
    "mock_line_labels, expected_line, falling_edges, rising_edges, warning",
    [
        # Basic tests
        (
            ["stim_running", "unrelated_line"],
            "stim_running",
            np.array([0, 1, 2, 3, 4]),
            np.array([4, 3, 2, 1, 0]),
            None
        ),
        (
            ["sweep", "unrelated_line"],
            "sweep",
            np.array([5, 6, 7, 8, 9]),
            np.array([9, 8, 7, 6, 5]),
            None
        ),
        # Test get_stim_starts_and_ends will grab first relevant line label
        (
            ["stim_running", "sweep", "unrelated_line"],
            "stim_running",
            np.array([0, 1, 2, 3, 4]),
            np.array([4, 3, 2, 1, 0]),
            None
        ),
        (
            ["sweep", "stim_running", "unrelated_line"],
            "sweep",
            np.array([0, 1, 2, 3, 4]),
            np.array([4, 3, 2, 1, 0]),
            None
        ),
        # Test that fallback is used and warning is raised if no relevant
        # line label found.
        (
            ["unrelated_line"],
            5,
            np.array([10, 11, 12, 13, 14]),
            np.array([14, 13, 12, 11, 10]),
            "Could not find 'stim_running' nor 'sweep' line labels"
        )
    ]
)
def test_get_stim_starts_and_ends(
    caplog, mock_line_labels, expected_line, falling_edges, rising_edges,
    warning
):
    mock_dataset = MagicMock()
    mock_dataset.line_labels = mock_line_labels
    mock_dataset.dfile.filename = "/tmp/dummy_sync_path.sync"
    mock_dataset.get_falling_edges.return_value = falling_edges
    mock_dataset.get_rising_edges.return_value = rising_edges

    with caplog.at_level(logging.WARNING):
        obt = create_stim_table.get_stim_starts_and_ends(
            sync_dataset=mock_dataset
        )

    if warning:
        assert warning in caplog.text

    mock_dataset.get_falling_edges.assert_called_with(
        expected_line, units="seconds"
    )
    mock_dataset.get_rising_edges.assert_called_with(
        expected_line, units="seconds"
    )
    assert np.allclose(rising_edges, obt[0])
    assert np.allclose(falling_edges, obt[1])


@pytest.fixture()
def mock_sync_dataset_fixture(request):
    """
    This fixture mimics a sync file `Dataset` object and allows parametrization
    for testing.
    """

    mock_line_labels: list = request.param.get(
        "line_labels", ["vsync_stim", "stim_running"]
    )
    get_rising_edges_map: dict = request.param.get(
        "get_rising_edges", {
            "stim_running": np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
            "vsync_stim": np.array([1, 2, 3, 4, 5])
        },

    )
    get_falling_edges_map: dict = request.param.get(
        "get_falling_edges",
        {
            "stim_running": np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
            "vsync_stim": np.array([1, 2, 3, 4, 5])
        }
    )

    def mock_get_rising_edges(stim_line, units, **kwargs):
        return get_rising_edges_map[stim_line]

    def mock_get_falling_edges(stim_line, units, **kwargs):
        return get_falling_edges_map[stim_line]

    def mock_get_edges(kind, keys, units, **kwargs):

        stim_line = keys[0]

        if stim_line == 4:
            stim_line = 'stim_photodiode'

        return np.sort(np.concatenate([
            get_rising_edges_map[stim_line],
            get_falling_edges_map[stim_line]
        ]))

    mock_sync_dataset = create_autospec(Dataset, instance=True)
    mock_sync_dataset.line_labels = mock_line_labels
    mock_sync_dataset.get_rising_edges.side_effect = mock_get_rising_edges
    mock_sync_dataset.get_falling_edges.side_effect = mock_get_falling_edges
    mock_sync_dataset.get_edges.side_effect = mock_get_edges

    return mock_sync_dataset


@pytest.mark.parametrize(
    "mock_sync_dataset_fixture, frame_counts, "
    "tolerance, expected_behavior, expected_mapping, expected_replay, "
    "warning, raises",
    [
        # Simple test case
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([.5, 2.5, 4.5, 6.5, 8.8]),
                    "stim_running": np.array([1, 3, 5, 7, 9])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1.5, 3.5, 5.5, 7.5, 9.5]),
                    "stim_running": np.array([2, 4, 6, 8, 10])
                }
            },
            # frame_counts
            [1, 1, 1, 1, 1],
            # tolerance
            0,
            # expected behavior
            [0, 1, 2, 3, 4],
            # expected mapping
            [1, 3, 5, 7, 9],
            # expected replay
            [2,  4,  6,  8, 10],
            # warning
            False,
            # raises
            False
        ),
        # Test warning when frame counts don't match
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([.5, 2.5, 4.5, 6.5, 8.5]),
                    "stim_running": np.array([1, 3, 5, 7, 9])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1.5, 3.5, 5.5, 7.5, 9.5]),
                    "stim_running": np.array([2, 4, 6, 8, 10])
                }
            },
            # frame_counts
            [10, 1, 1, 1, 1],
            # tolerance
            0,
            # expected behavior
            [0, 1, 2, 3, 4],
            # expected mapping
            [1, 3, 5, 7, 9],
            # expected replay
            [2,  4,  6,  8, 10],
            # warning
            "Number of frames derived from sync file",
            # raises
            False
        ),
        # Test when number of sync epochs don't match pkl case 1
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([0, 10.5, 21.5]),
                    "stim_running": np.array([.5, 11, 22])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1, 2, 3, 13, 14, 23, 24, 25, 26]),
                    "stim_running": np.array([10, 21, 32])
                }
            },
            # frame_counts
            [3, 4],
            # tolerance
            0,
            # expected behavior
            [0, 5],
            # expected mapping
            [0.5, 11, 22],
            # expected replay
            [10, 21, 32],
            # warning
            "Number of stim presentations obtained from sync (3) "
            "higher than number expected (2). Inferring start frames.",
            # raises
            False
        ),
        # Test when number of sync epochs don't match pkl case 2
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([0, 10.5, 21.5]),
                    "stim_running": np.array([.5, 11, 22])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1, 2, 3, 13, 14, 23, 24, 25, 26]),
                    "stim_running": np.array([10, 21, 32])
                }
            },
            # frame_counts
            [3, 2],
            # tolerance
            0,
            # expected behavior
            [0, 3],
            # expected mapping
            [0.5, 11, 22],
            # expected replay
            [10, 21, 32],
            # warning
            "Number of stim presentations obtained from sync",
            # raises
            False
        ),
        # Test when frame_counts more than sync epoch_frame_counts
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([0, 10.5, 21.5]),
                    "stim_running": np.array([.5, 11, 22])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1, 2, 3, 13, 14, 23, 24, 25, 26]),
                    "stim_running": np.array([10, 21, 32])
                }
            },
            # frame_counts
            [3, 2, 4, 5],
            # tolerance
            0,
            # expected behavior
            None,
            # expected mapping
            None,
            # expected replay
            None,
            # warning
            False,
            # raises
            "Do not know how to handle more pkl frame count entries"
        ),
        # Test when number of sync epochs don't match pkl but can be matched
        # with provided tolerance
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([0, 10.5, 21.5]),
                    "stim_running": np.array([.5, 11, 22])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1, 2, 3, 13, 14, 23, 24, 25, 26]),
                    "stim_running": np.array([10, 21, 32])
                }
            },
            # frame_counts
            [3, 5],
            # tolerance
            20,
            # expected behavior
            [0, 5],
            # expected mapping
            [0.5, 11, 22],
            # expected replay
            [10, 21, 32],
            # warning
            "Number of stim presentations obtained from sync",
            # raises
            False
        ),

        # Test when number of sync epochs don't match pkl and can't be matched
        # with provided tolerance
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "vsync_stim": np.array([0, 10.5, 21.5]),
                    "stim_running": np.array([.5, 11, 22])
                },
                "get_falling_edges": {
                    "vsync_stim": np.array([1, 2, 3, 13, 14, 23, 24, 25, 26]),
                    "stim_running": np.array([10, 21, 32])
                }
            },
            # frame_counts
            [3, 6],
            # tolerance
            0,
            # expected behavior
            None,
            # expected mapping
            None,
            # expected replay
            None,
            # warning
            False,
            # raises
            "Could not find matching sync frames for stim"
        ),
    ],
    indirect=["mock_sync_dataset_fixture"]
)
def test_get_frame_offsets(
    caplog, mock_sync_dataset_fixture, frame_counts, tolerance,
    expected_behavior, expected_mapping, expected_replay, warning, raises
):
    if raises:
        with pytest.raises(RuntimeError, match=raises):
            create_stim_table.get_frame_offsets(
                sync_dataset=mock_sync_dataset_fixture,
                frame_counts=frame_counts,
                tolerance=tolerance
            )
    else:
        obt = create_stim_table.get_frame_offsets(
            sync_dataset=mock_sync_dataset_fixture,
            frame_counts=frame_counts,
            tolerance=tolerance
        )

        if warning:
            assert warning in caplog.text

        assert np.allclose(expected_behavior, obt[INDEX_TO_BEHAVIOR])
        assert np.allclose(expected_mapping, obt[INDEX_TO_MAPPING])
        assert np.allclose(expected_replay, obt[INDEX_TO_REPLAY])


@pytest.mark.parametrize(
    "raw_behavior_stimulus_df, reward_times, expected",
    [
        # Basic test case
        (
            # raw_behavior_stimulus_df
            pd.DataFrame({
                "image_name": ['im_0', 'im_0', 'im_0', 'im_1', 'im_1', 'im_1'],
                "start_time": [0, 1, 2, 3, 4, 5],
                "omitted": [False, False, False, False, False, False]
            }),
            # reward times
            np.array([3, 20]),
            # expected
            {
                "flashes_since_change": [0, 1, 2, 0, 1, 2],
                "is_change": [False, False, False, True, False, False],
                "rewarded": [False, False, False, True, False, False]
            }
        ),
        # Test omitted doesn't incrememnt flashes_since_change
        (
            # raw_behavior_stimulus_df
            pd.DataFrame({
                "image_name": ['im_0', 'im_0', 'im_0', 'im_1', 'im_1', 'im_1'],
                "start_time": [0, 1, 2, 3, 4, 5],
                "omitted": [False, True, False, False, False, False]
            }),
            # reward times
            np.array([3, 20]),
            # expected
            {
                "flashes_since_change": [0, 0, 1, 0, 1, 2],
                "is_change": [False, False, False, True, False, False],
                "rewarded": [False, False, False, True, False, False]
            }
        ),
        # Test omitted not counted as change
        (
            # raw_behavior_stimulus_df
            pd.DataFrame({
                "image_name": ['im_0', 'im_0', 'im_0', 'im_1', 'im_1', 'im_1'],
                "start_time": [0, 1, 2, 3, 4, 5],
                "omitted": [False, False, False, True, False, False]
            }),
            # reward times
            np.array([4, 20]),
            # expected
            {
                "flashes_since_change": [0, 1, 2, 2, 0, 1],
                "is_change": [False, False, False, False, True, False],
                "rewarded": [False, False, False, False, True, False]
            }
        ),
        # Test subject missing reward times does not count as 'rewarded'
        (
            # raw_behavior_stimulus_df
            pd.DataFrame({
                "image_name": ['im_0', 'im_0', 'im_0', 'im_1', 'im_1', 'im_1'],
                "start_time": [0, 1, 2, 3, 4, 5],
                "omitted": [False, False, False, False, False, False]
            }),
            # reward times (subject responded too late)
            np.array([4, 20]),
            # expected
            {
                "flashes_since_change": [0, 1, 2, 0, 1, 2],
                "is_change": [False, False, False, True, False, False],
                "rewarded": [False, False, False, False, False, False]
            }
        ),
    ]
)
def test_determine_behavior_stimulus_properties(
    raw_behavior_stimulus_df, reward_times, expected
):
    obt = create_stim_table.determine_behavior_stimulus_properties(
        raw_behavior_stimulus_df=raw_behavior_stimulus_df,
        reward_times=reward_times
    )

    assert np.allclose(
        expected["flashes_since_change"], obt["flashes_since_change"]
    )
    assert np.allclose(expected["is_change"], obt["is_change"])
    assert np.allclose(expected["rewarded"], obt["rewarded"])


@pytest.fixture()
def mock_behavior_pkl_fixture(request):

    mock_beh_pkl = create_autospec(BehaviorPickleFile, instance=True)
    mock_beh_pkl.image_set = request.param.get("image_set", "test_img_set_1")
    mock_beh_pkl.num_frames = request.param.get("num_frames", 10)
    mock_beh_pkl.reward_frames = request.param.get("reward_frames", [42])

    return mock_beh_pkl


@pytest.mark.parametrize(
    "mock_behavior_pkl_fixture, mock_sync_dataset_fixture,"
    "stim_presentations_df, stim_properties, "
    "stim_start, stim_end, frame_offset,"
    "block_offset, expected",
    [
        (
            # mock_behavior_pkl_fixture
            {
                "image_set": "test_image_set",
                "num_frames": 1000,
                "reward_frames": np.array([4])
            },
            # mock_sync_dataset_fixture
            {
                "line_labels": [
                    "vsync_stim",
                    "stim_running",
                    'stim_photodiode'
                ],
                "get_falling_edges": {
                    "vsync_stim": np.arange(1, 1000, 1),
                    'stim_running': np.array(
                        [1]
                    ),
                    'stim_photodiode': np.arange(0, 1000, 60),
                },

                "get_rising_edges": {
                    "vsync_stim": np.array(
                        np.arange(.5, 1000, 1)
                    ),
                    'stim_running': np.array(
                        [1000]
                    ),
                    'stim_photodiode': [],
                },
            },
            # stim_presentations_df
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 2., 2., 2., 1.],
                "end_frame": [1, 3, 5, 7, 9],
                "image_name": ["a", "a", "b", "b", "b"],
                "index": [0, 1, 2, 3, 4],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [0, 2, 4, 6, 8],
                "start_time": [1., 4., 8., 12., 16.],
                "stop_time": [2., 5., 10., 14., 17.]
            }).set_index("stimulus_presentations_id"),
            # stim_properties
            {
                "flashes_since_change": [0, 1, 0, 1, 2],
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False]
            },
            # stim_start
            0,
            # stim_end
            1000,
            # frame_offset
            0,
            # block_offset
            0,
            # expected
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 2., 2., 2., 1.],
                "end_frame": [1, 3, 5, 7, 9],
                "image_name": ["a", "a", "b", "b", "b"],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [0, 2, 4, 6, 8],
                "start_time": [1., 4., 8., 12., 16.],
                "stop_time": [2., 5., 10., 14., 17.],
                "stimulus_block": [0] * 5,
                "stimulus_name": ["test_image_set"] * 5,
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False],
                "flashes_since_change": [0, 1, 0, 1, 2],
                "active": [True] * 5,
            }).set_index("stimulus_presentations_id")
        )
    ],
    indirect=["mock_behavior_pkl_fixture", "mock_sync_dataset_fixture"]
)
def test_generate_behavior_stim_table(
    monkeypatch, mock_behavior_pkl_fixture, mock_sync_dataset_fixture,
    stim_presentations_df, stim_properties, stim_start,
    stim_end, frame_offset, block_offset,
    expected
):
    # Set up mocks that need to be patched in
    mock_get_stimulus_presentations = create_autospec(
        create_stim_table.get_stimulus_presentations
    )
    mock_get_stimulus_presentations.return_value = stim_presentations_df

    mock_determine_behavior_stimulus_properties = create_autospec(
        create_stim_table.determine_behavior_stimulus_properties
    )
    mock_determine_behavior_stimulus_properties.return_value = stim_properties

    with monkeypatch.context() as m:
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.get_stimulus_presentations",
            mock_get_stimulus_presentations
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.determine_behavior_stimulus_properties",
            mock_determine_behavior_stimulus_properties
        )

        obt = create_stim_table.generate_behavior_stim_table(
            mock_behavior_pkl_fixture,
            mock_sync_dataset_fixture,
            stim_start,
            stim_end,
            frame_offset,
            block_offset
        )

    pd.testing.assert_frame_equal(expected, obt)


@pytest.fixture()
def mock_replay_pkl_fixture(request):

    mock_replay_pkl = create_autospec(ReplayPickleFile, instance=True)
    mock_replay_pkl.image_presentations = request.param.get(
        "image_presentations", ["a", "a"] + [None] * 71 + ["b", "b", "b"]
    )
    mock_replay_pkl.num_frames = request.param.get("num_frames", 10)
    return mock_replay_pkl


@pytest.mark.parametrize(
    "mock_replay_pkl_fixture, behavior_stim_table, raises_assert_error",
    [
        # Basic test case
        (
            {
                "image_presentations":
                    [None] * 18 + ["aa"] * 18 + [None] * 18 + ["bb"] * 18
            },
            pd.DataFrame({"image_name": ["aa", "bb"]}),
            False
        ),
        # Test case with omitted stimuli
        (
            {
                "image_presentations":
                    [None] * 18 + ["aa"] * 18 + [None] * 71 + ["bb"] * 18
            },
            pd.DataFrame({"image_name": ["aa", "omitted", "bb"]}),
            False
        ),
        # Test case where there is a mismatch (should raise)
        (
            {
                "image_presentations":
                    [None] * 18 + ["aa"] * 18 + [None] * 71 + ["bb"] * 18
            },
            pd.DataFrame({"image_name": ["aa", "bb"]}),
            True
        ),
        # Another kind of mismatch (should raise)
        (
            {
                "image_presentations":
                    [None] * 18 + ["aa"] * 18 + [None] * 71 + ["bb"] * 18
            },
            pd.DataFrame({"image_name": ["aa", "omitted", "cc"]}),
            True
        ),
    ],
    indirect=["mock_replay_pkl_fixture"]
)
def test_check_behavior_and_replay_pkl_match(
    mock_replay_pkl_fixture, behavior_stim_table, raises_assert_error
):
    if raises_assert_error:
        with pytest.raises(AssertionError):
            create_stim_table.check_behavior_and_replay_pkl_match(
                mock_replay_pkl_fixture, behavior_stim_table
            )
    else:
        create_stim_table.check_behavior_and_replay_pkl_match(
            mock_replay_pkl_fixture, behavior_stim_table
        )


@pytest.mark.parametrize(
    "mock_replay_pkl_fixture, mock_sync_dataset_fixture, stim_start, stim_end,"
    "behavior_stim_table, block_offset, frame_offset, expected",
    [
        (
            # mock_replay_pkl_fixture
            {
                "image_set": "test_image_set",
                "num_frames": 10,
                "reward_frames": np.array([4])
            },
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", 'stim_running'],
                "get_falling_edges": {
                    "vsync_stim": np.array(
                        # Behavior frame times
                        [1., 2., 4., 6., 8., 10., 12., 14., 16., 17.]
                        # Mapping frame times
                        + [18., 19., 20., 21., 22., 23., 24., 25., 26., 27.]
                        # Replay frame times
                        + [31., 32., 34., 35., 38., 40., 43., 44., 46., 47.]
                    ),
                    "stim_running": np.array(
                        [1.5, 10, 20, 30, 40, 50., 60]
                    ),
                },

                "get_rising_edges": {
                    "vsync_stim": np.array(
                        # Behavior frame times
                        [1.1, 2.1, 4.1, 6.1, 8.1, 10.1,
                            12.1, 14.1, 16.1, 17.1]
                        # Mapping frame times
                        + [18.1, 19.1, 20.1, 21.1, 22.1, 23.1,
                            24.1, 25.1, 26.1, 27.1]
                        # Replay frame times
                        + [31.1, 32.1, 34.1, 35.1, 38.1, 40.1,
                            43.1, 44.1, 46.1, 47.1]
                    ),
                    'stim_running': np.array(
                        [2., 25, 35., 45., 55., 65., 75]
                    ),
                },
            },
            # stim_start
            0,
            # stim_end
            1000,

            # behavior_stim_table
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 2., 2., 2., 1.],
                "end_frame": [1, 3, 5, 7, 9],
                "image_name": ["a", "a", "b", "b", "b"],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [0, 2, 4, 6, 8],
                "start_time": [1., 4., 8., 12., 16.],
                "stop_time": [2., 5., 10., 14., 17.],
                "stimulus_block": [0] * 5,
                "stimulus_name": ["test_image_set"] * 5,
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False],
                "flashes_since_change": [0, 1, 0, 1, 2],
                "active": [True] * 5,
            }).set_index("stimulus_presentations_id"),
            # block_offset
            5,
            # frame_offset
            20,
            # expected
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 1., 2., 1., 1.],
                "end_frame": [21, 23, 25, 27, 29],
                "image_name": ["a", "a", "b", "b", "b"],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [20, 22, 24, 26, 28],
                "start_time": [31., 34., 38., 43., 46.],
                "stop_time": [32., 35., 40., 44., 47.],
                "stimulus_block": [5] * 5,
                "stimulus_name": ["test_image_set"] * 5,
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False],
                "flashes_since_change": [0, 1, 0, 1, 2],
                "active": [False] * 5,
            }).set_index("stimulus_presentations_id"),
        ),
    ],
    indirect=["mock_replay_pkl_fixture", "mock_sync_dataset_fixture"]
)
@pytest.mark.skip(reason="this test needs to be updated")
def test_generate_replay_stim_table(
    monkeypatch, mock_replay_pkl_fixture,
    mock_sync_dataset_fixture, stim_start, stim_end,
    behavior_stim_table, block_offset, frame_offset, expected
):

    # Set up mocks that need to be patched in
    mock_check_behavior_and_replay_pkl_match = create_autospec(
        create_stim_table.check_behavior_and_replay_pkl_match
    )

    with monkeypatch.context() as m:
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.check_behavior_and_replay_pkl_match",
            mock_check_behavior_and_replay_pkl_match
        )

        obt = create_stim_table.generate_replay_stim_table(
            mock_replay_pkl_fixture,
            mock_sync_dataset_fixture,
            stim_start,
            stim_end,
            behavior_stim_table,
            block_offset,
            frame_offset
        )

    mock_check_behavior_and_replay_pkl_match.assert_called_once_with(
        replay_pkl=mock_replay_pkl_fixture,
        behavior_stim_table=behavior_stim_table
    )

    pd.testing.assert_frame_equal(expected, obt)


@pytest.fixture()
def mock_mapping_pkl_fixture(request):

    mock_mapping_pkl = create_autospec(CamStimOnePickleStimFile, instance=True)
    mock_mapping_pkl.pre_blank_sec = request.param.get("pre_blank_sec", 5)
    mock_mapping_pkl.frames_per_second = request.param.get(
        "frames_per_second", 30)
    mock_mapping_pkl.num_frames = request.param.get("num_frames", 10)
    mock_mapping_pkl.stimuli = request.param.get("stimuli", [{}])
    return mock_mapping_pkl


@pytest.mark.parametrize(
    "mock_mapping_pkl_fixture, mock_sync_dataset_fixture,"
    "mock_create_stim_table_return, frame_start, frame_end, expected",
    [
        (
            # mock_mapping_pkl_fixture (just use defaults)
            {},
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim"],
                "get_falling_edges": {
                    "vsync_stim": np.array(
                        # Behavior frame times
                        [1., 2., 4., 6., 8., 10., 12., 14., 16., 17.]
                        # Mapping frame times
                        + [18., 19., 20., 21., 22., 23., 24., 25., 26., 27.]
                        # Replay frame times
                        + [31., 32., 34., 35., 38., 40., 43., 44., 46., 47.]
                    ),
                }
            },
            # mock_create_stim_table_return
            pd.DataFrame({
                "Start": [0.0, 4.0, 5.0, 7.0, 8.0],
                "End": [4.0, 5.0, 7.0, 8.0, 9.0],
                "stimulus_name": [np.nan, "gabor", "gabor", "flash", "flash"],
                "stimulus_block": [np.nan, 0.0, 0.0, 1.0, 1.0],
                "TF": [np.nan, 4.0, 4.0, np.nan, np.nan],
                "SF": [np.nan, 0.08, 0.08, np.nan, np.nan],
                "Ori": [np.nan, 45.0, 90.0, np.nan, np.nan],
                "Contrast": [np.nan, 0.8, 0.8, 0.8, 0.8],
                "Pos_x": [np.nan, -30.0, 20.0, np.nan, np.nan],
                "Pos_y": [np.nan, 0.0, 40.0, np.nan, np.nan],
                "stimulus_index": [np.nan, 0.0, 0.0, 1.0, 1.0],
                "Color": [np.nan, np.nan, np.nan, -1.0, 1.0]
            }),
            # frame_start
            10,
            # frame_end
            20,
            # expected
            pd.DataFrame({
                "start_frame": [10, 14, 15, 17, 18],
                "end_frame": [14, 15, 17, 18, 19],
                "stimulus_name": ["spontaneous", "gabor", "gabor", "flash", "flash"],  # noqa: E501
                "stimulus_block": [1, 2, 2, 3, 3],
                "temporal_frequency": [np.nan, 4.0, 4.0, np.nan, np.nan],
                "spatial_frequency": [np.nan, 0.08, 0.08, np.nan, np.nan],
                "orientation": [np.nan, 45.0, 90.0, np.nan, np.nan],
                "contrast": [np.nan, 0.8, 0.8, 0.8, 0.8],
                "position_x": [np.nan, -30.0, 20.0, np.nan, np.nan],
                "position_y": [np.nan, 0.0, 40.0, np.nan, np.nan],
                "stimulus_index": [np.nan, 0, 0, 1, 1],
                "color": [np.nan, np.nan, np.nan, -1.0, 1.0],
                "start_time": [18., 22., 23., 25., 26.],
                "stop_time": [22., 23., 25., 26., 27.],
                "duration": [4., 1., 2., 1., 1.],
                "active": [False, False, False, False, False],
            })
        ),
        (
            # mock_mapping_pkl_fixture (just use defaults)
            {},
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim"],
                "get_falling_edges": {
                    "vsync_stim": np.array(
                        # Behavior frame times
                        [1., 2., 4., 6., 8., 10., 12., 14., 16., 17.]
                        # Mapping frame times
                        + [18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28.]  # noqa: E501
                        # Replay frame times
                        + [31., 32., 34., 35., 38., 40., 43., 44., 46., 47.]
                    ),
                }
            },
            # mock_create_stim_table_return
            pd.DataFrame({
                "Start": [0.0, 4.0, 5.0, 7.0, 8.0, 9.0],
                "End": [4.0, 5.0, 7.0, 8.0, 9.0, 10.0],
                "stimulus_name": [np.nan, "gabor", "gabor", "flash", "flash", np.nan],  # noqa: E501
                "stimulus_block": [np.nan, 0.0, 0.0, 1.0, 1.0, np.nan],
                "TF": [np.nan, 4.0, 4.0, np.nan, np.nan, np.nan],
                "SF": [np.nan, 0.08, 0.08, np.nan, np.nan, np.nan],
                "Ori": [np.nan, 45.0, 90.0, np.nan, np.nan, np.nan],
                "Contrast": [np.nan, 0.8, 0.8, 0.8, 0.8, np.nan],
                "Pos_x": [np.nan, -30.0, 20.0, np.nan, np.nan, np.nan],
                "Pos_y": [np.nan, 0.0, 40.0, np.nan, np.nan, np.nan],
                "stimulus_index": [np.nan, 0.0, 0.0, 1.0, 1.0, np.nan],
                "Color": [np.nan, np.nan, np.nan, -1.0, 1.0, np.nan]
            }),
            # frame_start
            10,
            # frame_end
            20,
            # expected
            pd.DataFrame({
                "start_frame": [10, 14, 15, 17, 18, 19],
                "end_frame": [14, 15, 17, 18, 19, 20],
                "stimulus_name": ["spontaneous", "gabor", "gabor", "flash", "flash", "spontaneous"],  # noqa: E501
                "stimulus_block": [1, 2, 2, 3, 3, 4],
                "temporal_frequency": [np.nan, 4.0, 4.0, np.nan, np.nan, np.nan],  # noqa: E501
                "spatial_frequency": [np.nan, 0.08, 0.08, np.nan, np.nan, np.nan],  # noqa: E501
                "orientation": [np.nan, 45.0, 90.0, np.nan, np.nan, np.nan],
                "contrast": [np.nan, 0.8, 0.8, 0.8, 0.8, np.nan],
                "position_x": [np.nan, -30.0, 20.0, np.nan, np.nan, np.nan],
                "position_y": [np.nan, 0.0, 40.0, np.nan, np.nan, np.nan],
                "stimulus_index": [np.nan, 0, 0, 1, 1, np.nan],
                "color": [np.nan, np.nan, np.nan, -1.0, 1.0, np.nan],
                "start_time": [18., 22., 23., 25., 26., 27.],
                "stop_time": [22., 23., 25., 26., 27., 28.],
                "duration": [4., 1., 2., 1., 1., 1.],
                "active": [False, False, False, False, False, False],
            })
        )
    ],
    indirect=["mock_mapping_pkl_fixture", "mock_sync_dataset_fixture"]
)
@pytest.mark.skip(reason="this test needs to be updated")
def test_generate_mapping_stim_table(
    monkeypatch, mock_mapping_pkl_fixture, mock_sync_dataset_fixture,
    mock_create_stim_table_return, frame_start, frame_end, expected
):

    # Set up mocks that need to be patched in
    mock_create_stim_table = create_autospec(
        create_stim_table.create_stim_table
    )
    mock_create_stim_table.return_value = mock_create_stim_table_return

    with monkeypatch.context() as m:
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.create_stim_table",
            mock_create_stim_table
        )

        obt = create_stim_table.generate_mapping_stim_table(
            mock_mapping_pkl_fixture,
            mock_sync_dataset_fixture,
            frame_start,
            frame_end
        )

    pd.testing.assert_frame_equal(expected, obt)


@pytest.mark.parametrize(
    "mock_sync_dataset_fixture, mock_behavior_pkl_fixture, "
    "mock_mapping_pkl_fixture, mock_replay_pkl_fixture, "
    "frame_offsets, behavior_stim_table, mapping_stim_table, "
    "replay_stim_table, expected",
    [
        (
            # mock_sync_dataset_fixture (can just use default)
            {},
            # mock_behavior_pkl_fixture
            {"num_frames": 10},
            # mock_mapping_pkl_fixture
            {"num_frames": 10},
            # mock_replay_pkl_fixture
            {"num_frames": 10},
            # frame_offsets
            [10, 10, 10],
            # behavior_stim_table
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 2., 2., 2., 1.],
                "end_frame": [1, 3, 5, 7, 9],
                "image_name": ["a", "a", "b", "b", "b"],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [0, 2, 4, 6, 8],
                "start_time": [1., 4., 8., 12., 16.],
                "stop_time": [2., 5., 10., 14., 17.],
                "stimulus_block": [0] * 5,
                "stimulus_name": ["test_image_set"] * 5,
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False],
                "flashes_since_change": [0, 1, 0, 1, 2],
                "active": [True] * 5,
            }).set_index("stimulus_presentations_id"),
            # mapping_stim_table
            pd.DataFrame({
                "start_frame": [10, 14, 15, 17, 18],
                "end_frame": [14, 15, 17, 18, 19],
                "stimulus_name": ["spontaneous", "gabor", "gabor", "flash", "spontaneous"],  # noqa: E501
                "stimulus_block": [1, 2, 2, 3, 4],
                "temporal_frequency": [np.nan, 4.0, 4.0, np.nan, np.nan],
                "spatial_frequency": [np.nan, 0.08, 0.08, np.nan, np.nan],
                "orientation": [np.nan, 45.0, 90.0, np.nan, np.nan],
                "contrast": [np.nan, 0.8, 0.8, 0.8, np.nan],
                "position_x": [np.nan, -30.0, 20.0, np.nan, np.nan],
                "position_y": [np.nan, 0.0, 40.0, np.nan, np.nan],
                "stimulus_index": [np.nan, 0, 0, 1, np.nan],
                "color": [np.nan, np.nan, np.nan, -1.0, np.nan],
                "start_time": [18., 22., 23., 25., 26.],
                "stop_time": [22., 23., 25., 26., 27.],
                "duration": [4., 1., 2., 1., 1.],
                "active": [False, False, False, False, False],
            }),
            # replay_stim_table
            pd.DataFrame({
                "stimulus_presentations_id": [0, 1, 2, 3, 4],
                "duration": [1., 1., 2., 1., 1.],
                "end_frame": [21, 23, 25, 27, 29],
                "image_name": ["a", "a", "b", "b", "b"],
                "omitted": [False, False, False, False, False],
                "orientation": [np.nan] * 5,
                "start_frame": [20, 22, 24, 26, 28],
                "start_time": [31., 34., 38., 43., 46.],
                "stop_time": [32., 35., 40., 44., 47.],
                "stimulus_block": [5] * 5,
                "stimulus_name": ["test_image_set"] * 5,
                "is_change": [False, False, True, False, False],
                "rewarded": [False, False, True, False, False],
                "flashes_since_change": [0, 1, 0, 1, 2],
                "active": [False] * 5,
            }).set_index("stimulus_presentations_id"),
            # expected
            pd.DataFrame({
                "stimulus_block": [0] * 5 + [1, 2, 2, 3, 4] + [5] * 5,
                "active": [True] * 5 + [False] * 10,
                "stimulus_name": (
                    ["test_image_set"] * 5
                    + ["spontaneous", "gabor", "gabor", "flash", "spontaneous"]
                    + ["test_image_set"] * 5
                ),
                "start_time": [
                    1., 4., 8., 12., 16.,
                    18., 22., 23., 25., 26.,
                    31., 34., 38., 43., 46.
                ],
                "stop_time": [
                    2., 5., 10., 14., 17.,
                    22., 23., 25., 26., 27.,
                    32., 35., 40., 44., 47.
                ],
                "duration": [
                    1., 2., 2., 2., 1.,
                    4., 1., 2., 1., 1.,
                    1., 1., 2., 1., 1.
                ],
                "start_frame": [
                    0, 2, 4, 6, 8,
                    10, 14, 15, 17, 18,
                    20, 22, 24, 26, 28
                ],
                "end_frame": [
                    1, 3, 5, 7, 9,
                    14, 15, 17, 18, 19,
                    21, 23, 25, 27, 29
                ],
                "flashes_since_change": [
                    0, 1, 0, 1, 2,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    0, 1, 0, 1, 2
                ],
                "image_name": [
                    "a", "a", "b", "b", "b",
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    "a", "a", "b", "b", "b",
                ],
                "is_change": [
                    False, False, True, False, False,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    False, False, True, False, False
                ],
                "omitted": [
                    False, False, False, False, False,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    False, False, False, False, False,
                ],
                "orientation": (
                    [np.nan] * 5
                    + [np.nan, 45.0, 90.0, np.nan, np.nan]
                    + [np.nan] * 5
                ),
                "rewarded": [
                    False, False, True, False, False,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    False, False, True, False, False,
                ],
                "temporal_frequency": (
                    [np.nan] * 5
                    + [np.nan, 4.0, 4.0, np.nan, np.nan]
                    + [np.nan] * 5
                ),
                "spatial_frequency": (
                    [np.nan] * 5
                    + [np.nan, 0.08, 0.08, np.nan, np.nan]
                    + [np.nan] * 5
                ),
                "contrast": (
                    [np.nan] * 5
                    + [np.nan, 0.8, 0.8, 0.8, np.nan]
                    + [np.nan] * 5
                ),
                "position_x": (
                    [np.nan] * 5
                    + [np.nan, -30.0, 20.0, np.nan, np.nan]
                    + [np.nan] * 5
                ),
                "position_y": (
                    [np.nan] * 5
                    + [np.nan, 0.0, 40.0, np.nan, np.nan]
                    + [np.nan] * 5
                ),
                "stimulus_index": (
                    [np.nan] * 5
                    + [np.nan, 0, 0, 1, np.nan]
                    + [np.nan] * 5
                ),
                "color": (
                    [np.nan] * 5
                    + [np.nan, np.nan, np.nan, -1.0, np.nan]
                    + [np.nan] * 5
                ),
            }),
        )
    ],
    indirect=[
        "mock_sync_dataset_fixture",
        "mock_behavior_pkl_fixture",
        "mock_mapping_pkl_fixture",
        "mock_replay_pkl_fixture"
    ]
)
@pytest.mark.skip(reason="this test needs to be updated")
def test_create_vbn_stimulus_table(
    monkeypatch, mock_sync_dataset_fixture, mock_behavior_pkl_fixture,
    mock_mapping_pkl_fixture, mock_replay_pkl_fixture, frame_offsets,
    behavior_stim_table, mapping_stim_table, replay_stim_table, expected
):

    # Set up mocks that need to be patched in
    mock_get_frame_offsets = create_autospec(
        create_stim_table.get_frame_offsets
    )
    mock_get_frame_offsets.return_value = frame_offsets

    mock_generate_behavior_stim_table = create_autospec(
        create_stim_table.generate_behavior_stim_table
    )
    mock_generate_behavior_stim_table.return_value = behavior_stim_table

    mock_generate_mapping_stim_table = create_autospec(
        create_stim_table.generate_mapping_stim_table
    )
    mock_generate_mapping_stim_table.return_value = mapping_stim_table

    mock_generate_replay_stim_table = create_autospec(
        create_stim_table.generate_replay_stim_table
    )
    mock_generate_replay_stim_table.return_value = replay_stim_table

    with monkeypatch.context() as m:
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.get_frame_offsets",
            mock_get_frame_offsets
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.generate_behavior_stim_table",
            mock_generate_behavior_stim_table
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.generate_mapping_stim_table",
            mock_generate_mapping_stim_table
        )
        m.setattr(
            "ecephys_etl.modules.vbn_create_stimulus_table"
            ".create_stim_table.generate_replay_stim_table",
            mock_generate_replay_stim_table
        )

        obt = create_stim_table.create_vbn_stimulus_table(
            mock_sync_dataset_fixture,
            mock_behavior_pkl_fixture,
            mock_mapping_pkl_fixture,
            mock_replay_pkl_fixture
        )

    pd.testing.assert_frame_equal(expected, obt)
