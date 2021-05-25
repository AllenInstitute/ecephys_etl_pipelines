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


@pytest.mark.parametrize(
    "mock_line_labels, expected_line, falling_edges, warning",
    [
        # Basic tests
        (
            ["vsync_stim", "unrelated_line"],
            "vsync_stim",
            np.array([0, 1, 2, 3, 4]),
            None
        ),
        (
            ["stim_vsync", "unrelated_line"],
            "stim_vsync",
            np.array([5, 6, 7, 8, 9]),
            None
        ),
        # Test that get_vsyncs will grab first relevant line label
        (
            ["vsync_stim", "stim_vsync", "unrelated_line"],
            "vsync_stim",
            np.array([0, 1, 2, 3, 4]),
            None
        ),
        (
            ["stim_vsync", "vsync_stim", "unrelated_line"],
            "stim_vsync",
            np.array([0, 1, 2, 3, 4]),
            None
        ),
        # Test that fallback is used and warning is raised if no relevant
        # line label found.
        (
            ["unrelated_line"],
            2,
            np.array([10, 11, 12, 13, 14]),
            "Could not find 'vsync_stim' nor 'stim_vsync' line labels"
        )
    ]
)
def test_get_vsyncs(
    caplog, mock_line_labels, expected_line, falling_edges, warning
):
    mock_dataset = MagicMock()
    mock_dataset.line_labels = mock_line_labels
    mock_dataset.dfile.filename = "/tmp/dummy_sync_path.sync"
    mock_dataset.get_falling_edges.return_value = falling_edges

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
        "get_rising_edges", {"stim_running": np.array([1, 2, 3, 4, 5])}
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

    mock_sync_dataset = create_autospec(Dataset, instance=True)
    mock_sync_dataset.line_labels = mock_line_labels
    mock_sync_dataset.get_rising_edges.side_effect = mock_get_rising_edges
    mock_sync_dataset.get_falling_edges.side_effect = mock_get_falling_edges

    return mock_sync_dataset


@pytest.mark.parametrize(
    "mock_sync_dataset_fixture, frame_counts, tolerance, expected, "
    "warning, raises",
    [
        # Simple test case
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
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
            # expected frame offsets
            [0, 1, 2, 3, 4],
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
            # expected frame offsets
            [0, 1, 2, 3, 4],
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
                    "stim_running": np.array([0, 11, 22])
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
            # expected frame offsets
            [0, 5],
            # warning
            "Number of stim presentations obtained from sync",
            # raises
            False
        ),
        # Test when number of sync epochs don't match pkl case 2
        (
            # mock_sync_dataset_fixture
            {
                "line_labels": ["vsync_stim", "stim_running"],
                "get_rising_edges": {
                    "stim_running": np.array([0, 11, 22])
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
            # expected frame offsets
            [0, 3],
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
                    "stim_running": np.array([0, 11, 22])
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
            # expected frame offsets
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
                    "stim_running": np.array([0, 11, 22])
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
            # expected frame offsets
            [0, 5],
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
                    "stim_running": np.array([0, 11, 22])
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
            # expected frame offsets
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
    expected, warning, raises
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

        assert np.allclose(expected, obt)


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
def mock_replay_pkl_fixture(request):

    mock_replay_pkl = create_autospec(ReplayPickleFile, instance=True)
    mock_replay_pkl.image_presentations = request.param.get(
        "image_presentations", ["a", "a"] + [None] * 71 + ["b", "b", "b"]
    )
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
