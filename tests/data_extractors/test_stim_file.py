import pickle
import operator as op
import os
from datetime import datetime

import pytest
import numpy as np

from ecephys_etl.data_extractors import stim_file


# ideally these would be fixtures, but I want to parametrize over them
def stim_pkl_data():
    return {
        'fps': 1000,
        'pre_blank_sec': 20,
        'stimuli': [{'a': 1}, {'a': 1}],
        'items': {
            'foraging': {
                'encoders': [
                    {
                        'dx': [1, 2, 3]
                    }
                ]
            }
        },
        "intervalsms": [1, 1, 1],
    }


def stim_pkl_data_toplevel_dx():
    return {
        'fps': 1000,
        'pre_blank_sec': 20,
        'dx': [1, 2, 3],
        'stimuli': [{'a': 1}, {'a': 1}],
        'items': {
            'foraging': {
                'encoders': []
            }
        },
        "intervalsms": [1, 1, 1],
    }


def behavior_pkl_data():
    return {
        "items": {
            "behavior": {
                "intervalsms": [1, 1, 1, 1],
                "rewards": [{"reward_times": np.array([[1.1, 5], [1.2, 8]])}],
                "params": {
                    "stimulus": {
                        "params": {
                            "image_set": (
                                "//allen/programs/braintv/workgroups/nc-ophys/"
                                "visual_behavior/image_dictionaries/"
                                "Natural_Images_Lum_Matched_set_ophys_G_2019"
                                ".05.26.pkl"
                            )
                        }
                    }
                }
            }
        }
    }


def replay_pkl_data():
    return {
        "intervalsms": [1, 1, 1],
        "stimuli": [
            {"sweep_params": {"ReplaceImage": (['a', 'b', None, 'a'],)}}
        ]
    }


@pytest.fixture(params=[stim_pkl_data, stim_pkl_data_toplevel_dx])
def pkl_on_disk(tmpdir_factory, request):
    time = datetime.now().strftime("%H-%M-%S")
    tmpdir = str(tmpdir_factory.mktemp('pkl_files'))
    file_path = os.path.join(tmpdir, f'test_{time}.pkl')

    with open(file_path, 'wb') as pkl_file:
        pickle.dump(request.param(), pkl_file)

    return file_path


@pytest.fixture
def camstimone_pickle_stim_file(pkl_on_disk):
    return stim_file.CamStimOnePickleStimFile.factory(pkl_on_disk)


@pytest.mark.parametrize('prop_name, expected, comp', [
    ['presentation_intervals', [1, 1, 1], np.allclose],
    ['num_frames', 4, op.eq],
    ['frames_per_second', 1000, op.eq],
    ['pre_blank_sec', 20, op.eq],
    ['angular_wheel_rotation', [1, 2, 3], np.allclose],
    ['angular_wheel_velocity', [1000, 2000, 3000], np.allclose]
])
def test_camstim_one_pickle_stim_file_properties(
    camstimone_pickle_stim_file, prop_name, expected, comp
):
    obtained = getattr(camstimone_pickle_stim_file, prop_name)
    assert comp(obtained, expected)


@pytest.mark.parametrize("pkl_on_disk, prop_name, expected, comp", [
    [behavior_pkl_data, "presentation_intervals", [1, 1, 1, 1], np.allclose],
    [behavior_pkl_data, "num_frames", 5, op.eq],
    [behavior_pkl_data, "image_set", "Natural_Images_Lum_Matched_set_ophys_G_2019", op.eq],  # noqa: E501
    [behavior_pkl_data, "reward_frames", [5, 8], np.allclose]
], indirect=["pkl_on_disk"])
def test_behavior_pickle_file_properties(
    pkl_on_disk, prop_name, expected, comp
):
    behavior_pickle_file = stim_file.BehaviorPickleFile.factory(
        pkl_on_disk
    )
    obt = getattr(behavior_pickle_file, prop_name)
    assert comp(obt, expected)


@pytest.mark.parametrize("pkl_on_disk, prop_name, expected, comp", [
    [replay_pkl_data, "presentation_intervals", [1, 1, 1], np.allclose],
    [replay_pkl_data, "num_frames", 4, op.eq],
    [replay_pkl_data, "image_presentations", ["a", "b", None, "a"], op.eq],
    [replay_pkl_data, "unique_image_names", ["a", "b"], op.eq]
], indirect=["pkl_on_disk"])
def test_replay_pickle_file_properties(
    pkl_on_disk, prop_name, expected, comp
):
    replay_pickle_file = stim_file.ReplayPickleFile.factory(
        pkl_on_disk
    )
    obt = getattr(replay_pickle_file, prop_name)
    assert comp(obt, expected)
