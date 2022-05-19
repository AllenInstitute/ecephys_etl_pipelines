import os
import numpy as np
import pandas as pd
import pytest

from ecephys_etl.data_transformers.visual_behavior_stimulus_processing import (
    get_stimulus_presentations, _get_stimulus_epoch, _get_draw_epochs,
    get_visual_stimuli_df
)


@pytest.fixture()
def behavior_stimuli_data_fixture(request):
    """
    This fixture mimicks the behavior experiment stimuli data logs and
    allows parameterization for testing
    """
    images_set_log = request.param.get("images_set_log", [
        ('Image', 'im065', 5.809, 0)])
    images_draw_log = request.param.get("images_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    grating_set_log = request.param.get("grating_set_log", [
        ('Ori', 90, 3.585, 0)
    ])
    grating_draw_log = request.param.get("grating_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    omitted_flash_frame_log = request.param.get("omitted_flash_frame_log", {
        "grating_0": []
    })
    grating_phase = request.param.get("grating_phase", None)
    grating_spatial_frequency = request.param.get("grating_spatial_frequency",
                                                  None)

    has_images = request.param.get("has_images", True)
    has_grating = request.param.get("has_grating", True)

    image_data = {
        "set_log": images_set_log,
        "draw_log": images_draw_log,
        "image_path": os.path.join('dummy_dir',
                                   'stimulus_template',
                                   'input',
                                   'test_image_set.pkl')
    }

    grating_data = {
        "set_log": grating_set_log,
        "draw_log": grating_draw_log,
        "phase": grating_phase,
        "sf": grating_spatial_frequency
    }

    data = {
        "items": {
            "behavior": {
                "stimuli": {},
                "omitted_flash_frame_log": omitted_flash_frame_log
            }
        }
    }

    if has_images:
        data["items"]["behavior"]["stimuli"]["images"] = image_data

    if has_grating:
        data["items"]["behavior"]["stimuli"]["grating"] = grating_data

    return data


@pytest.fixture()
def behavior_stimuli_time_fixture(request):
    """
    Fixture that allows for parameterization of behavior_stimuli stimuli
    time data.
    """
    timestamp_count = request.param["timestamp_count"]
    time_step = request.param["time_step"]

    timestamps = np.array([time_step * i for i in range(timestamp_count)])

    return timestamps


@pytest.mark.parametrize(
    "behavior_stimuli_data_fixture,current_set_ix,start_frame,"
    "n_frames,expected", [
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)
        ],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 0, 18, (0, 6)),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)
        ],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         2, 11, 18, (11, 18))
    ], indirect=["behavior_stimuli_data_fixture"]
)
def test_get_stimulus_epoch(behavior_stimuli_data_fixture,
                            current_set_ix, start_frame, n_frames, expected):
    items = behavior_stimuli_data_fixture["items"]
    log = items["behavior"]["stimuli"]["images"]["set_log"]
    actual = _get_stimulus_epoch(log, current_set_ix, start_frame, n_frames)
    assert actual == expected


@pytest.mark.parametrize(
    "behavior_stimuli_data_fixture,start_frame,stop_frame,expected,"
    "stimuli_type", [
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)
        ],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 6, [(1, 4)], 'images'),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)
        ],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 11, [(1, 4), (8, 11)], 'images'),
        ({'images_set_log': [
            ('Image', 'im065', 5.809955710916157, 0),
            ('Image', 'im061', 314.06612555068784, 6),
            ('Image', 'im062', 348.5941232265203, 12)
        ],
             'images_draw_log': ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 22, [(1, 4), (8, 11), (15, 18)], 'images'),
        ({"grating_set_log": [
            ("Ori", 90, 3.585, 0),
            ("Ori", 180, 40.847, 6),
            ("Ori", 270, 62.633, 12)
        ],
             "grating_draw_log": ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         0, 6, [(1, 4)], 'grating'),
        ({"grating_set_log": [
            ("Ori", 90.0, 3.585, 0),
            ("Ori", 180.0, 40.847, 6),
            ("Ori", 270.0, 62.633, 12)
        ],
             "grating_draw_log": ([0] + [1] * 3 + [0] * 3) * 3 + [0]},
         6, 11, [(8, 11)], 'grating')
    ], indirect=['behavior_stimuli_data_fixture']
)
def test_get_draw_epochs(behavior_stimuli_data_fixture,
                         start_frame, stop_frame, expected, stimuli_type):
    draw_log = (behavior_stimuli_data_fixture["items"]["behavior"]
    ["stimuli"][stimuli_type]["draw_log"])  # noqa: E128
    actual = _get_draw_epochs(draw_log, start_frame, stop_frame)
    assert actual == expected


@pytest.mark.parametrize("behavior_stimuli_time_fixture,"
                         "behavior_stimuli_data_fixture, "
                         "expected", [
                             ({"timestamp_count": 15, "time_step": 1},
                              {"images_set_log": [
                                  ('Image', 'im065', 5, 0),
                                  ('Image', 'im064', 25, 6)
                              ],
                                  "images_draw_log": (([0] * 2 + [1] * 2 +
                                                       [0] * 3) * 2 + [0]),
                                  "grating_set_log": [
                                      ("Ori", 90, 3.5, 0),
                                      ("Ori", 270, 15, 6)
                                  ],
                                  "grating_draw_log": (
                                          ([0] + [1] * 3 + [0] * 3)
                                          * 2 + [0])},
                              {"duration": [3.0, 2.0, 3.0, 2.0],
                               "end_frame": [4.0, 4.0, 11.0, 11.0],
                               "image_name": [np.NaN, 'im065', np.NaN,
                                              'im064'],
                               "index": [2, 0, 3, 1],
                               "omitted": [False, False, False, False],
                               "orientation": [90, np.NaN, 270, np.NaN],
                               "start_frame": [1.0, 2.0, 8.0, 9.0],
                               "start_time": [1, 2, 8, 9],
                               "stop_time": [4, 4, 11, 11]})
                         ], indirect=['behavior_stimuli_time_fixture',
                                      'behavior_stimuli_data_fixture'])
def test_get_stimulus_presentations(behavior_stimuli_time_fixture,
                                    behavior_stimuli_data_fixture,
                                    expected):
    presentations_df = get_stimulus_presentations(
        behavior_stimuli_data_fixture,
        behavior_stimuli_time_fixture)

    expected_df = pd.DataFrame.from_dict(expected)
    expected_df.index.name = 'stimulus_presentations_id'

    pd.testing.assert_frame_equal(presentations_df, expected_df)


@pytest.mark.parametrize("behavior_stimuli_time_fixture,"
                         "behavior_stimuli_data_fixture,"
                         "expected_data", [
                             ({"timestamp_count": 15, "time_step": 1},
                              {"images_set_log": [
                                  ('Image', 'im065', 5, 0),
                                  ('Image', 'im064', 25, 6)
                              ],
                                  "images_draw_log": (([0] * 2 + [1] * 2 +
                                                       [0] * 3) * 2 + [0]),
                                  "grating_set_log": [
                                      ("Ori", 90, 3.5, 0),
                                      ("Ori", 270, 15, 6)
                                  ],
                                  "grating_draw_log": (
                                          ([0] + [1] * 3 + [0] * 3)
                                          * 2 + [0])},
                              {"orientation": [90, None, 270, None],
                               "image_name": [None, 'im065', None, 'im064'],
                               "frame": [1.0, 2.0, 8.0, 9.0],
                               "end_frame": [4.0, 4.0, 11.0, 11.0],
                               "time": [1.0, 2.0, 8.0, 9.0],
                               "duration": [3.0, 2.0, 3.0, 2.0],
                               "omitted": [False, False, False, False]}),

                             # test case with images and a static grating
                             ({"timestamp_count": 30, "time_step": 1},
                              {"images_set_log": [
                                  ('Image', 'im065', 5, 0),
                                  ('Image', 'im064', 25, 6)
                              ],
                                  "images_draw_log": (([0] * 2 + [1] * 2 +
                                                       [0] * 3) * 2 + [
                                                          0] * 16),
                                  "grating_set_log": [
                                      ("Ori", 90, -1, 12),
                                      # -1 because that element is not used
                                      ("Ori", 270, -1, 24)
                                  ],
                                  "grating_draw_log": (
                                          [0] * 17 + [1] * 11 + [0, 0])},
                              {"orientation": [None, None, 90, 270],
                               "image_name": ['im065', 'im064', None, None],
                               "frame": [2.0, 9.0, 17.0, 24.0],
                               "end_frame": [4.0, 11.0, 24.0, 28.0],
                               "time": [2.0, 9.0, 17.0, 24.0],
                               "duration": [2.0, 2.0, 7.0, 4.0],
                               "omitted": [False, False, False, False]})
                         ],
                         indirect=["behavior_stimuli_time_fixture",
                                   "behavior_stimuli_data_fixture"])
def test_get_visual_stimuli_df(behavior_stimuli_time_fixture,
                               behavior_stimuli_data_fixture,
                               expected_data):
    stimuli_df = get_visual_stimuli_df(behavior_stimuli_data_fixture,
                                       behavior_stimuli_time_fixture)
    stimuli_df = stimuli_df.drop('index', axis=1)

    expected_df = pd.DataFrame.from_dict(expected_data)
    pd.testing.assert_frame_equal(stimuli_df, expected_df)
