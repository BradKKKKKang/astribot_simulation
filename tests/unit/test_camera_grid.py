"""Combined camera layout and single-window behavior without ROS or a desktop."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.unit.test_camera_viewer_lifecycle import viewer_class

PATH = Path(__file__).with_name('test_camera_viewer.py')
tree = ast.parse(PATH.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compose_camera_grid')
namespace = {'np': np, 'cv2': cv2}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATH), 'exec'), namespace)
compose = namespace['compose_camera_grid']
NAMES = ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd']


def test_three_cameras_keep_native_pixels_and_anatomical_order():
    images = {NAMES[2]: np.full((360, 640, 3), (30, 40, 50), np.uint8),
              NAMES[0]: np.full((720, 1280, 3), (60, 70, 80), np.uint8),
              NAMES[1]: np.full((360, 640, 3), (90, 100, 110), np.uint8)}
    combined = compose(images, NAMES[::-1])
    assert combined.shape == (1080, 1280, 3)
    np.testing.assert_array_equal(combined[30:720], images[NAMES[0]][30:])
    np.testing.assert_array_equal(combined[750:, :640], images[NAMES[1]][30:])
    np.testing.assert_array_equal(combined[750:, 640:], images[NAMES[2]][30:])
    assert images[NAMES[0]][0, 0].tolist() == [60, 70, 80]  # Labels never mutate source frames.


def test_partial_arrival_keeps_slots_and_other_camera_sizes_are_letterboxed():
    partial = compose({NAMES[2]: np.full((360, 640, 3), 99, np.uint8)}, NAMES)
    assert partial.shape == (1080, 1280, 3)
    assert not partial[30:720].any() and not partial[750:, :640].any()
    assert (partial[750:, 640:] == 99).all()
    images = {NAMES[0]: np.full((240, 320, 3), 50, np.uint8),
              NAMES[1]: np.full((100, 100, 3), 80, np.uint8),
              NAMES[2]: np.full((90, 160, 3), 90, np.uint8)}
    combined = compose(images, NAMES)
    assert combined.shape == (400, 320, 3)
    assert not combined[-30:, 160:].any()
    assert compose({}, ['astribot_global_camera']).shape == (720, 1280, 3)


@pytest.mark.parametrize('quit_key', [ord('q'), -1])
def test_viewer_uses_one_window_and_cleans_up_on_quit_or_window_close(quit_key):
    calls = []
    gui = SimpleNamespace(WINDOW_NORMAL=0, WINDOW_GUI_NORMAL=0, WND_PROP_VISIBLE=0,
        namedWindow=lambda name, flags: calls.append(('window', name)),
        imshow=lambda name, image: calls.append(('image', name, image.shape)),
        resizeWindow=lambda *args: None, waitKey=lambda timeout: quit_key,
        getWindowProperty=lambda *args: 0)
    cls = viewer_class({'cv2': gui, 'compose_camera_grid': compose})
    viewer = object.__new__(cls)
    viewer.stop_event, viewer.update_lock = threading.Event(), threading.Lock()
    viewer.camera_names, viewer.camera_images = NAMES, {}
    viewer.close = lambda: calls.append(('close',))
    viewer.run()
    assert calls == [('window', 'Astribot Cameras'), ('image', 'Astribot Cameras', (1080, 1280, 3)), ('close',)]
