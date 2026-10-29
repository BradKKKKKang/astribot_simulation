"""Exercise viewer lifecycle with a real thread and mocked ROS/GUI dependencies."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


def viewer_class(namespace):
    path = Path(__file__).with_name('test_camera_viewer.py')
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CameraViewer']


def test_viewer_signals_do_not_shutdown_context_before_spin_thread_finishes():
    calls, spinning, release = [], threading.Event(), threading.Event()
    context = {'ok': False}
    no_signals = object()
    def init(**kwargs):
        assert kwargs['signal_handler_options'] is no_signals
        context['ok'] = True
    def spin_once(**kwargs):
        spinning.set()
        release.wait(timeout=kwargs['timeout_sec'])
        assert context['ok'], 'context destroyed while constructing the wait set'
    def shutdown(**kwargs):
        assert not viewer.spin_thread.is_alive()
        calls.append('executor')
        return True
    def shutdown_context():
        calls.append('context')
        context['ok'] = False
    namespace = {'ros_version': '2', 'threading': threading, 'CvBridge': object,
        'rclpy': SimpleNamespace(init=init, ok=lambda: context['ok'], shutdown=shutdown_context),
        'SignalHandlerOptions': SimpleNamespace(NO=no_signals),
        'Node': lambda name: SimpleNamespace(destroy_node=lambda: calls.append('node')),
        'MultiThreadedExecutor': lambda: SimpleNamespace(add_node=lambda node: None,
            spin_once=spin_once, shutdown=shutdown),
        'cv2': SimpleNamespace(destroyAllWindows=lambda: calls.append('windows'))}
    cls = viewer_class(namespace)
    cls.setup_ros2_subscribers = lambda self: None
    viewer = cls([])
    assert spinning.wait(timeout=1.)
    # SIGTERM's handler now only sets this event; ROS remains valid.
    viewer.stop_event.set()
    release.set()
    viewer.close()
    assert calls == ['windows', 'executor', 'node', 'context']


def test_viewer_does_not_destroy_ros_context_when_thread_cannot_stop():
    calls = []
    cls = viewer_class({'ros_version': '2', 'cv2': SimpleNamespace(destroyAllWindows=lambda: None)})
    viewer = object.__new__(cls)
    viewer.stop_event = threading.Event()
    viewer.spin_thread = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: True)
    viewer.executor = SimpleNamespace(shutdown=lambda **kw: calls.append('executor'))
    viewer.node = SimpleNamespace(destroy_node=lambda: calls.append('node'))
    with pytest.raises(RuntimeError, match='spin thread did not stop'):
        viewer.close()
    assert not calls
