"""Verify lens direction, coefficient order, and the Isaac RGB output path."""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.simu_utils.camera_distortion import CameraDistortion
from src.simulation_config.isaac_scene_config import CAMERA_BODIES, camera_prim_path


ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = [0.00268249, -0.0404815, 8.09985e-06, -0.000577883, 0.027629]


def intrinsic_matrix(width=320, height=180):
    focal = 747.732499 * width / 1280
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])


def coordinate_image(width=320, height=180):
    y, x = np.indices((height, width), dtype=np.float32)
    return np.stack((x, y), axis=-1)


@pytest.mark.parametrize('coefficients', [COEFFICIENTS, [0, 0, 0.003, -0.006, 0]])
def test_warp_matches_forward_radial_and_tangential_projection(coefficients):
    matrix = intrinsic_matrix()
    image = coordinate_image()
    output = CameraDistortion(coefficients).apply(image, matrix)
    # Ramps encode the sampled ideal pixel. Project those samples forward with
    # an independent Brown-Conrady formula; they must land at the output pixel.
    sampled = output[12:-12:9, 12:-12:9]
    x = (sampled[..., 0] - matrix[0, 2]) / matrix[0, 0]
    y = (sampled[..., 1] - matrix[1, 2]) / matrix[1, 1]
    k1, k2, p1, p2, k3 = coefficients
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    projected = np.stack((xd*matrix[0, 0] + matrix[0, 2],
                          yd*matrix[1, 1] + matrix[1, 2]), axis=-1)
    np.testing.assert_allclose(projected, image[12:-12:9, 12:-12:9], atol=0.025)
    assert output.shape == image.shape and output.dtype == image.dtype


def test_zero_distortion_preserves_pixels_without_resampling():
    image = np.random.default_rng(0).integers(0, 256, (180, 320, 3), dtype=np.uint8)
    assert CameraDistortion([0]*5).apply(image, intrinsic_matrix()) is image


def test_remap_cache_tracks_resolution_and_intrinsics(monkeypatch):
    import src.simu_utils.camera_distortion as module

    calls = []
    original = module.cv2.undistortPointsIter
    def record(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module.cv2, 'undistortPointsIter', record)
    lens = CameraDistortion(COEFFICIENTS)
    image = coordinate_image()
    matrix = intrinsic_matrix()
    first = lens.apply(image, matrix)
    np.testing.assert_array_equal(lens.apply(image, matrix.copy()), first)
    assert len(calls) == 1
    changed = matrix.copy()
    changed[0, 0] *= 1.1
    assert not np.array_equal(lens.apply(image, changed), first)
    assert len(calls) == 2
    assert lens.apply(coordinate_image(160, 90), intrinsic_matrix(160, 90)).shape == (90, 160, 2)
    assert len(calls) == 3


@pytest.mark.parametrize('coefficients', [[0]*4, [0, 0, float('nan'), 0, 0], None])
def test_invalid_coefficients_fail(coefficients):
    with pytest.raises(ValueError, match='finite'):
        CameraDistortion(coefficients)


def isaac_methods():
    tree = ast.parse((ROOT / 'src/astribot_envs/astribot_isaaclab_env.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'AstribotIsaacLabEnv')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_setup_camera_distortion', 'render_single_camera')]
    namespace = {'np': np, 'camera_prim_path': camera_prim_path}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<camera-methods>', 'exec'), namespace)
    return namespace


def test_usd_calibration_reaches_head_rgb_output_only(monkeypatch):
    import sys
    Usd = pytest.importorskip('pxr.Usd')
    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim('/World/envs/env_0/Robot')
    robot.GetReferences().AddReference(str(
        ROOT / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1.usda'), '/World')
    omni = ModuleType('omni')
    omni.usd = ModuleType('omni.usd')
    omni.usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    monkeypatch.setitem(sys.modules, 'omni', omni)
    monkeypatch.setitem(sys.modules, 'omni.usd', omni.usd)
    methods = isaac_methods()
    env = SimpleNamespace(camera_names=list(CAMERA_BODIES), camera_bodies=CAMERA_BODIES,
                          robot_root_path='/World/astribot_whole_body')
    methods['_setup_camera_distortion'](env)
    assert list(env._camera_distortions) == ['head_rgbd']
    np.testing.assert_array_equal(env._camera_distortions['head_rgbd'].coefficients, COEFFICIENTS)

    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    image = coordinate_image()
    data = SimpleNamespace(output={'rgb': [Tensor(image)]},
                           intrinsic_matrices=[Tensor(intrinsic_matrix())])
    env.scene = {name: SimpleNamespace(data=data) for name in CAMERA_BODIES}
    render = methods['render_single_camera']
    head = render(env, 'head_rgbd')['rgb_img']
    assert not np.array_equal(head, image)
    np.testing.assert_array_equal(render(env, 'head')['rgb_img'], head)
    for name in ('left_wrist_rgbd', 'right_wrist_rgbd'):
        np.testing.assert_array_equal(render(env, name)['rgb_img'], image)
    assert render(env, 'unknown') is None
