"""CPU-only USD visual export. Requires pxr (usd-core), not Isaac Sim/GPU.

In the Isaac environment, bundled USD libraries can be used without starting
SimulationApp; see docs/DATASET_REPLAY.md for the exact CPU-only invocation.
"""
import json
from pathlib import Path

import numpy as np

from .recorder import write_json
from .usd_export import export_robot_glb
from .reference_pose import sdk_joint_positions, usd_reference_snapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = ROOT / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1_textured.usda'
DEFAULT_POSE_CONFIG = ROOT / 'config/placement_reference_pose.json'
DEFAULT_DEMO_OUTPUT = ROOT / 'web/placement-viewer/public/data/demo'


def open_robot_stage(usd):
    """Open the same fixed-base S1 USD for both geometry and event extraction."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(Path(usd).expanduser().resolve(strict=True)))
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError('S1 export requires a Z-up, meter stage')
    root = stage.GetPrimAtPath('/World/astribot_whole_body')
    if not root:
        raise ValueError('USD must contain /World/astribot_whole_body')
    names = [p.GetName() for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not names:
        raise ValueError('USD robot must contain rigid bodies')
    return stage, root, names


def export_reference_robot(output=DEFAULT_DEMO_OUTPUT, *, usd=DEFAULT_USD,
                           pose_config=DEFAULT_POSE_CONFIG):
    """Regenerate the persistent Web S1 geometry and SDK 103 reference pose."""
    stage, root, names = open_robot_stage(usd)
    settings = json.loads(Path(pose_config).expanduser().read_text())
    snapshot, joint_names = usd_reference_snapshot(stage, root.GetPath(), names, sdk_joint_positions(settings))
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    info = export_robot_glb(stage, root.GetPath(), names, output / 'robot.glb')
    write_json(output / 'model.json', {
        'body_names': names, 'base_body': 'astribot_torso_base',
        'reference_snapshot': snapshot, 'joint_names': joint_names,
        'reference_pose': settings, 'geometry': info,
    })
    print(f'CPU export: {len(names)} bodies, {info["triangle_count"]:,} triangles, {info["bytes"] / 1e6:.1f} MB')
    return info
