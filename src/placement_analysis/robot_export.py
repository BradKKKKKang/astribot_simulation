"""CPU-only USD visual export. Requires pxr (usd-core), not Isaac Sim/GPU.

In the Isaac environment, bundled USD libraries can be used without starting
SimulationApp. Model-only export requires an explicit --pose-config.
"""
import json
from pathlib import Path

import numpy as np

from .recorder import write_json
from .usd_export import export_robot_glb
from .reference_pose import sdk_joint_positions, usd_reference_snapshot

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = ROOT / 'astribot_descriptions/usd/astribot_s1_usd/astribot_s1_textured.usda'
DEFAULT_POSE_CONFIG = None  # The old reference-pose file was removed; require an explicit input.
DEFAULT_DEMO_OUTPUT = ROOT / 'web/placement-viewer/public/data/demo'


def validate_usd_composition(stage):
    """Reject missing geometry composition, including USD 22's layer-only errors."""
    from pxr import Sdf
    if stage is None:
        raise ValueError('USD geometry has unresolved composition dependencies')
    errors = getattr(stage, 'GetCompositionErrors', lambda: [])()
    if errors or any(prim.GetPrimIndex().localErrors for prim in stage.Traverse()):
        raise ValueError('USD geometry has unresolved composition dependencies')
    for layer in stage.GetUsedLayers():
        for path in layer.subLayerPaths:
            if Sdf.Layer.FindOrOpenRelativeToLayer(layer, path) is None:
                raise ValueError('USD geometry has unresolved composition sublayers')


def open_robot_stage(usd, *, root_path='/World/astribot_whole_body'):
    """Open the same fixed-base S1 USD for both geometry and event extraction."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(Path(usd).expanduser().resolve(strict=True)))
    validate_usd_composition(stage)
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError('S1 export requires a Z-up, meter stage')
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f'USD must contain {root_path}')
    names = [p.GetName() for p in root.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not names:
        raise ValueError('USD robot must contain rigid bodies')
    return stage, root, names


def export_reference_robot(output=DEFAULT_DEMO_OUTPUT, *, usd=DEFAULT_USD,
                           pose_config=DEFAULT_POSE_CONFIG):
    """Regenerate the persistent Web S1 geometry and SDK 103 reference pose."""
    if pose_config is None:
        raise ValueError('model-only export requires --pose-config with reference joint targets')
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


def geometry_cache_key(usd, body_names, *, root_path=None, stage=None):
    """Hash composed USD layers consumed by the neutral-material geometry exporter.

    Shader modules and textures are asset-valued attributes, not geometry
    composition dependencies. The exporter never reads them. Missing sublayers,
    references or payloads remain fatal because they can remove visual geometry
    or change body transforms even when a partial stage can still be opened.
    """
    import hashlib
    from pxr import Usd
    from .artifact_store import json_bytes

    usd = Path(usd).expanduser().resolve(strict=True)
    stage = stage or Usd.Stage.Open(str(usd))
    validate_usd_composition(stage)
    if stage.GetRootLayer().realPath and Path(stage.GetRootLayer().realPath).resolve() != usd:
        raise ValueError('geometry stage and USD source differ')
    dependencies = []
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            if not layer.empty:
                # Session opinions can affect transforms/visibility. Empty
                # anonymous session identifiers are process-local, not inputs.
                dependencies.append({'anonymous_layer_sha256': hashlib.sha256(
                    layer.ExportToString().encode()).hexdigest()})
            continue
        if layer.dirty:
            raise ValueError('geometry cache requires saved USD composition layers')
        path = Path(layer.realPath).resolve(strict=True)
        dependencies.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    dependencies.sort(key=lambda item: json_bytes(item))
    exporter = Path(__file__).with_name('usd_export.py').read_bytes()
    return hashlib.sha256(json_bytes({'export_version': 2,
        'exporter_sha256': hashlib.sha256(exporter).hexdigest(),
        'dependencies': dependencies, 'body_names': list(body_names), 'root_path': str(root_path)})).hexdigest()


def cached_robot_geometry(stage, root_path, body_names, store, *, usd=None):
    """Export once per dependency closure; return a shared immutable GLB reference."""
    import fcntl
    import tempfile
    from .artifact_store import atomic_write, json_bytes

    usd = usd or stage.GetRootLayer().realPath
    key = geometry_cache_key(usd, body_names, root_path=root_path, stage=stage)
    cache = store.root / 'geometry-cache'
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / f'{key}.json'
    with (cache / f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if entry.is_file():
            ref = json.loads(entry.read_text())
            store.resolve(ref)
            return ref
        with tempfile.TemporaryDirectory(prefix='.export-', dir=cache) as temporary:
            destination = Path(temporary) / 'robot.glb'
            export_robot_glb(stage, root_path, body_names, destination)
            ref = store.put_file(destination)
        atomic_write(entry, json_bytes(ref))
        return ref
