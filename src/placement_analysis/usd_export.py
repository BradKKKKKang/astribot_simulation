"""Export Isaac USD visual meshes into a compact, per-body GLB.

All mesh coordinates stay in meters and the Isaac Z-up basis. The browser
sets Z-up explicitly. Nodes are independent body frames so measured snapshots
can be displayed without a second, potentially different kinematics model.
"""
import json
from pathlib import Path
import struct

import numpy as np


def export_robot_glb(stage, robot_prim_path, body_names, output):
    from pxr import Usd, UsdGeom

    if UsdGeom.GetStageUpAxis(stage) != 'Z' or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError('robot GLB export requires an Isaac Z-up, meter stage')
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root:
        raise ValueError(f'robot USD prim absent: {robot_prim_path}')
    cache = UsdGeom.XformCache()
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bodies = {p.GetName(): p for p in prims if p.GetName() in body_names
              and 'visuals' not in str(p.GetPath()) and 'collisions' not in str(p.GetPath())}
    missing = set(body_names) - bodies.keys()
    if missing:
        raise ValueError(f'robot body prims absent: {sorted(missing)}')
    gltf = {'asset': {'version': '2.0', 'generator': 'Astribot placement USD exporter'},
            'scene': 0, 'scenes': [{'nodes': list(range(len(body_names)))}],
            'nodes': [], 'meshes': [], 'materials': [], 'accessors': [], 'bufferViews': []}
    binary = bytearray()
    material_map = {}

    def accessor(array, component_type, kind, bounds=False):
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        binary.extend(array.tobytes())
        view = len(gltf['bufferViews'])
        gltf['bufferViews'].append({'buffer': 0, 'byteOffset': start, 'byteLength': array.nbytes})
        item = {'bufferView': view, 'componentType': component_type, 'count': len(array), 'type': kind}
        if bounds:
            item.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        index = len(gltf['accessors'])
        gltf['accessors'].append(item)
        return index

    def material_for(prim):
        # Neutral diagnostic materials preserve shape readability. Textures
        # are deliberately unnecessary for metric alignment.
        name = str(prim.GetPath()).lower()
        dark = any(x in name for x in ('gripper', 'head', 'wheel'))
        color = (0.16, 0.21, 0.26, 1) if dark else (0.68, 0.74, 0.80, 1)
        if color not in material_map:
            material_map[color] = len(gltf['materials'])
            gltf['materials'].append({'pbrMetallicRoughness': {
                'baseColorFactor': color, 'metallicFactor': 0.15, 'roughnessFactor': 0.65},
                'doubleSided': True})
        return material_map[color]

    body_primitives = {name: [] for name in body_names}
    triangle_count = 0
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        components = str(prim.GetPath()).split('/')
        if 'visuals' not in components or 'collisions' in components:
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == 'invisible':
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor.GetName() not in bodies:
            ancestor = ancestor.GetParent()
        if not ancestor:
            raise ValueError(f'visual mesh has no body ancestor: {prim.GetPath()}')
        body = ancestor.GetName()
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(points) or not len(counts):
            continue
        if counts.sum() != len(indices) or indices.min() < 0 or indices.max() >= len(points):
            raise ValueError(f'invalid USD mesh topology: {prim.GetPath()}')
        # USD uses row vectors; bake all visual transforms relative to body.
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) *
                               cache.GetLocalToWorldTransform(bodies[body]).GetInverse())
        points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3].astype('<f4')
        triangles, cursor = [], 0
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        for face, count in enumerate(counts):
            polygon = indices[cursor:cursor + count]
            if face not in holes:
                triangles.extend((polygon[0], polygon[i], polygon[i+1]) for i in range(1, count-1))
            cursor += count
        if not triangles:
            continue
        triangles = np.asarray(triangles, dtype='<u4')
        if mesh.GetOrientationAttr().Get() == 'leftHanded':
            triangles = triangles[:, [0, 2, 1]]
        triangle_count += len(triangles)
        body_primitives[body].append({'attributes': {'POSITION': accessor(points, 5126, 'VEC3', True)},
                                     'indices': accessor(triangles.ravel(), 5125, 'SCALAR'),
                                     'material': material_for(prim)})
    for name in body_names:
        matrix = cache.GetLocalToWorldTransform(bodies[name])
        node = {'name': name, 'matrix': np.asarray(matrix).ravel().tolist()}
        primitives = body_primitives[name]
        if primitives:
            node['mesh'] = len(gltf['meshes'])
            gltf['meshes'].append({'name': name, 'primitives': primitives})
        gltf['nodes'].append(node)
    if not triangle_count:
        raise ValueError('no robot visual meshes exported')
    while len(binary) % 4:
        binary.append(0)
    gltf['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(gltf, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    size = 12 + 8 + len(encoded) + 8 + len(binary)
    with Path(output).open('wb') as stream:
        stream.write(struct.pack('<III', 0x46546C67, 2, size))
        stream.write(struct.pack('<II', len(encoded), 0x4E4F534A))
        stream.write(encoded)
        stream.write(struct.pack('<II', len(binary), 0x004E4942))
        stream.write(binary)
    return {'body_count': len(body_names), 'triangle_count': triangle_count,
            'bytes': size, 'basis': 'Z-up, meters; nodes are absolute body frames'}
