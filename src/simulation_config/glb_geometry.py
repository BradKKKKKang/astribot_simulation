"""Read triangle GLB geometry without importing a simulator or converting assets.

The supported subset is explicit: embedded GLB 2 buffers, ordinary accessors,
triangle primitives and static node transforms. Unsupported geometry fails
before launch rather than silently using an incomplete bounding box.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .poses import rotate


def rotation_matrix(quaternion):
    return np.column_stack([rotate(quaternion, axis) for axis in np.eye(3)])


def read_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
        raise ValueError('asset must be a valid GLB 2 file')
    chunks = {}
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if offset + length > len(data) or kind in chunks:
            raise ValueError('invalid GLB chunks')
        chunks[kind] = data[offset:offset + length]
        offset += length
    document = json.loads(chunks[0x4e4f534a])
    binary = chunks.get(0x004e4942, b'')
    if document.get('skins') or document.get('animations') or document.get('extensionsRequired'):
        raise ValueError('scene assets require static, uncompressed GLB geometry')
    if len(document.get('buffers', [])) != 1 or 'uri' in document['buffers'][0]:
        raise ValueError('scene assets require one embedded GLB buffer')

    def accessor(index):
        spec = document['accessors'][index]
        if 'sparse' in spec or 'bufferView' not in spec:
            raise ValueError('sparse or implicit GLB accessors are unsupported')
        view = document['bufferViews'][spec['bufferView']]
        types = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
        widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
        dtype, width = np.dtype(types[spec['componentType']]), widths[spec['type']]
        stride = view.get('byteStride', width * dtype.itemsize)
        start = view.get('byteOffset', 0) + spec.get('byteOffset', 0)
        end = start + max(spec['count'] - 1, 0) * stride + width * dtype.itemsize
        if view.get('buffer', 0) != 0 or end > view.get('byteOffset', 0) + view['byteLength']:
            raise ValueError('GLB accessor exceeds its buffer view')
        values = np.ndarray((spec['count'], width), dtype=dtype, buffer=binary,
                            offset=start, strides=(stride, dtype.itemsize)).copy()
        if spec.get('normalized'):
            limit = np.iinfo(dtype)
            values = np.maximum(values.astype(float) / limit.max, -1)
        return values

    primitives = []
    nodes = document.get('nodes', [])
    def visit(index, parent, ancestors):
        if index in ancestors:
            raise ValueError('cyclic GLB node graph')
        node = nodes[index]
        if 'skin' in node:
            raise ValueError('skinned GLB nodes are unsupported')
        if 'matrix' in node:
            local = np.asarray(node['matrix'], float).reshape((4, 4), order='F')
        else:
            local = np.eye(4)
            x, y, z, w = node.get('rotation', [0, 0, 0, 1])
            local[:3, :3] = rotation_matrix([w, x, y, z]) @ np.diag(node.get('scale', [1, 1, 1]))
            local[:3, 3] = node.get('translation', [0, 0, 0])
        transform = parent @ local
        if 'mesh' in node:
            mesh = document['meshes'][node['mesh']]
            for primitive in mesh['primitives']:
                if primitive.get('mode', 4) != 4 or primitive.get('targets') or primitive.get('extensions'):
                    raise ValueError('scene assets require ordinary triangle primitives')
                attributes = {name: accessor(value) for name, value in primitive['attributes'].items()}
                vertices = attributes.pop('POSITION').astype(float)
                vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
                if not len(vertices) or not np.isfinite(vertices).all():
                    raise ValueError('asset geometry must contain finite vertices')
                faces = (accessor(primitive['indices']).ravel() if 'indices' in primitive
                         else np.arange(len(vertices))).astype(np.int64)
                if len(faces) % 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
                    raise ValueError('invalid triangle indices')
                faces = faces.reshape((-1, 3))
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                if 'NORMAL' in attributes:
                    normals = attributes['NORMAL'] @ np.linalg.inv(transform[:3, :3])
                    attributes['NORMAL'] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                primitives.append({'node': node.get('name', str(index)), 'mesh': mesh.get('name', ''),
                                   'vertices': vertices, 'faces': faces, 'attributes': attributes,
                                   'material': primitive.get('material')})
        for child in node.get('children', []):
            visit(child, transform, {*ancestors, index})

    scene = document.get('scene', 0)
    for index in document['scenes'][scene].get('nodes', []):
        visit(index, np.eye(4), set())
    if not primitives:
        raise ValueError('asset has no triangle geometry')
    return {'sha256': hashlib.sha256(data).hexdigest(), 'primitives': primitives,
            'document': document, 'binary': binary}
