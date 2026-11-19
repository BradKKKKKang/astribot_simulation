"""CPU-side validation and pose estimates for volumetric scene assets."""
import math

import numpy as np

from .isaac_scene_config import _fields
from .glb_geometry import rotation_matrix


def validate_deformable(physics, primitives):
    cfg = physics.get('deformable', {})
    positive = ('youngs_modulus', 'contact_offset')
    nonnegative = ('elasticity_damping', 'vertex_velocity_damping', 'rest_offset')
    integers = ('simulation_hexahedral_resolution', 'collision_simplification_remeshing_resolution',
                'collision_simplification_target_triangle_count', 'solver_position_iteration_count')
    required = {*positive, *nonnegative, *integers, 'poissons_ratio', 'core_nodes', 'self_collision'}
    _fields(cfg, required, 'asset.physics.deformable')
    if set(cfg) != required or physics['collision'] != 'tetrahedral':
        raise ValueError('deformable body requires explicit material/meshing parameters and tetrahedral collision')
    for name in (*positive, *nonnegative, 'poissons_ratio'):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'deformable.{name} must be finite and nonnegative')
    if any(cfg[name] <= 0 for name in positive) or cfg['poissons_ratio'] >= .5:
        raise ValueError('deformable modulus/contact offset must be positive; Poisson ratio must be below 0.5')
    if cfg['rest_offset'] >= cfg['contact_offset']:
        raise ValueError('deformable rest_offset must be below contact_offset')
    for name in integers:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'deformable.{name} must be a positive integer')
    if type(cfg['self_collision']) is not bool:
        raise ValueError('deformable.self_collision must be boolean')
    nodes = cfg['core_nodes']
    available = {p['node'] for p in primitives}
    if (not isinstance(nodes, list) or not nodes or any(not isinstance(n, str) for n in nodes)
            or len(set(nodes)) != len(nodes) or not set(nodes) <= available):
        raise ValueError('deformable.core_nodes must uniquely name existing GLB nodes')
    if physics.get('restitution', 0) != 0:
        raise ValueError('deformable restitution is unsupported; use elasticity damping')


def tetrahedral_volume(points, indices):
    """Reject disconnected/degenerate cooked cores before creating a soft body."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points, tets = np.asarray(points), np.asarray(indices).reshape(-1, 4)
    if (len(points) < 4 or not len(tets) or not np.isfinite(points).all()
            or tets.min() < 0 or tets.max() >= len(points)):
        raise ValueError('deformable tetrahedral cooking produced invalid geometry')
    edges = (points[tets[:, 1:]] - points[tets[:, :1]])
    volumes = np.abs(np.linalg.det(edges)) / 6
    if np.any(volumes < 1e-15):
        raise ValueError('deformable tetrahedral core contains degenerate elements')
    rows = np.repeat(tets[:, 0], 3)
    cols = tets[:, 1:].ravel()
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(points)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError('deformable tetrahedral core is disconnected')
    return float(volumes.sum())


def fit_deformable_pose(rest_world, current_world, initial_pose):
    """Best-fit material frame plus nonrigid RMS residual, never a rigid-body state."""
    from scipy.spatial.transform import Rotation
    rest, current = np.asarray(rest_world), np.asarray(current_world)
    center, moved = rest.mean(0), current.mean(0)
    u, _, vt = np.linalg.svd((rest - center).T @ (current - moved))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    delta = vt.T @ correction @ u.T
    origin = moved + delta @ (np.asarray(initial_pose[:3]) - center)
    x, y, z, w = Rotation.from_matrix(delta @ rotation_matrix(initial_pose[3:])).as_quat()
    residual = current - ((rest - center) @ delta.T + moved)
    return [*origin, w, x, y, z], float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
