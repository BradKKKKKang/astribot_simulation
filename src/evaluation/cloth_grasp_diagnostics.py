"""Geometric cloth/gripper evidence, independent of Isaac and contact-force APIs.

Offset half-spaces conservatively enclose a rounded convex hull. Candidate counts
can therefore overestimate proximity at corners; they are not contact reports.
"""
import numpy as np

from src.simulation_config.glb_geometry import rotation_matrix


def compile_finger_hulls(body):
    """Compile cooked collider polygons in their rigid body's local frame."""
    hulls = []
    for collider in body['colliders']:
        transform = np.asarray(collider['collider_to_body'], dtype=float)
        for hull in collider['hulls']:
            vertices = np.asarray(hull['vertices'], dtype=float)
            vertices = vertices @ transform[:3, :3] + transform[3, :3]
            center = vertices.mean(0)
            normals, offsets = [], []
            for polygon in hull['polygons']:
                start, count = polygon['index_base'], polygon['num_vertices']
                face = vertices[np.asarray(hull['indices'][start:start + count])]
                normal = np.cross(face - face[0], np.roll(face, -1, axis=0) - face[0]).sum(0)
                length = np.linalg.norm(normal)
                if length < 1e-12:
                    raise ValueError('degenerate cooked finger polygon')
                normal /= length
                if np.dot(normal, face.mean(0) - center) < 0:
                    normal = -normal
                normals.append(normal)
                offsets.append(-np.dot(normal, face[0]))
            if len(normals) < 4 or not np.isfinite(vertices).all():
                raise ValueError('invalid cooked finger hull')
            hulls.append({'vertices': vertices, 'normals': np.asarray(normals),
                          'offsets': np.asarray(offsets)})
    if not hulls:
        raise ValueError('no cooked finger hulls')
    return hulls


def _plane_clearance(points, hulls, band):
    """Signed plane bound for the union, evaluated inside expanded AABBs only."""
    clearance = np.full(len(points), np.inf)
    for hull in hulls:
        vertices = hull['vertices']
        mask = np.all((points >= vertices.min(0) - band) &
                      (points <= vertices.max(0) + band), axis=1)
        if mask.any():
            values = (points[mask] @ hull['normals'].T + hull['offsets']).max(axis=1)
            clearance[mask] = np.minimum(clearance[mask], values)
    return clearance


def sample_cloth_grasp(points, finger_poses, finger_hulls, rest_offset, contact_offset):
    """Count particles near each finger and both, using cloth-only offset bands.

    The gap is a signed projection along L11's local X, not a minimum Euclidean
    distance. The between-finger region is the overlapping Y/Z bounding rectangle
    and the two inner X bounds, expanded by the cloth rest offset. Negative gaps
    mean overlapping projections, not proven collider penetration.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('cloth particles must be finite Nx3 positions')
    if not 0 < rest_offset < contact_offset or not np.isfinite(contact_offset):
        raise ValueError('invalid cloth proximity offsets')
    clearances, vertices_world = [], []
    for suffix in ('L11', 'R11'):
        pose = np.asarray(finger_poses[suffix], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('invalid finger pose')
        rotation = rotation_matrix(pose[3:])
        local = (points - pose[:3]) @ rotation
        hulls = finger_hulls[suffix]
        clearances.append(_plane_clearance(local, hulls, contact_offset))
        vertices = np.concatenate([hull['vertices'] for hull in hulls])
        vertices_world.append(vertices @ rotation.T + pose[:3])
    reference = np.asarray(finger_poses['L11'])
    rotation = rotation_matrix(reference[3:])
    left, right = [(v - reference[:3]) @ rotation for v in vertices_world]
    inner_left, inner_right = left[:, 0].min(), right[:, 0].max()
    lower = np.maximum(left.min(0), right.min(0))
    upper = np.minimum(left.max(0), right.max(0))
    lower[0], upper[0] = inner_right, inner_left
    local = (points - reference[:3]) @ rotation
    between = (np.all((local >= lower - rest_offset) &
                      (local <= upper + rest_offset), axis=1)
               if np.all(lower <= upper) else np.zeros(len(points), dtype=bool))
    result = {'status': 'ok', 'particle_count': len(points),
              'closing_axis_gap_m': float(inner_left - inner_right),
              'between_fingers_candidates': int(between.sum()),
              'cloth_rest_band_m': rest_offset, 'cloth_contact_band_m': contact_offset}
    for label, band in (('rest', rest_offset), ('contact', contact_offset)):
        first, second = [values <= band for values in clearances]
        result[label + '_candidates'] = {
            'L11': int(first.sum()), 'R11': int(second.sum()),
            'both': int((first & second).sum())}
    result['inside_hull_particles'] = {
        key: int((values < -1e-4).sum())
        for key, values in zip(('L11', 'R11'), clearances)}
    return result
