"""WXYZ pose composition shared by extraction and calibrated evaluation."""
import numpy as np


def quaternion_multiply(a, b):
    """Hamilton product, both inputs wxyz."""
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def rotate(q, point):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("invalid quaternion")
    q = q / norm
    return quaternion_multiply(quaternion_multiply(q, [0, *point]), q * [1, -1, -1, -1])[1:]


def compose_pose(parent, local):
    parent, local = np.asarray(parent, float), np.asarray(local, float)
    return np.r_[parent[:3] + rotate(parent[3:], local[:3]),
                 quaternion_multiply(parent[3:], local[3:])]


def relative_pose(base, world):
    base, world = np.asarray(base, float), np.asarray(world, float)
    inverse = base[3:] * [1, -1, -1, -1]
    return np.r_[rotate(inverse, world[:3] - base[:3]),
                 quaternion_multiply(inverse, world[3:])]
