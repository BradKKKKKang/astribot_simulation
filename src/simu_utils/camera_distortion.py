"""Apply OpenCV lens calibration to ideal pinhole RGB renders."""

import cv2
import numpy as np


class CameraDistortion:
    """Cache a destination-to-source map for the five-parameter plumb-bob model."""

    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.shape != (5,) or not np.isfinite(self.coefficients).all():
            raise ValueError('camera distortion requires finite [k1, k2, p1, p2, k3]')
        self._map_key = None
        self._maps = None

    @classmethod
    def from_prim(cls, prim):
        """Read optional calibration inherited from the robot USD camera."""
        model = prim.GetAttribute('astribot:distortion:model').Get()
        coefficients = prim.GetAttribute('astribot:distortion:coefficients').Get()
        if model is None and coefficients is None:
            return None
        if model != 'plumb_bob':
            raise ValueError(f'unsupported camera distortion model: {model}')
        return cls(coefficients)

    def apply(self, rgb, intrinsic_matrix):
        """Distort an ideal render, preserving resolution and the optical center.

        remap needs the ideal source pixel for each distorted destination pixel,
        so invert the lens model with undistortPoints. initUndistortRectifyMap
        would perform the opposite image transformation. Pixels outside the
        rendered field of view are black.
        """
        if not np.any(self.coefficients):
            return rgb
        height, width = rgb.shape[:2]
        matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
            raise ValueError('camera distortion requires a finite intrinsic matrix with positive focal lengths')
        key = (height, width, matrix.tobytes(), self.coefficients.tobytes())
        if key != self._map_key:
            y, x = np.indices((height, width), dtype=np.float32)
            pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
            source = cv2.undistortPointsIter(
                pixels, matrix, self.coefficients, None, matrix,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-9),
            ).reshape(height, width, 2)
            self._maps = (np.ascontiguousarray(source[..., 0]),
                          np.ascontiguousarray(source[..., 1]))
            self._map_key = key
        return cv2.remap(rgb, *self._maps, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
