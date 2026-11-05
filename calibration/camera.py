"""Load the saved camera parameters and undistort images and points."""
from pathlib import Path

import cv2
import numpy as np

PARAMS_PATH = Path(__file__).resolve().parent / "camera_params.npz"


def to_portrait(image):
    """Rotate a landscape image 90 degrees clockwise, the convention used for calibration."""
    if image.shape[1] > image.shape[0]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
    return image, False


def rotate_points_clockwise(points, height):
    """Map (x, y) points of a landscape image of the given height into its clockwise rotation."""
    points = np.asarray(points, np.float64)
    return np.stack([height - points[:, 1], points[:, 0]], axis=1)


class Undistorter:
    """Undistorts portrait images with the calibrated camera. alpha=0 keeps only valid pixels."""

    def __init__(self, params_path=PARAMS_PATH):
        params = np.load(params_path)
        self.matrix = params["camera_matrix"]
        self.dist = params["dist_coeffs"]
        width, height = (int(v) for v in params["image_size"])
        self.size = (width, height)
        self.new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.matrix, self.dist, self.size, alpha=0)
        self._maps = cv2.initUndistortRectifyMap(
            self.matrix, self.dist, None, self.new_matrix, self.size, cv2.CV_16SC2)

    def image(self, image):
        return cv2.remap(image, *self._maps, interpolation=cv2.INTER_LINEAR)

    def points(self, points):
        pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.matrix, self.dist, P=self.new_matrix).reshape(-1, 2)
