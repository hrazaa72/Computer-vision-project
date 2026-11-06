"""Calibrate the camera from checkerboard photos and save the parameters.

Usage:
    python calibration/calibrate.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# Inner corners (columns, rows) as seen in the portrait photos, and square size in mm.
PATTERN = (7, 9)
SQUARE_MM = 22.0

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(image_dir):
    """Return object points, image points, image size and per-image outcome."""
    template = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    template[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    obj_points, img_points, used, skipped, rotated = [], [], [], [], []
    size = None
    for path in sorted(image_dir.glob("*.jpeg")):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > gray.shape[0]:
            # A few photos were taken with the phone held sideways; bring them to portrait.
            gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            rotated.append(path.name)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            raise ValueError(f"{path.name} is {gray.shape[::-1]}, expected {size}")
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN)
        if not found:
            skipped.append(path.name)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        obj_points.append(template)
        img_points.append(corners)
        used.append(path.name)
    return obj_points, img_points, size, used, skipped, rotated


def per_image_error(obj, img, rvec, tvec, matrix, dist):
    projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
    diff = img.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    parser.add_argument("--out", type=Path, default=ROOT / "camera_params.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "calibration_report.json")
    args = parser.parse_args()

    obj, img, size, used, skipped, rotated = find_corners(args.images)
    if len(used) < 10:
        raise SystemExit(f"Only {len(used)} usable images, need at least 10")

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, size, None, None, flags=cv2.CALIB_FIX_K3)
    errors = [per_image_error(o, i, r, t, matrix, dist)
              for o, i, r, t in zip(obj, img, rvecs, tvecs)]

    np.savez(args.out, camera_matrix=matrix, dist_coeffs=dist, image_size=np.array(size))
    report = {
        "opencv_version": cv2.__version__,
        "pattern_inner_corners": PATTERN,
        "square_mm": SQUARE_MM,
        "image_size_wh": size,
        "images_used": len(used),
        "images_skipped": skipped,
        "images_rotated_to_portrait": rotated,
        "flags": "CALIB_FIX_K3",
        "rms_reprojection_error_px": rms,
        "per_image_error_px": dict(zip(used, errors)),
        "camera_matrix": matrix.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
    }
    args.report.write_text(json.dumps(report, indent=2))

    print(f"images used: {len(used)}, skipped: {skipped}, rotated: {rotated}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"worst image: {max(zip(errors, used))[1]} at {max(errors):.4f} px")
    print(f"fx={matrix[0, 0]:.1f} fy={matrix[1, 1]:.1f} cx={matrix[0, 2]:.1f} cy={matrix[1, 2]:.1f}")
    print(f"dist (k1 k2 p1 p2 k3, k3 fixed at 0): {np.round(dist.ravel(), 5).tolist()}")


if __name__ == "__main__":
    main()
