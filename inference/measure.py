"""One image in, box width and height in millimetres out.

Usage:
    python -m inference.measure photo.jpeg [--weights models/runs/best.pt] [--overlay out.jpg]

Conversion (see mm_per_pixel): the camera is a pinhole with focal length f in pixels, looking
straight down. A flat object at distance Z from the camera has Z / f millimetres per pixel.
The coin lies on the table and the box top is BOX_THICKNESS_MM closer to the camera, so
    Z_coin = f * COIN_DIAMETER_MM / coin_diameter_px
    Z_box  = Z_coin - BOX_THICKNESS_MM
and the box is measured at the scale Z_box / f.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from calibration.camera import Undistorter, to_portrait
from models.model import load_checkpoint

COIN_DIAMETER_MM = 22.0
BOX_THICKNESS_MM = 51.0   # height of the labelled top face above the table, calliper-measured

MIN_SCORE = 0.5            # below this the mask is not trusted
MIN_COIN_AXIS_RATIO = 0.9  # minor / major axis of the coin; lower means the plane is tilted
COIN_RADIUS_RANGE_PX = (40, 90)
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "runs" / "best.pt"


class Refusal(Exception):
    """The system declines to give a measurement. `code` is machine-readable."""

    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def segment(model, image_rgb, device):
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    if len(output["scores"]) == 0 or float(output["scores"][0]) < MIN_SCORE:
        best = float(output["scores"][0]) if len(output["scores"]) else 0.0
        raise Refusal("object_not_found", f"best mask score {best:.2f} is below {MIN_SCORE}")
    return (output["masks"][0, 0] > 0.5).cpu().numpy(), float(output["scores"][0])


def find_coin(gray, box_mask):
    """Return (centre, diameter_px, axis_ratio, ellipse) of the reference coin."""
    blurred = cv2.medianBlur(gray, 5)
    lo, hi = COIN_RADIUS_RANGE_PX
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.5, 200, param1=100, param2=40,
                               minRadius=lo, maxRadius=hi)
    if circles is None:
        raise Refusal("reference_not_found", "no circle of coin size in the image")

    keep = cv2.dilate(box_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    for x, y, radius in circles[0]:
        if keep[int(y), int(x)]:
            continue
        pad = int(radius * 1.5)
        x0, y0 = max(int(x) - pad, 0), max(int(y) - pad, 0)
        crop = blurred[y0:int(y) + pad, x0:int(x) + pad]
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        near = [c for c in contours if len(c) >= 20
                and cv2.pointPolygonTest(c, (x - x0, y - y0), False) >= 0]
        if not near:
            continue
        (cx, cy), axes, angle = cv2.fitEllipse(max(near, key=cv2.contourArea))
        major, minor = max(axes), min(axes)
        if abs((major + minor) / 2 - 2 * radius) > 0.15 * 2 * radius:
            continue
        ellipse = ((cx + x0, cy + y0), axes, angle)
        return (cx + x0, cy + y0), (major + minor) / 2, minor / major, ellipse
    raise Refusal("reference_not_found", "no coin-like blob outside the box")


def mm_per_pixel(focal_px, coin_diameter_px):
    z_coin = focal_px * COIN_DIAMETER_MM / coin_diameter_px
    return (z_coin - BOX_THICKNESS_MM) / focal_px


def box_size_px(mask):
    contour = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    return rect, min(rect[1]), max(rect[1])


def measure(image_bgr, undistorter, model, device, undistort=True):
    """Measure one photo. undistort=False skips lens correction, for the raw-vs-undistorted
    comparison; the focal length then comes from the original camera matrix."""
    image, _ = to_portrait(image_bgr)
    if image.shape[1::-1] != undistorter.size:
        raise Refusal("image_size_mismatch",
                      f"photo is {image.shape[1]}x{image.shape[0]}, the calibration is for "
                      f"{undistorter.size[0]}x{undistorter.size[1]}")
    matrix = undistorter.new_matrix if undistort else undistorter.matrix
    if undistort:
        image = undistorter.image(image)
    mask, score = segment(model, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device)
    centre, coin_px, axis_ratio, ellipse = find_coin(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
    if axis_ratio < MIN_COIN_AXIS_RATIO:
        raise Refusal("plane_tilted", f"coin axis ratio {axis_ratio:.3f} is below {MIN_COIN_AXIS_RATIO}")

    focal = float(matrix[0, 0] + matrix[1, 1]) / 2
    scale = mm_per_pixel(focal, coin_px)
    rect, short_px, long_px = box_size_px(mask)
    result = {"width_mm": short_px * scale, "height_mm": long_px * scale, "confidence": score,
              "mm_per_pixel": scale, "coin_diameter_px": coin_px, "coin_axis_ratio": axis_ratio,
              "box_centre_offset_px": float(np.hypot(rect[0][0] - image.shape[1] / 2,
                                                     rect[0][1] - image.shape[0] / 2))}
    return result, draw_overlay(image, mask, rect, ellipse, result)


def draw_overlay(image, mask, rect, ellipse, result):
    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.drawContours(overlay, [np.intp(cv2.boxPoints(rect))], -1, (0, 255, 0), 2)
    cv2.ellipse(overlay, ellipse, (255, 200, 0), 2)
    label = (f"W {result['width_mm']:.1f} mm  H {result['height_mm']:.1f} mm  "
             f"conf {result['confidence']:.2f}")
    cv2.putText(overlay, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(overlay, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--overlay", type=Path, help="where to save the overlay (default: next to the image)")
    parser.add_argument("--true-width-mm", type=float, help="calliper width, to report the error")
    parser.add_argument("--true-height-mm", type=float, help="calliper height, to report the error")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        sys.exit(f"cannot read {args.image}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.weights, device)
    try:
        result, overlay = measure(image, Undistorter(), model, device)
    except Refusal as refusal:
        print(json.dumps({"status": "refused", "reason": refusal.code, "detail": str(refusal)}))
        sys.exit(2)

    out = args.overlay or args.image.with_name(args.image.stem + "_measured.jpg")
    cv2.imwrite(str(out), overlay)
    report = {"status": "ok", **{k: round(v, 3) for k, v in result.items()}, "overlay": str(out)}
    if args.true_width_mm is not None and args.true_height_mm is not None:
        width_err = result["width_mm"] - args.true_width_mm
        height_err = result["height_mm"] - args.true_height_mm
        report["width_error_mm"] = round(width_err, 3)
        report["height_error_mm"] = round(height_err, 3)
        report["mean_abs_error_mm"] = round((abs(width_err) + abs(height_err)) / 2, 3)
        report["mean_pct_error"] = round(
            100 * (abs(width_err) / args.true_width_mm + abs(height_err) / args.true_height_mm) / 2, 2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
