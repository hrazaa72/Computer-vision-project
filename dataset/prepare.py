"""Rotate to portrait, undistort the labelled photos and rasterise the polygons to masks.

Usage:
    python -m dataset.prepare
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.camera import Undistorter, rotate_points_clockwise, to_portrait

ROOT = Path(__file__).resolve().parent


def prepare(raw_dir, out_dir):
    undistorter = Undistorter()
    image_dir, mask_dir = out_dir / "images", out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rotated_count = 0
    for image_path in sorted(raw_dir.glob("*.jpeg")):
        label = json.loads(image_path.with_suffix(".json").read_text())
        image, rotated = to_portrait(cv2.imread(str(image_path)))
        if image.shape[:2][::-1] != undistorter.size:
            raise ValueError(f"{image_path.name} is {image.shape[:2][::-1]}, expected {undistorter.size}")
        rotated_count += rotated

        mask = np.zeros(image.shape[:2], np.uint8)
        for shape in label["shapes"]:
            points = np.array(shape["points"])
            if rotated:
                points = rotate_points_clockwise(points, label["imageHeight"])
            polygon = np.round(undistorter.points(points)).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)

        cv2.imwrite(str(image_dir / f"{image_path.stem}.jpg"), undistorter.image(image),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
    print(f"prepared {len(list(image_dir.glob('*.jpg')))} images ({rotated_count} rotated to portrait)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw" / "Annotated images")
    parser.add_argument("--out", type=Path, default=ROOT / "undistorted")
    args = parser.parse_args()
    prepare(args.raw, args.out)


if __name__ == "__main__":
    main()
