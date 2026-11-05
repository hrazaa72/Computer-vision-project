# Plan: Measure It

Branch `dev-test01`. This file and `dataset/splits/` are committed before any code.

## Setup (fill in before submitting)

- Object: medicine box (`med_box`), 81 x 51 mm (length x width). Chosen because it is rigid, flat-faced and easy to photograph repeatedly.
- Reference object: coin, 22 mm diameter, 2 mm thick (calliper-measured), lying in the same plane as the object in every photo.
- Camera: phone, 1280x720 portrait, focus / zoom / resolution locked.
- Checkerboard: 20 images in `calibration/images/`. 7 x 9 inner corners, square size 22 mm (calliper-measured).

## Order of work and time budget (8 h)

| # | Stage | Hours | Notes |
|---|-------|-------|-------|
| 1 | Calibration (`calibration/`) | 0.75 | OpenCV, save `.npz`, report RMS reprojection error |
| 2 | Dataset (`dataset/`) | 1.0 | Undistort 70 images, transform LabelMe polygons to match, add reference mask, build masks |
| 3 | Model (`models/`) | 2.0 | Mask R-CNN (torchvision) fine-tuned from pretrained weights. Training runs unattended on GPU. |
| 4 | Inference command (`inference/`) | 1.5 | undistort, segment, find reference, mm/px, overlay. Handles object-not-found and reference-not-found. |
| 5 | Accuracy vs calliper (`measurement/`) | 1.0 | 10+ instances, per-instance table, mean / median / worst |
| 6 | Raw vs undistorted | 0.5 | Same instances, both errors side by side |
| 7 | Target, breaking condition, refusal threshold | 1.0 | Items 7, 8, 9 |
| 8 | Docs, README, Loom | 0.75 | |

Items 8 and 9 start only after item 5 is finished.

## Fixed decisions

- Split 49 / 14 / 7 (train / val / test), seed 42, list files in `dataset/splits/`. Committed before any code. The test set is never used for training, tuning or checkpoint choice.
- Seed, library versions and training config pinned. Weights published as a release asset with SHA-256 in the README.
- No Ultralytics YOLO and no Roboflow models.

## Deliberately skipped

- Multi-class or multi-object detection: one object class, one object per image.
- Perspective correction for a tilted plane. Tilt is measured as the breaking condition and refused beyond a threshold instead.
- Mobile / edge deployment.

## Changes from the plan (added during the work)

- Training ran on the laptop CPU for 7 epochs (59 minutes), not on a GPU. Cause: no GPU was set up and time was short.
- The reference coin is found with a Hough circle and ellipse fit. It was not labelled as a second class.
- The coin lies on the table, the box top is 51 mm higher. The photos could not be retaken, so the measurement
  code corrects the scale for that height.
- Breaking condition changed from camera tilt to distance from the frame centre, because it can be
  measured from the existing photos. Tilt is only guarded by the coin axis ratio.
- Four checkerboard photos and 27 object photos were taken sideways and are rotated to portrait.
- Accuracy target (MO-1) was missed; see `docs/OBJECTIVES.md`.

## Decision record

- **Approach taken:** Mask R-CNN (COCO-pretrained, fine-tuned on 49 images) for the box outline; the coin
  found by classical circle detection; scale from the coin diameter, corrected for the 51 mm height
  difference between the coin plane and the box top with a pinhole model (mm per pixel = Z / f).
- **Approach rejected:** retaking the photos with the reference lying in the same plane as the measured
  face (coin on top of the box), which is what the brief asks for and needs no height assumption.
- **What rejecting it cost:** the scale now depends on a measured thickness (51 mm). One millimetre of
  error in that value changes the scale by about 0.6%, roughly 0.5 mm on the 81 mm side. It also assumes
  the camera looks straight down. The existing 70 labelled photos were kept, which saved relabelling. The error
  floor from the 3-D box shape (side faces entering the mask) is probably also larger than it would be
  with a flat reference, but I did not measure that.
