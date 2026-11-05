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

## Decision record (completed at the end)

- Approach taken: _
- Approach rejected: _
- What rejecting it cost: _
