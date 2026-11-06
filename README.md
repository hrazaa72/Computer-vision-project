# Measure It: segmentation and metric measurement

Takes a phone photo of a medicine box with a 22 mm coin beside it and reports the box's
width and height in millimetres.

```
photo -> undistort -> Mask R-CNN mask -> coin (Hough + ellipse) -> mm per pixel -> width, height
```

Result on 19 held-out photos: mean absolute error 3.0 mm (4.5%) against calliper values of
81 x 51 mm. The 2.0 mm target was missed; the reason is in `docs/OBJECTIVES.md`.

Documents: `docs/PLAN.md`, `docs/OBJECTIVES.md`, `docs/DEFINITION_OF_DONE.md`.

## Setup

Python 3.13. Run every command from the repository root.

```
pip install -r requirements.txt
```

## Run

| Stage | Command |
|---|---|
| Calibrate camera | `python calibration/calibrate.py` |
| Undistort photos and build masks | `python -m dataset.prepare` |
| Train (seed 42, 7 epochs used) | `python -m models.train --epochs 7` |
| Evaluate on test (run once) | `python -m models.evaluate` |
| Measure one photo | `python -m inference.measure photo.jpeg` |
| Accuracy study | `python -m measurement.accuracy` |

`python -m inference.measure photo.jpeg` prints JSON with `width_mm`, `height_mm` and `confidence`
and saves `photo_measured.jpg` with the mask overlay. It exits with code 2 and a JSON reason
(`object_not_found`, `reference_not_found`, `plane_tilted`, `image_size_mismatch`) when it will not answer.
Add `--true-width-mm 51 --true-height-mm 81` to also print the error against calliper values.
Photos must be 720 x 1280 (or 1280 x 720), the resolution the camera was calibrated at.

## Scene requirements

The conversion is only valid if the camera looks straight down, the coin lies flat on the same
table as the box, and the box top is `BOX_THICKNESS_MM` (51 mm) above it. Change the constants at
the top of `inference/measure.py` for a different box or coin.

## Weights

The 176 MB checkpoint is too large for Git. Download it from the release assets and place it at
`models/runs/best.pt`.

SHA-256: `bc2ca3e953744fdee713db696f804ed87ab68193285b6c0c18ca822755dceee0`

## Dataset

| | |
|---|---|
| Object | medicine box, 81 x 51 x 51 mm |
| Images | 70 photos taken by me, 720 x 1280 portrait (27 taken sideways, rotated) |
| Labelling | LabelMe polygons, one class (`med_box`) |
| Split | 49 train / 14 val / 7 test, seed 42, lists in `dataset/splits/` |
| Calibration | 20 checkerboard photos, 7 x 9 inner corners, 22 mm squares |

## Hardware

Intel Core i7-8650U, 15.8 GB RAM, no GPU, Windows 11. Training took 59 minutes on the CPU.

## Layout

```
calibration/   checkerboard images, calibration script, saved parameters, camera helper
dataset/       raw labelled photos, undistorted images and masks, split lists
models/        Mask R-CNN builder, training, evaluation, metrics
inference/     image-to-millimetres command
measurement/   accuracy study script and results
docs/          plan, objectives, definition of done, assessment brief
```
