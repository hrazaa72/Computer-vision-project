# Objectives

All numbers are raw output from this repository, produced on the machine below.

## Setup

| | |
|---|---|
| CPU / RAM / GPU | Intel Core i7-8650U (4 cores, 8 threads), 15.8 GB, no usable GPU (Intel UHD 620) |
| OS / Python | Windows 11 Pro, Python 3.13.5, torch 2.15.0 (CPU), torchvision 0.29.2, OpenCV 5.0.0.93 |
| Camera | [FILL IN: phone model], 720 x 1280 portrait, focus / zoom locked |
| Object | medicine box, top face 81 x 51 mm, 51 mm tall. Reference: 22 mm coin on the table |
| Ground truth | 81 x 51 mm, calliper. One physical box, photographed 21 times, so every instance has the same true size |

## Calibration (item 1)

20 checkerboard images (7 x 9 inner corners, 22 mm squares), all corners found.
RMS reprojection error **0.95 px**, worst image 1.50 px (`calibration/calibration_report.json`).
k1 = 0.285, k2 = -0.745, p1 = -0.009, p2 = -0.009, k3 fixed at 0 (free k3 fitted to 2.85, a sign
of overfitting; fixing it cost 0.03 px). Four of the 20 images were taken sideways and were
rotated to portrait; 27 of the 70 object photos likewise.

## MO-1: measurement accuracy (required)

| Field | Entry |
|---|---|
| What is measured | Mean absolute error of the reported box width and height against the calliper values, averaged over the two dimensions, per photo |
| How | `python -m measurement.accuracy` runs the full pipeline (undistort, segment, coin, mm) on each photo and compares with 51 x 81 mm |
| Target | Mean absolute error 2.0 mm or less (about 3% of the long side), worst case under 5 mm |
| Conditions | Phone camera, dark cloth, box and coin flat on the table, camera roughly overhead, same 19 held-out photos every run |
| Not included | Photos where the coin is not found (2 of 21: IMG_07 and IMG_21, refused by the system) |

Why 2 mm: a tolerance of about 2.5% is what a pack-size check needs to tell a 81 mm box from
an 85 mm one. The target was written after the first run of the accuracy script, so it is
justified by that use, not by the result.

**Result: missed.** Mean 3.00 mm, median 2.84 mm, worst 5.97 mm, mean 4.5%.

Held-out photos only: 6 test + 13 val. None was used for gradient updates. Val picked the
checkpoint, test did not.

| Image | Split | Measured W x H (mm), undistorted | Err W / H (mm) | Mean abs err undistorted (mm) | Mean abs err raw (mm) |
|---|---|---|---|---|---|
| IMG_08 | test | 48.9 x 76.9 | -2.1 / -4.1 | 3.11 | 2.77 |
| IMG_16 | test | 51.8 x 78.6 | +0.8 / -2.4 | 1.59 | 0.98 |
| IMG_23 | test | 52.8 x 84.9 | +1.8 / +3.9 | 2.84 | 2.26 |
| IMG_40 | test | 45.7 x 75.5 | -5.3 / -5.5 | 5.40 | 7.30 |
| IMG_43 | test | 54.8 x 85.3 | +3.8 / +4.3 | 4.03 | 3.45 |
| IMG_67 | test | 49.1 x 77.8 | -1.9 / -3.2 | 2.58 | 4.65 |
| IMG_05 | val | 50.4 x 77.9 | -0.6 / -3.1 | 1.86 | 1.97 |
| IMG_09 | val | 52.2 x 79.6 | +1.2 / -1.4 | 1.30 | 0.83 |
| IMG_17 | val | 54.6 x 87.4 | +3.6 / +6.4 | 5.03 | 4.72 |
| IMG_19 | val | 50.7 x 79.1 | -0.3 / -1.9 | 1.12 | 2.15 |
| IMG_26 | val | 53.6 x 78.0 | +2.6 / -3.0 | 2.79 | 1.25 |
| IMG_30 | val | 46.8 x 73.3 | -4.2 / -7.7 | 5.97 | 5.82 |
| IMG_34 | val | 53.5 x 84.9 | +2.5 / +3.9 | 3.19 | 2.79 |
| IMG_41 | val | 47.4 x 77.4 | -3.6 / -3.6 | 3.60 | 6.22 |
| IMG_45 | val | 46.2 x 78.6 | -4.8 / -2.4 | 3.58 | 3.32 |
| IMG_49 | val | 52.3 x 87.4 | +1.3 / +6.4 | 3.83 | 2.39 |
| IMG_51 | val | 50.8 x 84.7 | -0.2 / +3.7 | 1.94 | 1.98 |
| IMG_54 | val | 50.5 x 79.9 | -0.5 / -1.1 | 0.80 | 2.17 |
| IMG_63 | val | 53.2 x 83.5 | +2.2 / +2.5 | 2.35 | 3.63 |

| | Mean abs err (mm) | Median | Worst | Mean % err |
|---|---|---|---|---|
| Undistorted | 3.00 | 2.84 | 5.97 | 4.5 |
| Raw | 3.19 | 2.77 | 7.30 | 4.8 |

Source: `measurement/results/accuracy.csv`, `summary.json`.

**Why the error is about 3 mm.** Width and height errors move together (correlation 0.76), which
looks like a scale error. But the hand-drawn masks, measured with the same scale, give a larger
error (5.3% val, 6.4% test) than the model (4.5%), and their short / long ratio ranges 0.59
to 0.72 where the true ratio is 0.63. The box is 51 mm tall, so off-centre photos show its side
faces and the polygons include varying amounts of them. The floor is label consistency and
3-D shape, not the network. Five alternative coin-diameter estimators (ellipse, area, enclosing
circle, Hough, four thresholds) all gave 4.5% or worse on val, so the coin fit is not the limit.

## Undistortion (item 6)

The 0.19 mm mean difference (3.19 raw vs 3.00 undistorted) is small against a per-photo
scatter of about 1.5 mm, and raw is the better of the two on 11 of 19 photos (undistorted on 8). With these
numbers undistortion cannot be shown to help much; it reduces the worst case (7.30 to 5.97 mm)
and the signed width bias (-1.10 to -0.20 mm). The lens distortion is moderate (k1 = 0.28), and
the box sits near the centre of the frame where distortion is smallest. Real photos with the
box near the edge would show a larger effect; I did not test those.

## MO-2: inference time

| Field | Entry |
|---|---|
| What is measured | Wall-clock time for the full pipeline (undistort, segment, coin, mm) on one photo |
| How | `measurement.accuracy`: one warm-up, then 5 timed runs on the same photo |
| Target | 10 s or less per photo on this CPU |
| Conditions | i7-8650U, CPU only, batch of 1 |
| Not included | Model loading and Python start-up |

**Result: met.** Median 5.92 s, min 5.21 s, max 5.94 s.

## MO-3: segmentation quality

| Field | Entry |
|---|---|
| What is measured | Mask IoU and mask AP on the test split |
| How | `python -m models.evaluate`, run once after the checkpoint was fixed |
| Target | Mean mask IoU 0.90 or more on the 7 test images |

**Result: met.**

| Mask IoU (mean / median / min) | mAP@0.5 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 |
|---|---|---|---|---|
| 0.961 / 0.965 / 0.916 | 1.000 | 0.986 | 1.000 | 1.000 |

Seven images is a small test set, so these are indicative. Training: 7 epochs, 49 train images,
seed 42, SGD lr 0.005, batch 2, CPU, 59 minutes. Best checkpoint on val: epoch 7, IoU 0.958
(`models/runs/history.json`).

## Condition that breaks it (item 8): box far from the frame centre

Chosen because it can be measured from the existing photos without retaking any, and because a
tall box off-centre shows its side faces and gets more lens distortion.

| Group (6 photos each) | Distance of box centre from frame centre (px) | Mean abs err (mm) |
|---|---|---|
| Near centre | 21 to 96 | 2.19 |
| Middle | 101 to 137 | 3.51 |
| Far from centre | 186 to 314 | 3.24 |

Error rises from 2.2 to about 3.4 mm once the box leaves the centre, but the middle and far
groups do not differ, so this shows an effect, not a trend. Six photos per group.

## Knowing when not to answer (item 9)

Refusal rules in `inference/measure.py`: mask score below `MIN_SCORE` (0.5), no coin found,
coin axis ratio below `MIN_COIN_AXIS_RATIO` (0.9). Sweep over the 19 measured photos:

| Min axis ratio | Photos kept | Refused | Mean abs err of kept (mm) | Worst (mm) |
|---|---|---|---|---|
| 0.00 | 19 | 0% | 3.00 | 5.97 |
| 0.90 | 18 | 5% | 3.01 | 5.97 |
| 0.95 | 9 | 53% | 2.80 | 4.03 |
| 0.97 | 2 | 89% | 1.57 | 2.35 |

The mask-score threshold has no effect here: every score is 0.997 or higher, so it cannot
separate good from bad masks on these photos. The coin axis ratio is a weak guard. Refusing
half the photos lowers the mean error by only 0.2 mm. I kept 0.9, which costs one photo. A
trustworthy refusal rule for bad masks would need photos where the model is actually unsure.
The coin axis ratio sits around 0.93 to 0.95 on good photos, which is lower than a flat coin
seen from above should give, so it also reflects the coin outline, not only tilt.
