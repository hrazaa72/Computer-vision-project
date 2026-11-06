# Definition of Done

Written as a checklist of what "finished" means. Ticked only where there is evidence beside it.

## Floor (items 1 to 4)

- [x] Calibration: 20 checkerboard images, RMS reprojection error **0.95 px**, parameters in
  `calibration/camera_params.npz`, loaded by `calibration/camera.py`. Report: `calibration/calibration_report.json`.
- [x] 70 own images labelled with polygon masks (LabelMe), undistorted, split 49 / 14 / 7. Lists in `dataset/splits/`.
- [x] Test split committed before any code: commit `5c03ffe` (first commit after the README base).
- [x] Segmentation model trained, not Ultralytics, not Roboflow: torchvision Mask R-CNN. Test metrics in
  `docs/OBJECTIVES.md` (IoU 0.961, mAP@0.5 1.0, mAP@0.5:0.95 0.986, precision 1.0, recall 1.0).
- [x] One command from image to mm: `python -m inference.measure photo.jpeg` prints width, height and confidence
  as JSON and saves an overlay. Refuses with `object_not_found`, `reference_not_found` or `plane_tilted`
  (tested on a blank image and on a photo with the coin painted over).

## Beyond the floor

- [x] Item 5, accuracy against calliper values: 19 photos, per-instance table, mean 3.00 mm, median 2.84 mm,
  worst 5.97 mm, 4.5%. `measurement/results/accuracy.csv`.
- [x] Item 6, raw vs undistorted, side by side: 3.19 vs 3.00 mm. Difference is small and not consistent
  across photos; written up honestly in `docs/OBJECTIVES.md`.
- [x] Item 7, targets set and measured: MO-1 **missed** (3.00 mm against 2.0 mm, reason written down),
  MO-2 met (5.9 s against 10 s), MO-3 met (IoU 0.961 against 0.90).
- [x] Item 8, condition that breaks it: distance from frame centre, 2.2 mm near centre vs 3.2 to 3.5 mm away.
- [x] Item 9, refusal threshold and its cost: coin axis ratio 0.9 refuses 1 of 19 photos (5%); the sweep is
  in `docs/OBJECTIVES.md`. The mask-score threshold does nothing on these photos.
- [x] Item 10, decision record: at the end of `docs/PLAN.md`.

## Reproducibility

- [x] Seed 42, library versions in `requirements.txt`, training config in `models/runs/config.json`
  (not committed; weights and run files are git-ignored).
- [ ] Weights published as a release asset with SHA-256 in the README. Hash computed
  (`bc2ca3e9...dceee0`); the release upload is not done yet.
- [ ] Pull request from `dev-test01` into `main` opened.
- [ ] Loom recording (5 minutes or less) made and linked.
- [ ] Camera model written into `docs/OBJECTIVES.md` (placeholder left).

## What I did not finish or could not do

- **One physical box.** All 21 measured photos show the same box, so "10 instances" means 10 poses, not 10
  different objects. A different box size would test the pipeline more strongly.
- **Reference and object not in one plane.** The coin is on the table; the box top is 51 mm higher. The code
  corrects for that with the known height (`BOX_THICKNESS_MM`), so a wrong height gives a wrong scale. Photos
  with the coin on top of the box would remove this assumption. The photos could not be retaken.
- **Accuracy target missed.** The floor on error is the label and 3-D shape noise described in
  `docs/OBJECTIVES.md`, not the model. I did not relabel or retrain to fix it.
- **Camera tilt, distance and lighting** were not varied. Only distance from the frame centre was studied.
- **Training was 7 epochs on CPU**, chosen to finish in about an hour. The val metrics were still improving
  slowly.
- **No separate coin labels.** The coin is found with a Hough circle and an ellipse fit and failed on 2 of 21
  photos.
- **Only 7 test images**, so the test metrics have wide uncertainty.
- **Val photos were used for the accuracy table** (13 of 19), after val picked the checkpoint. Test photos
  were not used for any choice.
