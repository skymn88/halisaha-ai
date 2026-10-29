# Halı Saha Analytics — v0.2

Python football video analytics for fixed sideline and fixed corner cameras. Uses the existing COCO YOLO model; no new model has been trained. Person detections are not guaranteed to be players (referees and spectators may appear).

## Environment and execution

Python 3.10–3.12 recommended. Install using `pip install -r requirements.txt` in your existing virtual environment. Validation used Python 3.10 and Ultralytics 8.4.192. Existing dependency requirements are unchanged. Model weights must be available locally or Ultralytics may download them.

```powershell
python baseline_tracking.py --video ilk_15_saniye.mp4 --output results/baseline_v01
python run_tracking.py --video ilk_15_saniye.mp4 --output results/v02
python -m unittest -v test_tracking_helpers
```

`baseline_tracking.py` preserves the original v0.1 script verbatim for reproducible comparisons. Existing `results/` outputs are not overwritten by the example validation commands. Output directories should be distinct per run.

## Changes

- Retained YOLO person detection and persistent BoT-SORT/ByteTrack tracking, original CSV fields, video annotation and foot coordinates.
- Ball detection now uses a separate inference pass with its own confidence threshold. This costs additional inference time but avoids discarding ball candidates because of player tracker initiation thresholds.
- `tracking_helpers.py` associates ball observations by nearest center within a configurable pixel radius. Candidates need repeated observations before acceptance. Expired hypotheses are discarded. Only observed boxes are exported; no predicted boxes fill missing frames, and initial confirmation frames are not backfilled.
- White pixels inside and around a candidate form a pitch-marking caution cue. Unconfirmed candidates with that cue require measurable displacement before acceptance. Confirmed balls can remain stationary, including on a white line. Stationary candidates away from the cue can confirm normally. This is a heuristic: a ball initially stationary on a marking can be suppressed, isolated penalty spots can escape the cue, camera motion or detection jitter can clear it, and multiple hypotheses can be accepted. It does not establish ball identity or guarantee false-positive removal.
- Default player track buffer increased from 30 to 60 processed frames to tolerate longer occlusions. Global motion compensation is disabled for the fixed-camera use case. These are tuning choices, not proven stability improvements; a longer buffer can also retain incorrect associations. Tracker parameters are saved in each output directory.
- `IDTransitions` records new IDs appearing near recently disappeared IDs. These are possible spatial transitions, not measured ground-truth identity switches. Existing-ID swaps cannot be measured by this method. No IDs are silently merged.
- `jersey_feature` exports median OpenCV Lab color from a central torso crop, preparing future unsupervised team grouping without assuming jersey/vest colors. No team assignment is made. Skin, grass, occlusion and mixed-color jerseys can contaminate features; a validated grouping stage remains future work.
- Both camera angles use the same reusable detection and tracking helpers. No field layout, team palette, homography, metric speed or distance is assumed. Pixel thresholds require tuning for video resolution and perspective. This implementation is for fixed cameras.
- Added argument validation, output summary, and capture/writer cleanup on failure.

## Configuration

Run `python run_tracking.py --help` for all options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--conf` | 0.25 | Person confidence |
| `--ball-conf` | 0.15 | Ball detector confidence |
| `--ball-hits` | 3 | Observations required to confirm a hypothesis |
| `--ball-gap` | 8 | Maximum gap in source frames between ball observations |
| `--ball-radius` | 45 | Maximum center association distance, pixels |
| `--ball-movement` | 4 | Displacement from initial position to clear marking caution, pixels |
| `--track-buffer` | 60 | Player lost-track buffer, processed frames |
| `--track-high-thresh` | 0.25 | Player first-stage threshold |
| `--track-low-thresh` | 0.1 | Player second-stage threshold |
| `--new-track-thresh` | 0.25 | Player initiation threshold |
| `--match-thresh` | 0.8 | Tracker association threshold |
| `--id-gap` / `--id-radius` | 8 / 35 | Source-frame gap / pixel radius for possible ID transitions |
| `--tracker` | botsort.yaml | Also supports bytetrack.yaml |
| `--imgsz` / `--stride` | 960 / 1 | Inference size / source-frame sampling |

Use stride 1 for continuity. Larger strides alter association behavior; ball and ID gaps remain in source frames while the underlying player tracker advances per processed frame. Relaxing person confidence to the low tracker threshold can enable more second-stage recovery but increases candidate noise; tune on annotated data.

## Outputs

- `tracked.mp4`: annotations with confidence and class-specific ID. Ball IDs and person IDs occupy separate namespaces; use object type together with ID.
- `detections.csv`: original frame/time/ID/class/confidence/box/foot columns plus `object_type` (`player` or `ball`) and `jersey_l/a/b`. Ball foot columns retain box-bottom coordinates for schema compatibility; they are not physical contact points. Untracked person IDs are -1.
- `ball_candidates.csv`: every ball proposal with confidence, floating-point coordinates, hypothesis ID, acceptance reason and white-marking cue. Useful for auditing rejected proposals.
- `id_changes.csv`: possible old/new person-ID transitions, gap and center distance.
- `summary.json`: accepted ball detection rows, unique nonnegative person IDs, possible transitions, processed frames and invocation parameters.
- `tracker.yaml`: resolved player tracker configuration.

## Validation and limits

The test input `ilk_15_saniye.mp4` has 450 frames, 640×360 resolution and approximately 29.97 FPS. Baseline and v0.2 use the same local `yolo11n.pt`, image size 960 and stride 1. The ball confidence threshold differs intentionally (baseline 0.25, v0.2 0.15); tracker settings and ball inference differ, so counts are operational comparisons, not controlled accuracy measurements.

Full-run counts are recorded below after validation. Five synthetic helper tests cover stationary-ball acceptance, white-marking caution, stopping after confirmation, gap expiry, one-to-one hypothesis association, no fabricated detections, possible ID transitions and color descriptor shape. These verify implemented rules, not real-world accuracy.

There is no annotated evaluation set. More/fewer ball detections or person IDs do not prove higher accuracy or better identity stability. Precision/recall, false marking detections and actual ID switches require reviewed ground-truth boxes and identities. Test both camera angles before deployment. Further improvements may require an annotated football dataset; do not fine-tune without one. Camera calibration needs known pitch correspondences (at least four for a planar homography), and lens correction where necessary.

Check Ultralytics and model weight license terms before redistribution or commercial use.

### Completed validation — 2026-10-29

| Metric | Original v0.1 | v0.2 |
| --- | ---: | ---: |
| Decoded annotated video frames | 450 | 450 |
| Person detection rows | 6,143 | 6,146 |
| Ball detection rows | 316 | 375 |
| Unique nonnegative person IDs | 82 | 80 |
| Possible spatial ID transitions | 18 | 18 |

Baseline transition hints were computed after the run using the same `IDTransitions` helper and default gap/radius as v0.2. V0.2 had 396 raw ball proposals: 375 accepted, 19 withheld pending temporal confirmation and 2 withheld with the white-marking cue. These rejected proposals have not been ground-truth labeled. The lower ball threshold and separate inference path prevent attributing the count increase to temporal filtering alone. The ID counts and equal transition-hint counts do not establish improved player stability.

Both video files were decoded end to end successfully. An annotated sample at frame 300 was visually checked; this is output inspection, not an accuracy evaluation. All five helper tests passed, Python compilation and CLI help succeeded, and `git diff --check` reported no whitespace errors. The baseline script was verified against the original tracked source. Outputs are in `results/baseline_v01/` and `results/v02/` (ignored by git). No training, commit or push was performed.
