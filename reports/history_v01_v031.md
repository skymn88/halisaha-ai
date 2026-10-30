> Historical README through v0.3.1. For current commands and limits see [README](../README.md). Counts below describe historical runs, not the current test suite.

# Halı Saha Analytics - v0.3.1

Python football video analytics for fixed sideline and fixed corner cameras. Uses the existing COCO YOLO model; no new model has been trained. Person detections are not guaranteed to be players (referees and spectators may appear).

## Latest iteration: detector cache and label-loss fixes

The v0.3 checkpoint is commit `b4be422`, on the existing `feature/tracking-v02` branch. This iteration keeps both player and ball work in scope, following the user's clarification, while preparing player statistics as the first product milestone. Long montage inference is not a routine development step: use short clips, provided CSVs and selected regression frames.

```powershell
# Online: one full-frame detection pass, then independent trackers and diagnostic exports.
python run_tracking.py --video ilk_15_saniye.mp4 --output results/new_online
# No model or video needed: replay the saved raw person observations.
python track_csv.py --detections results/new_online/raw_detections.csv --output results/new_replay
# Controlled old-confidence comparison, using exactly the same detections.
python track_csv.py --detections results/new_online/raw_detections.csv --output results/new_strict --person-floor 0.25 --hold-frames 0
# External original-schema CSV: declare FPS; provide frame count to preserve trailing empty frames.
python track_csv.py --detections provided.csv --fps 30 --frame-count 450 --output results/external_replay
# Player-only operation remains available without invoking ball detection.
python run_tracking.py --video results/scene_clips/scene_000.mp4 --ball-mode off --output results/player_only
```

`detector.py` wraps Ultralytics behind normalized `Detection` records in `detection_io.py`. Full-frame person/ball detection now shares one inference pass, followed by separate thresholds and tracking. `player_tracking.py` accepts normalized observations from any detector and adapts them to the existing BoT-SORT/ByteTrack backend. This is pixel-space tracking; it does not implement metric speed gating, persistent physical identities or offline tracklet merging yet. The detector/tracker adapter separation does not select a project license or remove dependency licensing considerations.

### Why person labels disappeared

The earlier person detector floor (0.25) prevented observations in the tracker's low-score recovery interval (0.10–0.25) from reaching it. Ultralytics' tracking callback also retains only associated tracks when tracks exist and can hide unconfirmed new tracks. The new adapter receives weak observations and shows unmatched strong observations as `P?` with ID -1 rather than hiding them or inventing identities. `--conf` remains the strong/unassigned display threshold (default 0.25); `--person-det-conf` is the raw floor (default 0.10). Tracker initiation/high/low thresholds remain separate configurable options. More observations do not prove more correct identities.

`--label-hold 3` shows a short, explicitly **predicted** dashed gray label when an existing track loses observations; 0 disables it. Predictions never enter `detections.csv`. Their confidence is blank and their source-frame age is exported in `track_states.csv`; exclude them from measurement by default. Predictions can drift or belong to an incorrect identity. A longer tracker buffer alone does not make a person visible through an occlusion.

When per-video geometry is supplied, person filtering uses the bbox foot point before association. An explicit image polygon works without metric calibration. If measured homography correspondences and `pitch_size_m: [width, length]` are provided, the image pitch quadrilateral can instead be derived from the homography. No 25×40 assumption or camera-specific polygon is enabled by default. Fitting a transform still does not independently validate physical distances or speeds.

### Cache and CSV-only replay contract

- `raw_detections.csv`: first eleven original CSV columns retained, with `track_id=-1`, plus `scene_id,detector_pass`. Full-precision model scores/boxes are preserved for replay. These are post-NMS observations, before player association or ball acceptance. Person observations rejected by a configured pitch filter remain visible here for audit.
- `raw_detections.meta.json`: FPS, source frame count/size, stride, source path, scene boundaries and detector floors. Empty frames and scene changes can therefore be replayed even when no rows exist for them.
- `player_diagnostics.csv`: raw observations, pitch exclusions, associations, weak associations, unassigned observations and short predictions for each processed frame.
- `detections.csv`: original leading columns and previous extensions retained; `tracking_state` distinguishes `observed` from `detected_unassigned`. Person ID -1 is not an identity.
- `track_states.csv`: person observed/unassigned/predicted states and ball predictions. Observed ball states remain in `ball_track.csv`, with accepted boxes in `detections.csv`. `BALL?` denotes a current provisional candidate, whose box/reason is in `ball_candidates.csv`; it is not an accepted ball row.

`track_csv.py` reads original eleven-column or extended CSV and optional metadata; incoming track IDs are ignored. Without metadata, FPS must be declared. Inferred frame ranges cannot recover omitted trailing empty frames; pass `--frame-count` where known. External CSV may already contain filtered/tracked data: replay cannot reconstruct discarded observations. CSV requires canonical `person`/`player` labels for person rows. Parquet is planned and is not currently supported.

The CSV-only path does not load YOLO weights. It currently replays people only; ball filtering still requires image/reference evidence. Optional `--video` produces a person review video and torso descriptors; it must be the matching raw input, with the same resolution/frame numbering, not an annotated video. `--camera-config` plus metadata or `--image-size WIDTH HEIGHT` enables video-free geometry filtering. A cached `camera_config.json` is reused automatically. The saved tracker configuration is reused; explicit tracker arguments override it. Replay requires fixed-camera GMC disabled and appearance ReID disabled. No team features can be reconstructed from coordinates alone without descriptors or representative images.

### Ball changes and limits

Competing tentative ball hypotheses now survive the current active hypothesis's motion gate. A consistent challenger can take over after repeated observations, rather than being permanently rejected. Association also considers distance to the last observation, limiting overshoot when a ball stops or turns. Body/foot overlap is a soft ranking penalty, not an unconditional exclusion zone. The white-reference test examines the central crop as well as the whole bbox to avoid diluting a small penalty spot inside a padded box. The exported marking/foreground fractions in this version are the maximum of the whole-box and central-crop fractions; do not reinterpret historical v0.3 values with this definition.

Optional `--ball-search` enables periodic tile acquisition and up to two active/challenger ROIs using the same weights. It is **off by default**: on the current short clip an experimental 268 extra inference passes yielded no retained ROI detections, so no ROI recall improvement is claimed. Search is configurable with `--ball-search-interval`, `--ball-tile-fraction`, `--ball-roi-fraction` and `--ball-crop-imgsz`; a different camera may need a separately reviewed experiment. No moving-white-blob shortcut or new model training was introduced.

The checked example still misselects clothing/footwear at frames 331–332, while the new association accepts the visible ball at 333–334 where v0.3 had no accepted observation. Many small balls still have no usable detector proposal. Accepted row counts are not true-ball recall. Reliable ball tracking will need annotated evaluation and potentially a separate temporal model; no such model is implemented or trained here.

### Validation and environment

31 tests passed, including CSV-only replay and input consistency checks. Online/CSV-only replay agreed in all original eleven columns for all 6,848 person rows; both the combined review video and CSV-rendered person video decoded to 450 frames. See [the v0.3.1 Turkish report](v031_validation.md). The controlled cache replay changes person associated observations from 6,146 to 6,446, scene-scoped IDs from 80 to 75 and spatial transition hints from 18 to 11. The weak path accounts for 296 associated rows. There are no ground-truth identities, so IDF1/HOTA or accuracy improvement is not claimed. Predicted labels and unassigned detections are reported separately.

`constraints-tested.txt` records exact installed direct/runtime versions for an optional constrained install (`pip install -r requirements.txt -c constraints-tested.txt`); it is not a complete transitive lock. The existing requirements remain unchanged. A project LICENSE decision is outstanding; no license was assigned on the owner's behalf. Team classification, Parquet, metric statistics and event recognition remain future milestones, with physical validation required before calling distances/speeds accurate.

## v0.3 checkpoint: scene-aware ball tracking (historical)

Development continues on `feature/tracking-v02`; versions are recorded in commit messages rather than creating a new branch per version. The v0.2 checkpoint is commit `da4c260` and has been pushed to origin. `baseline_tracking.py` remains the original v0.1 implementation.

```powershell
python run_tracking.py --video ilk_15_saniye.mp4 --output results/v03_short
# Broad validation across the long montage: every third source frame, smaller inference size.
python run_tracking.py --video halisaha.mp4 --output results/v03_long_sampled --stride 3 --imgsz 640
# Restore the v0.2 ball filter for comparison (no scene resets in this mode).
python run_tracking.py --video ilk_15_saniye.mp4 --output results/v02_comparison --ball-mode v02
python -m unittest -v test_tracking_helpers test_pitch_tracking
# Replay the old ball filter over the exact saved proposals; no extra YOLO inference.
python compare_ball_filters.py --video ilk_15_saniye.mp4 --output results/v03_short --review-region 165 126 180 139
```

`halisaha.mp4` is approximately 146.4 seconds / 4,388 frames and contains multiple fixed-camera clips with hard cuts and gradual transitions. It is not one calibrated camera view. Default scene analysis found nine segments. Thumbnail differences over a 30-frame window detect transitions; boundaries are approximate, not annotated cut locations. Very short shots, similar views, flashes and camera movement can defeat this heuristic. Tracker state resets at each detected boundary; IDs are local to a scene. Do not interpret a montage as one match or join identities across its cuts.

For each segment, `pitch_context.py` samples 41 frames, trims transition margins and builds a median reference plus a white-reference mask. A candidate overlapping reference white pixels, with insufficient change from the reference, is marked `static_marking`. This catches isolated white spots that the old surrounding-ring cue missed and rechecks even previously confirmed hypotheses. Thresholds do not depend on whether the candidate itself moved. A real ball that looks different from the reference can remain stationary and be accepted, including over a marking. The white mask is an appearance cue, not a semantic reconstruction of pitch lines.

`ball_tracking.py` chooses at most one observed ball per processed frame using confidence, foreground change and a bounded motion gate. It confirms after repeated observations, retains a hypothesis through short gaps and reacquires after expiry. A simple velocity estimate assists association; observation of a stopped ball clears prediction drift. No coordinates are emitted for unseen frames, and there is no physical 3D ball model. False clothing detections, fast motion, occlusion and tiny missed balls remain possible.

This is **offline** processing: scene segmentation and reference construction inspect future frames from the same clip. It is not a causal/live tracker or an independent evaluation dataset. A ball occupying one position for most reference samples can become part of the background. Lighting changes, compression, foreground people and gradual transitions can weaken the reference test. A ball visually indistinguishable from a reference marking remains ambiguous; the filter cannot guarantee correct treatment of that case.

### Additional options

| Option | Default | Meaning |
| --- | --- | --- |
| `--ball-mode` | pitch | New filter; `v02` retains the original temporal helper and unsplit video behavior |
| `--background-samples` | 41 | Uniform reference samples per detected segment |
| `--background-difference` | 30 | Per-pixel maximum channel difference, 0–255 scale |
| `--foreground-min` | 0.25 | Minimum changed-pixel fraction to overcome the reference-white caution |
| `--scene-threshold` | 23 | Mean thumbnail color change over the scene window |
| `--scene-window` | 30 | Source frames in lagged scene comparison |
| `--scene-min-frames` | 60 | Minimum segment length; tiny final segments are not split |
| `--threads` | 1 | PyTorch CPU threads; configurable for the local machine |
| `--camera-config` | absent | Optional JSON with per-scene ROI/calibration |

Existing `--ball-hits`, `--ball-gap` and `--ball-radius` also configure the new filter. `--ball-movement` only applies to `--ball-mode v02`. The new foreground difference and white-reference thresholds are heuristics, not trained probabilities. Source FPS/resolution are preserved in the output video apart from stride-adjusted FPS. The long sampled validation uses about 9.99 output FPS and does not test every source-frame detection or continuity at full rate.

### New/extended outputs

- `ball_track.csv`: one row per processed frame: `frame,time_s,scene_id,ball_id,state,confidence,center_x,center_y`. `observed` means an accepted detection; `tentative` means confirmation is pending; `occluded` means an unexpired hypothesis without an accepted observation; `missing` means none remains. Non-observed rows have blank positions and confidence. `v02_multi_hypothesis` is a legacy-mode placeholder, not a selected ball.
- `ball_candidates.csv`: all proposals with reasons (`accepted`, `static_marking`, `temporal_unconfirmed`, `motion_gate`, `alternative`, `outside_pitch`). Added scene, reference-white fraction, changed-pixel fraction and prediction innovation in pixels. `white_marking_cue` in pitch mode records the combined static-reference caution, whereas legacy mode retains its old ring cue. Rejected/alternative proposals have ball ID -1 unless selected pending confirmation; no assigned identity is implied.
- `detections.csv`: original columns retained; appended `scene_id,pitch_x_m,pitch_y_m`. Metric columns are populated only for people when measured calibration is supplied. Ball metric positions are blank because an airborne ball is not on the pitch plane.
- `id_changes.csv`: appended scene ID. Events never link across detected scene boundaries.
- `scenes.json`: exclusive end/source-frame boundaries, calibration status and reprojection error; current runs also export the fitted homography when present.
- `pitch_context/scene_XXX_background.png` and `scene_XXX_markings.png`: references for visual auditing.
- `summary.json`: added raw ball candidate count, rejection/acceptance reason counts and scene count. Unique person IDs count distinct `(scene_id, track_id)` pairs, not actual individuals.
- `comparison.json` and `v02_ball_replay.csv`: produced by `compare_ball_filters.py`; original v0.2 postprocessing on the exact saved candidates. This isolates ball filtering from detector changes, but is not an end-to-end v0.2 player-tracking rerun. The optional review rectangle counts proposal centers in an explicitly supplied scene/region; it is not a general accuracy score.

### Camera calibration

`calibration.py` accepts per-scene pixel polygons and optional image-to-pitch point pairs. Copy `configs/camera.example.json`, fill in actual image coordinates and corresponding measured pitch coordinates in meters, then pass `--camera-config your_camera.json`. The supplied template contains empty arrays, so it enables no ROI or metric conversion. There are no invented field dimensions or assumed corner correspondences.

The file must match source `image_size: [width, height]`; scene keys must exist in `scenes.json`. A polygon needs at least three non-collinear points. Homography fitting needs at least four distinct paired non-collinear points. Finite/invertibility and pixel reprojection-RMSE checks reject invalid fits. Reprojection error is measured on the fitting points: four exact correspondences do not independently verify calibration. Use extra held-out pitch landmarks to validate metric measurements. Fitted matrices and optional input configuration are saved with outputs. Scene boundaries and thus scene keys can change with scene parameters or video edits.

No real-world calibration has been validated on these clips because measured dimensions/correspondences were not provided. Lens-distortion correction, automatic field-line fitting, calibrated speed/distance/heatmaps and 3D ball reconstruction remain future work. An ROI can suppress off-pitch clutter but may exclude an airborne ball above the ground silhouette; configure conservatively.

## Environment and execution

Python 3.10–3.12 recommended. Install using `pip install -r requirements.txt` in your existing virtual environment. Validation used Python 3.10 and Ultralytics 8.4.192. Existing dependency requirements are unchanged. Model weights must be available locally or Ultralytics may download them.

```powershell
python baseline_tracking.py --video ilk_15_saniye.mp4 --output results/baseline_v01
python run_tracking.py --video ilk_15_saniye.mp4 --output results/current_run
python -m unittest -v test_tracking_helpers
```

`baseline_tracking.py` preserves the original v0.1 script verbatim for reproducible comparisons. Use a fresh output directory for each run. Example directory names are illustrative; repeating a command with the same directory overwrites its output files.

## v0.2 changes (historical behavior)

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

## Original v0.2 outputs (extended by v0.3 above)

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

## Iterating on separate scenes

Prefer a selected short scene for everyday development; do not repeatedly infer over the full montage. The completed long sampled run has already produced reusable boundaries:

```powershell
# Export all original-resolution/FPS raw clips without YOLO inference.
python split_scenes.py --video halisaha.mp4 --scenes results/v03_long_sampled/scenes.json --output results/new_scene_clips
# Or export only selected scenes into a fresh folder.
python split_scenes.py --video halisaha.mp4 --scenes results/v03_long_sampled/scenes.json --output results/selected_clips --scene-ids 0 6
# Then analyze one raw clip, using a fresh output folder.
python run_tracking.py --video results/scene_clips/scene_000.mp4 --output results/scene000_next
```

`--scenes` reuses boundaries; omitting it runs scene scanning without YOLO. `split_scenes.py` validates coverage, exports exact frame ranges, decodes every exported clip to verify counts and refuses to overwrite existing output files/manifest. Clips are re-encoded as mp4v and do not retain audio. Re-encoding can change detection results; compare algorithms on the same input. Source videos stay unchanged. Boundaries are heuristic, especially across dissolves.

Completed exports: `results/scene_clips/scene_000.mp4` through `scene_008.mp4` (4,388 total raw frames), and annotated sampled clips under `results/v03_long_sampled/scene_clips/` (1,463 total frames). Both folders contain `manifest.json` for parent-scene/source-frame mapping. A raw clip local frame maps to `source_start_frame + frame`; in the sampled annotated clips use the recorded original first frame plus local frame × original stride. New analysis of a raw clip typically starts again at scene ID 0. Analyze raw clips, not annotated review videos.

## Completed v0.3 validation

| Metric | Short, stride 1 / imgsz 960 | Long, stride 3 / imgsz 640 |
| --- | ---: | ---: |
| Source / processed frames | 450 / 450 | 4,388 / 1,463 |
| Raw ball proposals | 396 | 280 |
| v0.2 filter replay accepted rows, same proposals | 375 | 204 |
| v0.3 accepted rows | 52 | 62 |
| Static-reference marking rejections | 334 | 168 |
| Scene-scoped unique person IDs | 80 | 438 |
| Possible spatial ID transitions | 18 | 54 |

17 unit tests passed. Both output videos and all exported clips decoded completely. These counts are not precision/recall or proof of identity stability. The reviewed white-spot region had 322 → 0 accepted rows in the short run but 5 → 2 in the long sampled run: box geometry/resolution can still let markings escape the current white-fraction threshold. A clothing false positive was visually confirmed at short frame 331. No reliable continuous-ball-tracking claim is made. Real measured camera calibration and labeled evaluation remain outstanding. See [the detailed Turkish validation report](v03_validation.md) for decisions, per-scene results, examples and limits.
