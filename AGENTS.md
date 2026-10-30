# Project guide for coding agents

## Purpose and scope

Build a Python-based amateur football video analytics system for recordings from fixed sideline and fixed corner cameras. Preserve the working tracking pipeline while adding small, testable capabilities. This file distinguishes existing behavior from planned work; a roadmap item is not an implemented feature.

Long-term product goals:

- Maintain persistent player identities throughout a match, including occlusions.
- Classify teams using jersey/vest colors and jersey numbers without assuming specific team colors.
- Track the football reliably, including when it stops moving.
- Calculate player distance, speed and heatmaps in calibrated pitch coordinates.
- Detect passes, shots, goals, tackles and possession changes.
- After a match, let users identify themselves from representative player images linked to match identities.

## Current implementation

The repository now contains a v0.4 evaluation/review foundation on v0.3.1 checkpoint `6b67f79`. Checkpoints `6b67f79`, `b4be422` (v0.3) and `da4c260` (v0.2) were pushed to origin. This is not an accuracy milestone. Continue on `feature/tracking-v02`; use versioned commit messages instead of opening a branch for each version unless requested. Check the actual branch and working tree before starting new work. Existing uncommitted changes must be preserved.

| File | Actual responsibility |
| --- | --- |
| `baseline_tracking.py` | Preserved v0.1 CLI: one YOLO tracking pass for people and sports balls; annotated video and original detection CSV. Keep as a comparison baseline. |
| `run_tracking.py` | Online v0.3.1 orchestration: normalized raw cache, separate trackers, observations versus predictions, legacy ball mode and player-only mode. |
| `detection_io.py` / `detector.py` | Detector-neutral records/original CSV transport and a replaceable Ultralytics adapter; one shared full-frame detection pass. |
| `player_tracking.py` / `track_csv.py` | Person BoT-SORT/ByteTrack adapter and CSV-only replay without a required video or detector inference. |
| `annotations.py` | Different rendering for observations, unassigned candidates and explicitly predicted labels. |
| `ball_detection.py` | Optional tiled/ROI experiment; disabled by default because no ROI recall gain was measured on the current clip. |
| `test_detection_tracking.py` | CSV, weak-observation recovery, visibility/prediction, foot filtering and ball reacquisition regressions. |
| `constraints-tested.txt` | Exact reported local direct/runtime versions; not a full transitive dependency lock. |
| `ball_tracking.py` | One observed-ball selection, reference/ROI checks, motion gate, confirmation and gap expiry. |
| `split_scenes.py` | Validated raw clip export using saved or detected scene boundaries, source-frame manifest, no-overwrite and decoded-frame checks; no YOLO inference. |
| `test_split_scenes.py` | Clip boundary/source mapping and no-overwrite/invalid-manifest regressions. |
| `pitch_context.py` | Offline scene detection, per-scene median backgrounds and white-reference/foreground evidence. |
| `calibration.py` | Optional per-scene ROI and measured homography, point validation, reprojection diagnostics and player ground-plane projection. |
| `configs/camera.example.json` | Empty geometry template; no supplied real calibration or invented dimensions. |
| `compare_ball_filters.py` | Replay original v0.2 ball filtering on saved v0.3 proposals; output integrity and per-scene comparison. |
| `test_pitch_tracking.py` | Pitch-aware stationary-ball/marking/gap/motion tests and synthetic scene/calibration checks. |
| `tracking_helpers.py` | `TemporalBall`, `white_marking`, `IDTransitions` and `jersey_feature`; pixel-space logic without pitch calibration. |
| `test_tracking_helpers.py` | Five synthetic unittest cases for helper behavior. These are not an annotated football benchmark. |
| `requirements.txt` | `ultralytics>=8.3,<9`, `opencv-python>=4.8`, `numpy>=1.24`, `pandas>=2.0`. Pandas is declared but not used by the current tracking scripts. |
| `README.md` | Setup, CLI parameters, output formats, limitations and previously recorded validation results. Update when behavior changes. |

Local test assets include `ilk_15_saniye.mp4`, `halisaha.mp4` and `yolo11n.pt`. The longer file is a multi-camera montage, not a single fixed view. Video files, model weights, `results/` and `runs/` are ignored by git and may be absent in another checkout. No annotated training or evaluation dataset was found in the inspected repository.

### Runtime architecture

1. Scan source frames for hard cuts and gradual transitions using lagged thumbnail change. Default pitch mode builds separate offline median backgrounds and white-reference masks for detected segments. These use future frames, not causal/live state. `--ball-mode v02` skips scene analysis and retains the legacy filter.
2. Load optional per-scene ROI/measured calibration. With no measured configuration, metric columns remain blank; no field dimensions are assumed.
3. Use a replaceable detector to write pre-association normalized observations, then run person tracking and separate ball postprocessing. Person detector floor is 0.10, with strong/unassigned display threshold 0.25. One full-frame inference handles both classes. Existing confidence/tracker settings remain configurable; CPU thread count is configurable (default 1).
4. In pitch mode, suppress candidates that match reference-white areas without sufficient foreground change, even if previously confirmed. Rank other candidates by confidence, foreground evidence and bounded velocity-based association. Accept at most one observed ball per processed frame after repeated observations. Stationary balls can pass when distinguishable from the reference. Preserve original `TemporalBall` behavior in legacy mode.
5. Reset ball/player tracker state and ID diagnostics on scene boundaries. IDs are local to each scene; count identities by `(scene_id, track_id)`.
6. Export video, observations, candidate decisions, per-frame ball states, scene/reference diagnostics and summary. No predicted coordinates fill unseen frames. Team features and player-ID hints remain diagnostics, not identities/team assignments.

Ball confirmation counts observations across allowed gaps, not necessarily consecutive frames. Ball/ID gaps and scene windows use source frames; the underlying player tracker advances per processed frame. Prefer stride 1 for continuity. Long-video sampled validation is explicitly labeled and does not establish full-rate tracking quality.

### Output contract

- `tracked.mp4`: processed frames with class-specific IDs and confidence annotations.
- `detections.csv`: `frame,time_s,track_id,class_name,confidence,x1,y1,x2,y2,foot_x,foot_y,object_type,jersey_l,jersey_a,jersey_b`.
- `ball_candidates.csv`: every ball proposal, confidence, box, hypothesis ID, status and white-marking cue.
- `id_changes.csv`: old/new person IDs, source-frame gap, center distance and `possible_spatial_transition` evidence.
- `summary.json`: accepted ball detection count, unique person IDs, possible transitions, processed frames and invocation parameters.
- `tracker.yaml`: resolved player tracker configuration.
- v0.3 appends `scene_id,pitch_x_m,pitch_y_m` to detections and scene ID to ID events. Metric positions apply only to player foot estimates with supplied calibration; no ball-plane position is claimed.
- `ball_track.csv`: observed/tentative/occluded/missing state per processed frame; unobserved coordinates are blank.
- `scenes.json` and `pitch_context/`: scene boundaries, calibration diagnostics and per-scene reference images/masks.
- Candidate CSV and summary include background/foreground evidence and decision counts. See README for full versioned semantics.
- `comparison.json` and `v02_ball_replay.csv` are optional analysis outputs from the replay tool.

Frames are zero-based; timestamps use source frame / source FPS. Coordinates are image pixels. Person foot coordinates are the box-bottom center; ball foot columns are schema-compatible box-bottom coordinates, not physical contact points. Person ID -1 means untracked. Ball and person IDs have separate namespaces: identify tracks by `(scene_id, object_type, track_id)` together.

## Known problems and evidence limits

- **Priority issue:** penalty spots and white pitch markings caused persistent false ball detections in v0.2. v0.3 adds reference-based suppression; it is a heuristic, not a general solution or accuracy guarantee. Foreground clothing false positives and true-ball misses remain possible.
- In legacy v0.2 mode, isolated white spots can escape the surrounding-white cue and jitter can satisfy movement confirmation; multiple hypotheses can be accepted. Pitch mode revalidates against the reference and selects at most one observed candidate, but can select the wrong foreground object.
- A real ball first observed stationary on a white marking may be suppressed. Never solve the marking problem by rejecting all stationary objects or by permanently masking every marking location.
- Generic COCO `sports ball` detection is not football-specific. Small, distant or occluded balls can be missed; pixel thresholds vary with perspective and resolution.
- Tracker IDs can fragment or switch. `IDTransitions` only flags a previously unseen ID near a recently disappeared ID; it does not verify identity, detect all swaps or merge tracks. Unique person IDs are not a count of actual players.
- The person class can include referees and spectators. Torso color features can include skin, grass, occlusion and mixed jersey colors. There is no team classifier or jersey-number recognition yet.
- Optional ball crop inference increases runtime; the default v0.3.1 full-frame pass is shared. Tracking parameter changes are tuning choices, not proven stability improvements.
- Optional calibration mathematics/ROI support exists, but no measured real-camera calibration is supplied or validated. There is no lens-distortion correction, semantic pitch map, metric trajectory analysis, heatmap, validated event recognizer or possession model, identity gallery or user identification interface yet. v0.4 adds only explicitly unvalidated event/possession proposals and review.
- Offline median references can absorb a ball stationary through most samples; lighting changes, transitions and camera motion can invalidate appearance evidence. A real ball indistinguishable from reference white pixels remains ambiguous. Scene boundaries are heuristic and need inspection.
- Fitting-point reprojection error is not independent calibration validation. Planar transforms do not recover airborne-ball 3D position.
- The shared pixel-space pipeline is intended for both fixed camera angles, but validation on the single local clip does not establish performance across both angles.

The README records this completed comparison on `ilk_15_saniye.mp4` (450 frames, 640×360, approximately 29.97 FPS), using local `yolo11n.pt`, image size 960 and stride 1:

| Metric | v0.1 baseline | v0.2 |
| --- | ---: | ---: |
| Ball detection rows | 316 | 375 |
| Person detection rows | 6,143 | 6,146 |
| Unique nonnegative person IDs | 82 | 80 |
| Possible spatial ID transitions | 18 | 18 |
| Decoded output video frames | 450 | 450 |

These are historical operational counts, not accuracy scores or results from every future run. v0.2 accepted 375 of 396 ball proposals; 19 were pending temporal confirmation and 2 had a marking caution. Ball confidence and inference paths differ between versions. Do not attribute a count change to improved accuracy. Reviewed boxes and identities are needed to measure precision/recall and actual ID switches.

## Coding and development conventions

- Inspect relevant code and the current diff before editing. Preserve unrelated work and the original baseline; avoid large unnecessary refactoring.
- Keep Python 3.10 compatibility and the existing environment. Prefer the current dependencies; justify dependency additions and avoid upgrades unrelated to the task.
- Use snake_case functions/variables, PascalCase classes, four-space indentation, and concise docstrings for public helper behavior. Use explicit units for thresholds and coordinate systems.
- Keep the CLI as orchestration and reusable vision/tracking logic in focused helpers. Keep camera-specific geometry/configuration separate from shared detection logic.
- Expose tunable tracking/ball parameters through validated configuration or CLI arguments. Avoid hard-coded team colors, camera angles, pitch dimensions or marking positions.
- Preserve existing CSV columns, class/object semantics, timestamps and ID namespaces. Document extensions and any necessary compatibility changes.
- Maintain diagnostic exports so rejected ball candidates and uncertain identity associations can be reviewed. Keep observations distinguishable from any future predictions or inferred events.
- Use `pathlib`, UTF-8 text/CSV output and reliable video resource cleanup. Use a fresh output directory per validation run: existing filenames are overwritten when a directory is reused.
- Add focused tests for meaningful behavior changes, especially stationary balls, marking ambiguity, occlusion/gaps and identity association. Synthetic tests verify rules, not match accuracy.
- Do not train or fine-tune a model unless an annotated training dataset is available. Split future training/validation data by match to avoid frame leakage.
- Report modified files, configuration, validation results and unresolved limitations. Do not claim accuracy or stability improvements without evidence. Do not commit or push without user authorization. The user authorized and completed the v0.2 checkpoint push; version numbers belong in commit messages on the existing branch.

## Local commands and validation

Python 3.10–3.12 is recommended in the README; historical validation used Python 3.10 and Ultralytics 8.4.192. Requirements specify ranges, so record the actual environment for comparisons.

```powershell
pip install -r requirements.txt
python run_tracking.py --help
python -m unittest -v test_tracking_helpers test_pitch_tracking
python -m py_compile run_tracking.py tracking_helpers.py baseline_tracking.py
python baseline_tracking.py --video ilk_15_saniye.mp4 --output results/baseline_comparison
python run_tracking.py --video ilk_15_saniye.mp4 --output results/candidate_comparison
git diff --check
```

Install dependencies only when necessary. Use available local weights to avoid an unintended download. Prefer one exported raw scene clip for routine tracking development. The user explicitly prefers splitting the long montage instead of repeatedly processing it. Reuse `results/scene_clips/` and its manifest, or run `split_scenes.py` with the saved `scenes.json`; only repeat the full montage when a broader regression is needed. Exported clips are re-encoded and silent; compare methods on the same input. Full-frame runs of selected clips are appropriate for continuity changes; documentation-only work does not require rerunning inference. For behavioral comparisons, report ball rows, unique nonnegative person IDs, possible ID transitions and processed/decoded frame counts, inspect annotations and candidate diagnostics, and document parameter differences. Update README validation evidence after meaningful tracking changes.

## Development roadmap and validation status

### v0.3: implemented foundation and remaining validation

The code now includes offline scene-specific references, static-marking checks, one-ball motion association, optional measured calibration, diagnostics and synthetic tests. Remaining work includes independently reviewed ball boxes/identities, held-out real-camera calibration checks, lens correction and robust field geometry. The objectives below describe validation and further work, not a completed accuracy milestone.

1. Review candidate CSVs and representative frames to label true balls, penalty spots and line false positives. Include moving and stopped balls on/off markings, occlusions and both camera angles.
2. Add per-camera pitch configuration: visible pitch region, known pitch dimensions, image-to-pitch correspondences and marking geometry. Keep configurations reusable for sideline and corner recordings rather than hard-coding one clip.
3. Implement and validate planar homography using at least four non-collinear point correspondences; handle lens distortion where needed. Report calibration/reprojection quality before using metric coordinates. A planar pitch transform is not a valid 3D position estimate for an airborne ball.
4. Combine pitch/marking context, appearance and temporal evidence to rank or reject ball hypotheses. Treat markings as evidence rather than unconditional exclusion zones; retain a route for real balls to stop on a line or penalty spot.
5. Add focused regression tests and reviewed-clip comparisons against v0.1/v0.2. Measure marking false positives and true-ball recall, including stationary-ball cases. Document what remains unresolved.

### Subsequent milestones

1. **Persistent identities:** combine motion, appearance and tracklet association; evaluate occlusions, re-entry, fragmentation and actual ID switches with identity annotations.
2. **Teams and numbers:** aggregate jersey/vest colors over reliable observations, infer team groupings without a fixed palette, add number recognition and handle uncertain/unknown assignments.
3. **Metric trajectories:** project validated player ground-contact estimates to calibrated pitch coordinates; handle gaps and uncertainty before calculating distance, speed and heatmaps.
4. **Match events:** build possession and event logic for passes, shots, goals, tackles and possession changes from validated trajectories and labeled events. Record confidence and ambiguous cases.
5. **Post-match identification:** select representative player images, link them to persistent match identities and provide a user-facing self-selection flow. Resolve fragmented tracks and uncertain matches explicitly.

Keep implementation incremental. Do not mark a milestone complete merely because a feature descriptor, diagnostic or heuristic exists.

## Current v0.3 validation checkpoint

See `reports/v03_validation.md` for detailed evidence. Short run: 396 proposals, 52 accepted ball rows versus 375 for replayed v0.2, 80 person IDs, 18 possible transitions, 450 decoded frames. Long sampled run: stride 3/imgsz 640, 280 proposals, 62 accepted rows versus 204 for replayed v0.2, 438 scene-scoped person IDs, 54 possible transitions and 1,463 decoded frames from 4,388 source frames. Seventeen tests passed. Nine raw clips and nine annotated sampled clips were exported and decoded successfully.

Do not overstate these results: the reviewed white-spot region improved from 322 accepted rows to 0 in the short run, but still had 2 accepts versus 5 in the differently configured long run. A white-reference fraction below the threshold can let markings escape. Foreground clothing and motion association also caused a visually confirmed error at short frame 331. These are concrete regressions to address using selected clips; counts alone do not establish ball accuracy.

## User-agreed direction and this iteration

- Both ball and person improvements are important now; the user explicitly clarified this after proposing a player-statistics-first product milestone. Do not silently remove ball work or make ball a prerequisite for player statistics.
- The user approved the bounded architecture step: canonical raw CSV plus the same player tracker replayed without YOLO. Do not expand this approval into a wholesale rewrite, metric tracker, team engine or model training. Discuss a short plan before major architectural changes.
- External detection CSVs and representative frames are a first-class workflow. Do not require videos or rerun detectors merely to tune association. Preserve the original eleven leading CSV fields, ignore imported track IDs when retracking, and retain metadata for empty frames/FPS/stride/scene boundaries. Parquet is future work.
- Person label losses have at least two evidenced causes: the old 0.25 detector floor removed weak recovery observations, and the original tracking callback could hide unassigned/unconfirmed strong observations. Keep these cases distinct from no detector observation. Use `P?` for real unassigned observations and dashed `predicted` labels for bounded estimates; predictions never become observation rows.
- Cameras are usually fixed but location, height, lens and pitch vary. Use video/scene-specific measured calibration and configured dimensions; never silently assume 25×40 m, one camera angle or a fixed team palette. Foot-point pitch filtering now precedes player association. Lens correction remains future work.
- Known 5v5–7v7 roster sizes and jersey colors can inform future association; no top-k player cap, goalkeeper classification, team assignment or jersey-number OCR exists yet.
- Persistent identities, offline forward/backward tracklet merging and meter-space speed gating are future work. Validate calibration with held-out landmarks and a measured walking test before claiming real distance/speed accuracy. No distance, speed or heatmap product is implemented in this iteration.
- Evaluate annotated clips per angle using IDF1/HOTA when identities are available; current counts/transition hints do not substitute for those metrics. Do not claim true ball recall from accepted proposal counts.
- Keep a future multi-frame ball model independent of the player pipeline. Current crop-search experiments are opt-in, not a proven recall improvement; do not enable costly full-montage searches by default.
- Keep dependency compatibility; use the optional tested-version constraints when reproducing this environment. A project LICENSE is not selected: do not assign one without the owner's decision or imply the detector adapter changes dependency licenses.
- Historical authorization for the v0.3/v0.3.1 commits and push was fulfilled. The owner subsequently explicitly requested committing and pushing this v0.4 evaluation/review checkpoint to origin on the existing branch. This is not blanket authorization for future changes.

## v0.3.1 validation checkpoint

31 tests passed. On the same 450-frame input and identical person cache, strict 0.25-floor replay gives 6,146 associated rows / 80 IDs / 18 spatial transition hints; the 0.10 weak-recovery path gives 6,446 associated rows / 75 IDs / 11 hints, plus 402 unassigned observations and 93 separate predicted labels. Online and CSV-only replay match all original eleven columns for all 6,848 person observation rows. These are not ground-truth identity metrics. The original schema is supported; Parquet is explicitly not implemented.

The same 396 ball candidates yield 52 accepted observations under checkpoint b4be422 and 54 under the new filter. Frames 333–334 recover the visible ball, but the known clothing/footwear error at 331–332 remains. Many missed proposals remain unsolved. The long montage was not reinferred: only cached failure frames 78/81 were reviewed, and the new central-white test rejects those two static markings. Tiled/ROI experiments did not justify their cost on this clip and are disabled by default. See reports/v031_validation.md; final outputs are under results/v031_final, with CSV-only replay under results/v031_final_csv.

## v0.4 evaluation and human review foundation

- Reuse `ball_cache.py` / `replay_ball.py` for image-free pitch-ball replay. The original selection rules are unchanged; image and cached paths share association. New runs export full precision evidence/feature contracts. Legacy CSV import declares rounded fractions and requires an explicit feature definition. Recompute innovations; never replay saved selection as new evidence.
- `review_schema.py`, `prepare_review.py`, `review_store.py`, `review_app.py` and `review_ui.html` define separate frame/detection/identity/event labels, localhost review and append-only revisions. Never count proposal or ai_reviewed records as human ground truth.
- `evaluate_review.py` separates raw ball candidate coverage from selection precision/recall and source-pixel error. Config tolerance is frozen/hashed. Report each window/angle/scene, cadence and support; uncertain/unreviewed are not negatives.
- `audit_players.py` exports weak/strong overlapping pairs without suppressing boxes. High overlap is not a proven duplicate. `event_candidates.py` emits only unvalidated control/salience hypotheses, with source-time hysteresis, unknown/contested states and cut exclusions. No team engine or automatic event statistics was added.
- `export_trackeval.py` only exports complete human identity windows to a reference tool. No homegrown IDF1/HOTA and no installed reference dependency.
- Local pilot: 360 raw PNGs, 13 raw clips/1,350 decoded frames, zero human labels. Historical caches differ in stride/resolution; the montage split is not a proven independent-match split. Human angle/replay verification remains required.
- CSV-only legacy and full precision replay both match all saved 396 candidate decisions and 450 states (54 accepts). This is operational parity, not accuracy. Existing short-run person cache remains 75 scene-scoped IDs / 11 spatial hints; no person tracker logic changed.
- See `docs/review_workflow.md`, `docs/detector_options.md` and `reports/v04_validation.md`. No new model, dependency or license was selected. The owner explicitly requested committing and pushing this checkpoint after reviewing the implementation report. No labeled candidate-miss evidence exists to trigger alternate-detector comparison.
- Browser visual QA timed out at app access; HTTP/storage/synthetic tests do not prove pointer interaction or video playback works. User should review these before relying on the UI.
