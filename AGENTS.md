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

The repository now contains v0.3 development code built on the v0.2 checkpoint `da4c260` (pushed to origin). Continue on `feature/tracking-v02`; use versioned commit messages instead of opening a branch for each version unless requested. Check the actual branch and working tree before starting new work. Existing uncommitted changes must be preserved.

| File | Actual responsibility |
| --- | --- |
| `baseline_tracking.py` | Preserved v0.1 CLI: one YOLO tracking pass for people and sports balls; annotated video and original detection CSV. Keep as a comparison baseline. |
| `run_tracking.py` | v0.3 CLI and orchestration, legacy `--ball-mode v02`, scene resets, player tracking, ball inference and extended exports. |
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
3. Retain persistent YOLO person tracking and separate `sports ball` inference. Existing confidence/tracker settings remain configurable; CPU thread count is configurable (default 1).
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
- Separate player and ball inference increases runtime. Tracking parameter changes are tuning choices, not proven stability improvements.
- Optional calibration mathematics/ROI support exists, but no measured real-camera calibration is supplied or validated. There is no lens-distortion correction, semantic pitch map, metric trajectory analysis, heatmap, event detector, possession model, identity gallery or user identification interface yet.
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
