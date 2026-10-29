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

The repository currently contains v0.2 code. The inspected branch is `feature/tracking-v02`; v0.3 is planned, not implemented. Check the actual branch and working tree before starting new work. Existing uncommitted changes must be preserved.

| File | Actual responsibility |
| --- | --- |
| `baseline_tracking.py` | Preserved v0.1 CLI: one YOLO tracking pass for people and sports balls; annotated video and original detection CSV. Keep as a comparison baseline. |
| `run_tracking.py` | v0.2 CLI, argument validation, model loading, video loop, player tracking, separate ball inference, helper orchestration and exports. |
| `tracking_helpers.py` | `TemporalBall`, `white_marking`, `IDTransitions` and `jersey_feature`; pixel-space logic without pitch calibration. |
| `test_tracking_helpers.py` | Five synthetic unittest cases for helper behavior. These are not an annotated football benchmark. |
| `requirements.txt` | `ultralytics>=8.3,<9`, `opencv-python>=4.8`, `numpy>=1.24`, `pandas>=2.0`. Pandas is declared but not used by the current tracking scripts. |
| `README.md` | Setup, CLI parameters, output formats, limitations and previously recorded validation results. Update when behavior changes. |

Local test assets include `ilk_15_saniye.mp4` and `yolo11n.pt`. Video files, model weights, `results/` and `runs/` are ignored by git and may be absent in another checkout. No annotated training or evaluation dataset was found in the inspected repository.

### Runtime architecture

1. Open a video with OpenCV; optionally sample source frames with `--stride`.
2. Resolve an Ultralytics tracker configuration and save it as `tracker.yaml` in the output directory.
3. Use persistent `YOLO.track` for the `person` class, with BoT-SORT by default or ByteTrack optionally. v0.2 defaults to a 60-processed-frame track buffer and disables BoT-SORT global motion compensation for fixed cameras.
4. Use a separate YOLO instance and inference pass for `sports ball`, with independent confidence (`--ball-conf`, default 0.15; person `--conf` defaults to 0.25).
5. Associate ball candidates to hypotheses with nearest-center matching within a pixel radius, one observation per hypothesis per frame. Confirm after sufficient observations; expire hypotheses after a source-frame gap.
6. Use white pixels inside and around a candidate as a marking caution cue. An unconfirmed candidate with that cue needs displacement to become accepted. Confirmed hypotheses remain accepted when stationary. Export observed boxes only, without gap interpolation or backfilling confirmation frames.
7. Log possible spatial player-ID transitions and extract median OpenCV Lab torso colors. Neither operation assigns persistent player identities or teams.
8. Write annotations, CSV diagnostics and a JSON summary.

Ball confirmation counts observations across allowed gaps, not necessarily consecutive frames. Ball and ID helper gaps use source frames; the underlying player tracker advances once per processed frame. Prefer stride 1 for continuity.

### Output contract

- `tracked.mp4`: processed frames with class-specific IDs and confidence annotations.
- `detections.csv`: `frame,time_s,track_id,class_name,confidence,x1,y1,x2,y2,foot_x,foot_y,object_type,jersey_l,jersey_a,jersey_b`.
- `ball_candidates.csv`: every ball proposal, confidence, box, hypothesis ID, status and white-marking cue.
- `id_changes.csv`: old/new person IDs, source-frame gap, center distance and `possible_spatial_transition` evidence.
- `summary.json`: accepted ball detection count, unique person IDs, possible transitions, processed frames and invocation parameters.
- `tracker.yaml`: resolved player tracker configuration.

Frames are zero-based; timestamps use source frame / source FPS. Coordinates are image pixels. Person foot coordinates are the box-bottom center; ball foot columns are schema-compatible box-bottom coordinates, not physical contact points. Person ID -1 means untracked. Ball and person IDs have separate namespaces: identify a row by object type and ID together.

## Known problems and evidence limits

- **Priority issue:** penalty spots and white pitch markings still produce false ball detections. v0.2 temporal filtering does not solve this reliably.
- Isolated white spots can escape the surrounding-white cue; bounding-box jitter can satisfy the movement threshold. Once a false hypothesis confirms, it is not revalidated against pitch geometry. Multiple ball hypotheses can be accepted simultaneously.
- A real ball first observed stationary on a white marking may be suppressed. Never solve the marking problem by rejecting all stationary objects or by permanently masking every marking location.
- Generic COCO `sports ball` detection is not football-specific. Small, distant or occluded balls can be missed; pixel thresholds vary with perspective and resolution.
- Tracker IDs can fragment or switch. `IDTransitions` only flags a previously unseen ID near a recently disappeared ID; it does not verify identity, detect all swaps or merge tracks. Unique person IDs are not a count of actual players.
- The person class can include referees and spectators. Torso color features can include skin, grass, occlusion and mixed jersey colors. There is no team classifier or jersey-number recognition yet.
- Separate player and ball inference increases runtime. Tracking parameter changes are tuning choices, not proven stability improvements.
- There is no calibration, pitch map, homography, metric trajectory calculation, heatmap, event detector, possession model, identity gallery or user identification interface yet.
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
- Report modified files, configuration, validation results and unresolved limitations. Do not claim accuracy or stability improvements without evidence. Do not commit or push automatically.

## Local commands and validation

Python 3.10–3.12 is recommended in the README; historical validation used Python 3.10 and Ultralytics 8.4.192. Requirements specify ranges, so record the actual environment for comparisons.

```powershell
pip install -r requirements.txt
python run_tracking.py --help
python -m unittest -v test_tracking_helpers
python -m py_compile run_tracking.py tracking_helpers.py baseline_tracking.py
python baseline_tracking.py --video ilk_15_saniye.mp4 --output results/baseline_comparison
python run_tracking.py --video ilk_15_saniye.mp4 --output results/candidate_comparison
git diff --check
```

Install dependencies only when necessary. Use available local weights to avoid an unintended download. Full video runs are appropriate for tracking changes; documentation-only work does not require rerunning inference. For behavioral comparisons, report ball rows, unique nonnegative person IDs, possible ID transitions and processed/decoded frame counts, inspect annotations and candidate diagnostics, and document parameter differences. Update README validation evidence after meaningful tracking changes.

## Development roadmap — planned, not implemented

### v0.3: pitch-aware ball detection and camera calibration

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
