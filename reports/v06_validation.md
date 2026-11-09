# v0.6: explicit review evidence, not an accuracy milestone

Branch: `feature/tracking-v02`. Baseline: `92e0a4b` / `results/full_analysis_v052_release`.
Final artifacts: `results/review_v06_release`, `results/halisaha-analiz-v060.html`.

## What changed

- A separate cached multi-hypothesis review path preserves observed contact frames. Being inside a person box is a penalty, not immediate removal. HSV skin fraction, white-component roundness and nearby-person size are diagnostic appearance cues; weights affect duplicate preference, not calibrated probabilities. Tiny blurred patches make these weak cues. Relative motion can reject body-attached tracks, including candidates outside a box. No pose model, new dependency, model training or detector pass.
- Trajectory association retains multiple alternatives and bounded gaps. Every exported coordinate came from a source-frame candidate; none is a predicted point. Reference mismatch resets trajectory epochs; no cross-cut event chain. Missing intervals and incomplete flight remain explicit. Multiple hypotheses are **not** multiple balls or a unique selected trajectory.
- Event rules normalize movement to person height, establish foot-control hysteresis, require release and classify team-unknown transfers as control changes. A/B are stronger/weaker rule evidence, neither confirmed truth. The function supports supplied team mappings; no automatic team assignment or reviewed team-input UI was added.
- `reviewed_geometry.py` accepts only named, reviewed human geometry with matching scene, SHA-256 and source image size. Nonfinite, degenerate and crossed goal mouths are rejected. `geometry_sketch.js` exports this format after explicit confirmation, supports dragging corners, and resets approval after edits. Existing draft schema was not loosened into human truth. Automatic suggestions remain drafts.
- Default v0.6 geometry is unavailable until supplied. Existing AI pilot geometry is used only with explicit `--allow-ai-geometry` or `--reuse-geometry`; its provenance is carried into events. Partial pilot goals remain excluded. This experiment uses AI geometry and does not establish generalization.
- Event-only replay consumes new human geometry without rerunning detection or observation extraction. Geometry changes replace stale overlaid video with the raw clip. Heatmaps reuse unchanged v0.5.2 person observations and remain approximate visible-region density, not calibrated field coordinates.

## Frozen configuration and correction

`configs/review_v06.json` was written before processing or reading event matches. Numeric thresholds were not tuned to obtain more hits. Revision `0.6.0-frozen-1` initially called a release plus gapped recontrol near an approximate goal a weak shot. Visual inspection found body/footwear matches and showed that proximity in body heights cannot establish the attacking half. Revision `0.6.0-frozen-2` changes that event's **type**, without changing numeric thresholds: `release_recontrol_review`, excluded from shot/goal/line coverage. Actual shot candidates still need goal-directed geometric evidence. Earlier 5/9 temporal coverage was therefore rejected as a misleading headline; final result is 3/9. A synthetic test also caught an implementation bug that omitted observed intermediate release frames and incorrectly marked flight as missing; the buffer now includes them.

Original observation runs are retained. Final event-only replay records its parent and config SHA. This is development inspection, not a held-out experiment.

## Nine-clip AI timing proxy

All 4,388 frames / 146.413 seconds use existing person and white-motion/YOLO proposal caches. Raw chronological sheets at one-second intervals and denser event intervals were visually inspected. `configs/ai_event_windows_v06.json` records AI-only timing ranges, clip hashes, reviewer and uncertainty. Cropped goals, small balls and goalkeeper reactions prevent frame-exact goal-line assertions. The user's statement that each clip contains a goal is not treated as a human timestamp label.

Matching: shot / goal-region / end-line proposal within **±2 seconds of the fixed AI anchor**. Unknown-goal fast releases and recontrol-only proposals do not qualify. Temporal overlap alone cannot verify the object or event; even a rebound can match. The denominator includes all nine clips, including failures.

| Clip (zero-based) | AI anchor s | v0.5.2 hit | v0.6 hit | All candidates | Candidates/min | Unmatched review load/min |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 13.41 | 0 | 0 | 17 | 70.0 | 70.0 |
| 001 | 3.24 | 0 | 0 | 6 | 36.1 | 36.1 |
| 002 | 9.91 | 0 | 0 | 3 | 15.8 | 15.8 |
| 003 | 3.67 | 1 | 1 | 1 | 5.8 | 0.0 |
| 004 | 2.64 | 0 | 0 | 0 | 0.0 | 0.0 |
| 005 | 8.58 | 0 | 1 | 12 | 34.8 | 31.9 |
| 006 | 13.18 | 0 | 0 | 24 | 65.3 | 65.3 |
| 007 | 17.25 | 0 | 1 | 24 | 63.7 | 55.8 |
| 008 | 15.42 | 0 | 0 | 1 | 2.4 | 2.4 |

Final totals: **88** proposals: 35 fast releases without goal context, 19 release/recontrol review items, 17 team-unknown control changes, 15 line crossings, 2 goal-region entries. **0 shot candidates**, 0 confirmed events; 87 tier B and 1 tier A (a control change, not a shot). Baseline had one line-crossing proposal, so baseline coverage is **1/9**, not the suspected 0/9.

All-candidate burden is about 36.1/min vs baseline 0.4/min. True false candidates/min, human precision and human recall are **null**, not zero. Unmatched candidates may include real events outside the selected time window. The increased burden and remaining zero-shot result mean this is not ready for automatic user statistics.

## Selected visual comparisons, explicitly AI-only

`compare_review_probes.py` exports five point probes and source-image illustrations in `ai_probes.json` and the HTML. The fixed 5-source-pixel tolerance is a pilot inspection tolerance, not calibrated accuracy.

- scene_000 frame 375: visible ball near (205,207); baseline has no selected ball, new hypotheses include a point about 2 px away.
- scene_000 frame 390: foot-contact ball near (180,174); baseline has no selection, new hypotheses include a point about 2.3 px away.
- scene_003 frame 113: visible ball after the shot near (600,247); both paths have a nearby observation. This does not recover the earlier missed flight.
- scene_001 frame 97 and scene_002 frame 329: the new set still includes player-associated / footwear hypotheses. Their temporally matching recontrol records are no longer called shots. Appearance and body correlation have **not** solved this problem.

These manually selected AI examples are not a representative sample, do not establish precision/recall, and do not enter `evaluate_review.py` ground truth. No hand-specific success claim is supported by this inspection.

## Human pilot and checks

- `prepare_pilot_v06.py`: 18 raw single-frame tasks (pre-event and event context), plus nine seeded random two-second event windows. All start unreviewed; AI timing only selects tasks. Existing localhost review app can edit them. All windows are development, not held-out. Existing evaluator measures its declared original detector/selection caches; it must not be presented as v0.6 multi-hypothesis precision.
- `evaluate_review.py` on this pilot: **0 human-reviewed frame rows**, no confirmed events, accuracy unavailable.
- **94 unit tests passed**, including new body-motion, foot/skin penalty, provenance/hash, direction, observed/gapped flight, unknown-team, trajectory separation and recontrol-not-shot regressions.
- Chromium/Playwright, 390 px mobile with JS disabled/enabled: all nine native scene links, all nine visible heatmaps; JS-enabled video playback and event seeks. Geometry workflow verifies missing-name rejection, four-point drawing, explicit confirmation, hash-bound download and approval reset. Synthetic browser downloads are test fixtures, never imported as real human labels.
- Desktop/mobile screenshots inspected. No videos, weights, generated results or human labels are committed.

## Deliberately unresolved

True attacking-half geometry, goal/keeper intervention semantics, pass-intent cone and reviewed team workflow are not implemented; no pass-attempt or completion-rate claim. Actual flight recovery, candidate ambiguity, dribble/body coupling, clipped goals, stationary real balls and reference mismatch remain concerns. Pose/model/license changes are deferred. No physical speed, distance, rating, on-target shot or montage match total is produced. Genuine human ball/geometry labels are the next evidence needed; they cannot be supplied by this assistant as human truth.
