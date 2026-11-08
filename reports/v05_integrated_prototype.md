# v0.5 integrated analysis prototype

## Implemented

- Automatic image-space grass hull, marking segments and goal-structure hypotheses (`auto_pitch.py`). These are uncalibrated proposals. Visual inspection found railing/logo false goals and missed cropped goals. The automatic goal detector is NOT a solved capability.
- For these nine pilot clips only, approximate visible goal regions/end lines were inspected by the assistant and saved as `ai_reviewed` in `configs/pilot_goal_regions.json`. SHA-256 binds each override to its exact re-encoded source clip. Overrides are labeled separately from automatic geometry, and are not human ground truth or independent calibration. Near goals are partly cropped; their regions are approximate.
- Immediate person-box exclusion, as requested, replaces the persistence condition in this integrated experiment. A ball candidate whose center lies within any exported person box is discarded, even at the feet. True contact observations can therefore be lost. Existing person detector boxes/tracking were not retrained or made demonstrably tighter.
- Combines cached white candidates with raw YOLO ball proposals. Existing static-marking evidence is reapplied to prevent stationary logo/line observations from becoming ball selections. Ambiguous top-ranked candidates remain missing. Scores are heuristics, not calibrated ball probabilities.
- Image-space end-line crossing, rapid approach to a goal region, goal-region entry and control-transfer candidate paths. A 2D crossing/entry is not a confirmed goal, and a control transfer without team/intent evidence is not a completed pass. All candidate events retain provenance and source times.
- A standalone user dashboard with per-clip annotated video, automatic field/AI-reviewed goal views, heatmaps, observation duration/pixel movement tables and seekable event timeline. Confirmed event counts, meter distances and ratings remain null.

## Validation and known failures

All nine clips processed at every source frame from existing caches, without YOLO retraining or full-montage reinference. Synthetic tests cover immediate box exclusion (including feet), finite line-segment crossing, empty-image geometry and uncalibrated green-field output. Earlier 61 regressions plus these 4 passed before final static-evidence integration; final suite rerun is recorded at delivery.

Representative event frames were visually inspected. White logo selection caused a bogus transfer in the initial integrated run, motivating restoration of static-marking rejection. Clothing, field markings and track jumps remain possible. More selected observations or event candidates are not an accuracy gain. Nine clips are a montage, not an independent-match benchmark; no human labels were created.

## Incomplete product capabilities

General automatic goal localization remains unreliable; pilot overrides do not generalize to arbitrary uploaded videos. Accurate player identity/team separation, completed-pass measurement, shot/goal confirmation, calibrated running distance/speed and meaningful ratings remain unvalidated or absent. The dashboard is a local self-contained prototype, not a hosted multi-user service. Media/caches remain ignored by Git.

## Final operational checks

65 unittest tests passed after static-evidence integration. All 4,388 source frames accounted for. Every selected ball observation was checked against exported person boxes and none was inside; control-state durations reconcile with each clip duration. Output includes 10 shot candidates, 2 control-transfer candidates, 3 goal-region entry candidates and 3 end-line crossing candidates. These are overlapping unreviewed hypotheses, not verified totals. One clip still has zero selected observations.

Browser verification: all nine embedded videos played in Chromium at 390x844; all nine heatmap tabs opened; available event buttons sought to the expected timestamp; no JavaScript errors. Desktop view at 1440x1000 was captured and visually inspected. The final static-evidence run passed all 65 unit tests. These checks validate delivery and invariants, not shot/goal accuracy.
