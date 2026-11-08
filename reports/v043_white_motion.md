# v0.4.3 — white-motion proposal experiment

Implemented an opt-in image-only candidate generator, without changing YOLO, the accepted-ball tracker, model dependencies or confidence thresholds. It combines HSV brightness/saturation, local top-hat contrast, offline median-background difference, connected-component size and bounded temporal association. An optional grass-neighborhood cue is compared separately. Both modes process the same nine scene clips; no new YOLO inference.

Candidate scores are deliberately not represented as detector probabilities. CSVs contain frame, source, component bounds, support and motion evidence. Temporally supported candidates are still proposals, not accepted balls. A moving candidate can remain observable when it stops briefly; a ball stationary throughout median samples may disappear. Camera movement, lighting and white clothing can create many candidates. Temporal matching among dense candidates can switch identity. Green-neighborhood filtering can miss airborne balls, occlusions, worn turf and unusual lighting.

Visual inspection of scene 0 at frame 220 found wall/stand, clothing and footwear proposals in the basic mode. Grass context removes many wall/stand proposals but still retains clothing and footwear. Thus this experiment does NOT solve reliable ball detection and must not be silently promoted into the baseline. Candidate-count reduction is not measured precision or recall.

58 tests passed, including synthetic moving-white-region/stationary-continuation and static-marking rejection checks. Synthetic tests do not validate match accuracy. `motion_report.py` exports the separate standalone comparison under results/top-deneyi.html. All media and caches remain ignored by git.

Shots/goals/passes are not inferred from these noisy proposals. The user's statement that each clip contains a goal is useful clip-level context, not a frame-level goal label. Reliable ball evidence and goal-region/event-time review remain necessary. Human ground truth remains zero; no artificial human labels were written.
