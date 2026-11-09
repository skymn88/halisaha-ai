# v0.5.2 — motion/size continuity and conservative shot evidence

## Changes

`MotionConsistentBall` associates multiple observation hypotheses with size-ratio, velocity-residual and direction-consistency gates. At least three real observations are required. Selection needs nontrivial displacement and at least 0.7 source px/frame in the latest step; stationary/jittering hypotheses are not emitted. Duplicate spatial evidence is merged. Competing trajectories remain unknown, and an active-track gap cannot immediately jump to an unrelated track. No predicted positions become observations.

Candidate size is also bounded by nearby observed player heights, with an absolute image-size cap. This is perspective-sensitive pixel logic, not a calibrated physical ball diameter. Person-box exclusion and static-marking evidence remain mandatory. Default video now shows only the selected supported trajectory, not every unconfirmed yellow proposal.

Shot-exit evidence requires a recent nearby-player departure, a consistent observed path, fast image motion and finite end-line crossing. Speed alone, a rebound near the goal, or a region proximity cannot trigger a shot. Control/goal/shot outputs remain unverified candidates. Kinematic speed thresholds are not m/s.

The partial near-goal quadrilaterals previously inferred from incomplete posts are no longer drawn or used for goal-region entry. They remain explicitly unresolved in geometry metadata. Complete visible pilot goal regions still use AI-reviewed approximate geometry; end lines are approximate. This does not solve automatic goal calibration or field boundaries.

## Real-video inspection and experiment

Scene 003 frames 70–134 were inspected, including cropped flight frames 96–118. Earlier selected observations at 113–125 follow the later ball near the goal, not sufficient evidence of the original shot release/flight. The new logic does not label that later motion as a confirmed shot. The actual shot remains missed when flight observations are absent or interrupted by the requested player-box exclusion.

An optional low-contrast motion-blur proposal generator was tested on scene 003. It added 15,694 proposals and produced unstable selections around people/netting without establishing recovery of the shot. It remains OFF in the delivered output. Tests for this experimental generator validate only synthetic proposal behavior.

## Comparison and validation

Fresh output: results/full_analysis_v052_release. Baseline results/full_analysis is preserved. All 9 clips / 4,388 frames reused cached detections; no YOLO rerun or model training.

Baseline: 2,180 selected frames, including 1,013 consecutive selection steps below 0.7 px. New: 977 selected frames, all satisfying the selected-motion gate and outside every exported person box. These are operational invariants, not precision/recall or verified false-positive counts. Recall can decrease under stricter exclusion. No human labels were created.

The delivered run has one end-line crossing candidate and no shot candidates. This is NOT a claim that the montage contains no shots; it reflects insufficient combined departure/flight evidence. No goal or pass accuracy improvement is claimed.

81 tests passed, including static high-score points, jitter, fast linear motion, size jumps, missing frames, stopping, competing tracks, duplicate evidence and shot-exit prerequisites. Native scene navigation and schematic heatmaps remain compatible with JavaScript-disabled previews. Browser results are recorded in the completion message.


## Transition/reference mismatch correction

Visual QA exposed blend-transition tails: a clip's first frames can still show the previous camera while its median background belongs to the next camera. Reference mismatch is now measured on 80×45 thumbnails. Frames above a configurable mean-channel difference of 8 are marked unknown, no pitch/goal overlays are drawn, motion/control state resets and these frames are excluded from schematic density. This heuristic may also reject large lighting changes; it is not a ground-truth cut detector. The final run marks 98 frames unevaluable and retains all 4,388 frames in the video and duration accounting.

Final selected observations were checked against person boxes, minimum movement and excluded-frame lists. Every control-state duration sums to its clip duration. No selections or event chains bridge reference mismatches.

Browser verification after the final transition fix: all nine scene and heatmap links passed with JavaScript disabled and enabled; all videos played, available event links sought correctly, and no JavaScript errors occurred. Mobile and desktop screenshots were generated. The report continues to expose approximate geometry and per-scene rejection/comparison details.
