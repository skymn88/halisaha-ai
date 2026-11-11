# v0.6.1 — shot-logic fixes and held-out pitch calibration (handoff note)

Author of this change set: Claude (Anthropic), at the owner's explicit request to fix the logic and push to
`feature/tracking-v02` on top of `f3f5e62` (v0.6). Written so that Codex or any other agent can continue
without re-deriving the reasoning. **This is a logic/contract fix, not an accuracy milestone**: no videos,
caches, detector or human labels were available in this session, so nothing was re-run on real footage.

## Why: two reasons the v0.6 event layer could not emit shots

Found by reading `tiered_events.py`, then confirmed with synthetic scenarios (see tests below).

1. **Direction test used the wrong target.** `directed_to_mouth` intersected the ball's forward ray with
   `goal['mouth_line']`. In the reviewed-geometry schema (`reviewed_geometry.py`) `mouth_line` joins the two
   post *feet* (`[polygon[0], polygon[3]]`), i.e. the base line on the ground. A ball flying through the air
   into the mouth appears *above* that line in the image and never crosses it, so `direction` was False for
   every airborne shot and `shot_candidate` could only appear for balls that rolled across the base line.
   The pre-existing synthetic tests used a vertical `mouth_line`, which is why this was not caught.
2. **Possession gap tolerance was effectively 0.11 s.** `Possession` (event_candidates.py) resets control when
   consecutive ball observations are further apart than `max_ball_gap_s` (default 0.11 s). `tiered_events`
   did not override it, while selected-ball observations cover only ~22 % of frames (typical spacing of
   3-5 frames). Control (`enter_s` 0.15 s) was therefore rarely established, so `release` and every
   shot/pass/recovery event that depends on it was rarely reachable.

A third, separate reason is **not** fixed here: the flight frames of real shots are often never observed by
the upstream ball proposals (motion blur, size/aspect gates in `motion_consistency.py`). See "Not done".

## What changed

| File | Change |
| --- | --- |
| `tiered_events.py` | New `directed_to_goal(a, b, goal, margin)`: forward ray must cross an edge of the goal-mouth quadrilateral (edges extended by `margin` px; origin already inside the mouth returns False). `generate_events` now uses it instead of `directed_to_mouth` (kept unchanged for compatibility/tests). `Possession` gets `max_ball_gap_s = config['association_gap_s']` (reuses an existing threshold; no new tuned number). New review-only event type `shot_direction_review` (see below). |
| `reviewed_geometry.py` | Optional `pitch_calibration` block with a **held-out check**: `validate_pitch_calibration`, `project_to_pitch`; `validate_geometry` rejects malformed blocks; `resolve_geometry` attaches `pitch_calibration_result` for human geometry only. Reuses `calibration.SceneGeometry` for the homography fit. |
| `configs/review_v06.json` | `version` renamed `0.6.0-frozen-2` -> `0.6.1-logic-fix-1`. **No numeric threshold changed.** `load_config` still requires the same key set. |
| `review_report_v06.py` | Label for `shot_direction_review`; hard-coded "0.6.0-frozen-2" text replaced by a pointer to the config. |
| `test_shot_logic_v061.py` (new) | 9 tests, realistic geometry (base-line `mouth_line`). 5 of them fail on the old logic. |
| `test_pitch_calibration_v061.py` (new) | 8 tests: synthetic camera, held-out failure, malformed input, hash/provenance. |
| `docs/v061_changes.md` (new) | This note. `README.md` and `AGENTS.md` carry a short pointer. |

No CSV columns, schemas (review schema 1, geometry schema 2 base fields), detector, tracker or caches changed.
`evaluate_clip_proxy.TARGET_TYPES` is unchanged, so `shot_direction_review` does **not** count as a clip-level
hit; it only adds to the unmatched review burden. Do not add it there without a protocol-version note.

### New event type `shot_direction_review` (tier B, review-only)

Emitted when, after a release, the observed flight is fast (`fast_heights_per_s`), far enough
(`travel_heights`), straight enough (`direction_consistency`), starts near the goal
(`goal_half_radius_heights`) and points into the goal-mouth polygon, but **no goal-region entry or end-line
crossing was ever observed** (ball lost to blur, net, keeper, or recontrolled by another player). It is
reported once per release, when the release expires, the trajectory ends, or another player takes control.
It is *not* a shot decision and never a goal/on-target claim. Reasons include
`no_goal_crossing_observed` and `human_review_required`.

`shot_candidate` semantics are unchanged: A needs fully observed flight plus an observed entry/crossing;
B allows gaps near the goal. With the direction fix both can now be reached by airborne shots.

### Held-out pitch calibration contract (optional, human only)

The old reprojection error is measured on the fitting points, so it cannot validate a calibration. v0.6.1 adds
an optional block inside a schema-2 human geometry record (hash-bound like the rest of the record):

```json
"pitch_calibration": {
  "pitch_size_m": [WIDTH_M, LENGTH_M],
  "fit_points":     [{"image": [x, y], "pitch_m": [X, Y]}],
  "holdout_points": [{"image": [x, y], "pitch_m": [X, Y]}],
  "max_holdout_error_m": ACCEPTANCE_LIMIT_M
}
```

`fit_points` >= 4 (not collinear), `holdout_points` >= 2 and different from the fit points, pitch coordinates
inside `[0,width] x [0,length]`, `max_holdout_error_m` declared by the reviewer. All values are measured/declared by
a named human; nothing here assumes pitch dimensions. Result: `status` `validated` (`metric_enabled` true) or
`failed_holdout` (`metric_enabled` false, goals/end lines still usable); malformed input raises `ValueError`.
`project_to_pitch(result, point)` returns metres only when validated, and is valid only for ground-plane points
(not an airborne ball, no lens correction). **Nothing consumes this yet**: heatmaps/speeds still use image
space. There is also no UI for entering it (`geometry_sketch.js` was not touched); the JSON can be authored by
hand or by extending the sketch tool.

## Verification (this session)

- `python -m unittest test_shot_logic_v061 test_pitch_calibration_v061 test_review_v06` — 30 tests pass.
- Same shot tests against the pre-fix `tiered_events.py` (shimmed import): 5 failures
  (`airborne into mouth`, `fully observed airborne shot`, `sparse observation`, `review-only`,
  `possession gap follows association gap`).
- Full discovery: 111 tests, 103 pass; the 8 errors are all `ModuleNotFoundError` for `torch`/`ultralytics`
  (not installed in this sandbox), in `test_detection_tracking` and `test_review_workflow`.
  Run the complete suite in the real environment before trusting this.
- Not run: any video, cache, report generation or browser test. `review_report_v06.py` was only import-checked.

## Not done / next steps (in priority order)

1. **Re-run `replay_review_events.py` on the real caches** (no detector needed) and compare per-clip proxy
   hits and candidates/min against `results/review_v06_release`. The effect on the 9-clip proxy is unknown;
   the ±2 s AI anchors are not human labels.
2. **Cross-trajectory release chain**: `generate_events` keeps one `Possession` per trajectory, so control on
   trajectory A and flight on trajectory B are never linked. Trajectories may overlap in time (alternative
   hypotheses), so linking needs a deliberate design, not a quick patch.
3. **Upstream flight recovery**: size/aspect/step gates in `motion_consistency.py` (area <= 200 px^2,
   aspect <= 4, step <= 65 px/frame, size ratio <= 2.2) can reject fast blurred balls; unverified without data.
4. **Pitch/goal geometry**: automatic detection (`auto_pitch.py`) is still unreliable (convex grass hull,
   white-only line colour, paired-post goal rule, cropped near goals excluded). Next practical step is a UI for
   entering goal mouths + `pitch_calibration` per clip, then using validated metres.
5. Real human labels are still zero; keep AI and human results separate.
