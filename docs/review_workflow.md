# Review schema v1 and evaluation protocol

This layer supports review/evaluation, not a completed event recognition product.
Coordinates are source-image pixels; frames are zero-based source frames; times are frame/FPS.

## Dataset and split

`dataset.json` records source video/hash, cached run/hash, FPS/size, invocation, model
provenance when recorded, scene boundaries and dev/eval scene groups. Historical model
version/weights hashes remain explicitly unknown when absent from old metadata. Future
online runs record implementation version, weights hash and previous/future frame counts.
Offline backgrounds use future frames separately from the single-frame detector.

`evaluation_config.json` is hashed at creation. Evaluation refuses silent configuration changes.
Make an explicit dataset/config version to change a tolerance; record the reason before
assessing held-out labels. The hash is provenance, not protection against intentional editing
of both files. Pilot values: person IoU 0.5; ball center tolerance 5 source pixels; event
tolerances 0.5/1 second. Five pixels is a pilot convention, not a universal requirement.

The prepared set has nine 40-frame detection windows (360 PNGs) and four event-context windows:
the short sample plus seeded random six-second windows in montage scenes 2, 6 and 8.
All 13 clips are exported/decoded through `split_scenes.split_video`: 1,350 clip frames.
PNGs come directly from raw source decoding, never annotated video; MP4V clips are re-encoded
without sound. Known-error and random windows stay separately identified.
Short scene 0 / montage scene 2 are dev; montage scenes 6/8 are eval.
Scene-disjoint is NOT an independent-match split: montage repeats may cross groups.
Camera families and repeated footage need human verification before generalization claims.

## Separate CSV tables

All tables record `review_status` (unreviewed/reviewed), `annotation_source`
(proposal/ai_reviewed/human), reviewer and notes. Only reviewed human records are truth.
AI observations are excluded; proposals are never promoted automatically.

| Table | Meaning |
|---|---|
| `frame_labels.csv` | window/source/scene/frame, ball_state, people_complete, identities_complete |
| `detection_labels.csv` | annotation_id, player/ball, accepted/rejected, bbox/center, proposal_id, error_groups, fp_group |
| `identity_labels.csv` | annotation_id link, identity_id, team_id, role, original track_hint |
| `event_labels.csv` | event_id, anchor frame, type, actors/team, decision, candidate_id, duplicate_of |
| `window_reviews.csv` | events_complete assertion for an entire event window |
| `review_history.jsonl` | append-only UTC revisions with previous/new tables and review provenance |

Table files are atomically replaced; a row-delta journal is written first. A crash between table
replacements can require recovery from the journal. This is a single-user local tool,
not a concurrent annotation database. Back up reviewed labels.

Ball states: visible, partially_occluded, fully_occluded, out_of_frame, uncertain, unreviewed.
Visible/partial needs exactly one accepted ball. Fully hidden/outside has no invented observed
coordinates. Uncertain/unreviewed is not negative. If the tiny ball is unresolved, do not guess.

Semicolon-separated error tags: small_far, fast_blurred, on_line, person_overlap, stationary.
These are reviewed conditions, not physical speed. Rejected FP groups: white_marking,
clothing_footwear, other, unknown. Retain rejected proposals; add/correct a missing ball separately.

Person completeness means **all visible people in the full image**, including spectators and
referees. Roles are separate; there is no implicit player-only ROI or fixed roster cap.
Do not label selected players and treat other true persons as FPs. Identity completeness
additionally requires a distinct identity per person and coherent chains across the window.
Track hints are not identities. Team IDs are arbitrary local labels; empty means unknown.

## Human workflow

Run `python review_app.py --dataset results/review_v04`; open localhost:8765.
Move/resize proposals, add centers/rectangles and reject false boxes. Center is independently
editable. Choose visibility, reviewer/source and completeness; save explicitly.
Tracker occluded is never automatically imported as ground-truth visibility.

Review raw clips or PNG sequences; MP4V browser support varies and preview is not guaranteed
real time. Event anchors always use source frame/FPS. Add non-proposed events as well.
Approve/reject/correct/leave uncertain; mark replays with duplicate_of.
Candidates are retained independently of corrections. Mark events_complete only after
watching the whole window, including moments without proposals. Event approval does not
produce ball boxes, player identity truth or team classification.

The server binds 127.0.0.1, rejects cross-origin JSON writes, and only serves manifest-selected
frames/clips. Do not expose it publicly. No authentication, label propagation or multi-user DB.

## Measurements

`evaluate_review.py` reports per window source/angle/scene, selection, sample counts, frozen
configuration/hash, model provenance and limits. No human labels gives null, not zero accuracy.
Reviewed but unprocessed sampled frames are explicitly counted and excluded from detection
scoring. Historical short/long runs have different stride/image size: no controlled model claim.

Raw person and tracked observations are scored separately on complete person frames.
Matching is maximum-cardinality one-to-one above fixed IoU. P? can be a real detection;
predictions are excluded. Raw person metrics stay null if historical raw observations are absent.

Raw ball coverage asks whether any proposal center is within tolerance of the visible human
ball. Selection precision/recall is separate. Visible-without-candidate and selection-miss-despite-
candidate distinguish the stages. A wrong selected ball is FP+FN; no selection is FN.
Reviewed hidden/out-of-frame observations are unsupported FPs, not measured hidden coordinates.
Uncertain frames are ignored. Matched-only and all-selected-visible center errors are separate.
Condition groups and reviewed FP causes are reported; unknown causes stay unknown.

Events are scored only in fully human-reviewed event windows, using confirmed non-duplicate
labels. One-to-one maximum-cardinality matching uses 0.5/1 s; exact-type and type-agnostic
proposal coverage are separate. A shot_candidate prediction maps to shot for matching only;
a human label left as shot_candidate is not a confirmed shot count. Other types are exact:
a generic control change is not converted into a pass. Unknown/cut-limited intervals are
not ordinary negatives. Nearest-first augmenting paths maximize match count, not globally
minimal summed timestamp error. Actor accuracy stays null without a verified tracker-to-human
identity mapping. Human-uncertain anchors mask a ±maximum-tolerance interval unless the reviewer
supplies an explicit [start,end) interval; these are not treated as negatives.
Known-error and random-audit windows must remain separately reported.

## Reference identity metrics

`export_trackeval.py` only exports windows with human-complete boxes/identities for every frame.
MOTChallenge-style GT/tracker data, seqinfo and sequence map use 1-based local frames.
It computes no homemade IDF1/HOTA. No reference package was installed/run this iteration.

With a separately approved/available TrackEval checkout:

~~~powershell
python export_trackeval.py --dataset results/review_v04 --output results/new_mot --split eval
python PATH_TO_TRACKEVAL/scripts/run_mot_challenge.py --GT_FOLDER results/new_mot/gt --TRACKERS_FOLDER results/new_mot/trackers --SEQMAP_FILE results/new_mot/seqmap.txt --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL cached --DO_PREPROC False --METRICS HOTA Identity CLEAR
~~~

Check exported coverage first. Sampled runs have sparse predictions on the full source
timeline; that measures sampled pipeline coverage, not full-rate identity continuity.
Do not directly compare different cadences. [Reference TrackEval](https://github.com/JonathonLuiten/TrackEval).

## Ball evidence contract

`ball_evidence.csv/.meta.json` v1 contains full source boxes/scores; marking/foreground
fractions, static flag, body/person overlap, outside-pitch flag; scene timeline including
empty frames; thresholds/version and reference hashes where reconstructed from images.
Body height fraction 0.8, marking threshold 0.25 and HSV/difference/core definitions reproduce
the current heuristic. They are not validated optimal parameters.

CSV-only import needs raw people/ball observations, saved candidate invariants and an explicit
historical feature version. Scores recover precision from raw CSV; old fractions stay rounded.
Missing context fails rather than assuming no people. Innovations, IDs, hypotheses, velocities
and selections are recomputed. Reference comparison checks every old candidate field and state,
not only counts. v0.3 mean fractions cannot become v0.3.1 max-core fractions by renaming.
Changing geometry/body definition or image-feature thresholds requires rebuilding evidence.
Tracker hits/gaps/radius can be replayed without images. No algorithm tuning preceded parity.

## Possession and event meanings

All ownership rows are proposals: player, free_ball, contested, unknown. Distance/person-height
is a perspective-adaptive hint, not meters. Competing nearby people means contested;
weak/unassigned nearby people means unknown. Missing ball exports unknown, without a held
owner. Entry/release use separate positive source-time durations. Large gaps reset pending
control; one-frame proximity never establishes possession.

Completed pass: intentional A-to-different-B transfer on the same team with B's control.
Attempt/outcome and deflections need separate review. Control change needs established
opposing control; absent team/contact evidence, output stays a generic change hypothesis.
Recovery is not automatically tackle. Shot means goal attempt, not fast movement.
Rapid image motion only emits shot_candidate, goal context unknown, human review required.
Slow shots can be missed and hard passes can trigger candidates: this is a salience baseline.

Cuts reset all state; unavailable context is in unevaluable_intervals.csv, never joined across
cuts. Unvalidated thresholds and absent team/goal context are explicit in event_generation.json.
Review random windows to find missed events; proposed-only review cannot measure recall.
[Reference football definitions](https://www.statsperform.com/opta-event-definitions/).

Display confirmed events with reviewed/uncertain/unreviewed intervals. No full-match promise
from selected clips. Physical distance/speed needs independent measured calibration, not fit
reprojection alone. Ground-plane homography cannot recover airborne ball 3D motion.
