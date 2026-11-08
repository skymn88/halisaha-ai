# v0.4.4 — persistent player overlap and richer report

Uses existing white-context candidates and cached person detections, with no YOLO rerun. A candidate is suppressed after three consecutive source frames inside the top 80% of the same confidently tracked person's box, with relative-position change at most 0.15. Suppression is reversible when the candidate leaves the body region. Feet remain eligible. This is bounding-box context, not pixel segmentation; true airborne balls/body contact can be suppressed and footwear false positives remain.

All nine clips were processed. Audit CSVs preserve suppressed proposals and reasons. Preview videos show person boxes plus retained candidates. Scene 0/frame 220 visually inspected: some body/clothing candidates removed, remaining clothing/footwear errors still evident. No accuracy claim or human labels.

`progress_report.py` combines embedded H.264 videos, filter counts, image-coordinate observation heatmaps and longest-observed track tables. Pixel displacement sums only consecutive observations within a scene-scoped ID. It includes jitter/identity error and is neither running distance nor speed. These tables are not player rankings.

`geometry_sketch.js` adds locally stored, downloadable goal/end-line sketches on source images. Sketches remain unreviewed and are NOT used for event inference. Automatic goal localization, shot detection, confirmed passing statistics and calibrated running measurements remain unfinished. A 2D line overlap cannot prove crossing a 3D goal plane.

Validation: 3 new regressions cover three-frame persistence, release outside torso, missing-frame reset, feet retention and relative motion. Full suite and mobile browser checks are recorded in the completion message. All video/candidate artifacts stay under ignored results/.
