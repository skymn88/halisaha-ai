# v0.6 cached review workflow

Run from the repository root in the existing environment. Outputs must be new directories.

```bash
# Default: no unreviewed geometry enters event inference.
.venv/bin/python review_pipeline_v06.py --output results/review_new --geometry-dir results/human_geometry

# Explicit AI-only development comparison, never human ground truth.
.venv/bin/python review_pipeline_v06.py --output results/review_ai --allow-ai-geometry

# Human correction: save each HTML download as scene_000.json etc. in this directory.
# Replay events from existing hypotheses, without detector inference or image decoding.
.venv/bin/python replay_review_events.py --input results/review_v06_release --output results/review_human_geometry --geometry-dir results/human_geometry

# Temporal proxy + portable HTML with embedded video, source examples and heatmaps.
.venv/bin/python evaluate_clip_proxy.py results/review_v06_release
.venv/bin/python review_report_v06.py --root results/review_v06_release --output results/halisaha-analiz-v060.html

# Small genuine-human pilot (already prepared in this workspace).
.venv/bin/python prepare_pilot_v06.py --output results/human_pilot_new
.venv/bin/python review_app.py --dataset results/human_pilot_v06
.venv/bin/python evaluate_review.py --dataset results/human_pilot_v06 --split all
```

The portable HTML works without a server for viewing. JavaScript is needed for drawing/downloading human geometry; scene links and heatmaps also work with JavaScript disabled. Downloading geometry does not update the displayed results until a replay and report export. The review server remains localhost-only; no unauthenticated public annotation endpoint was added.

Geometry schema 2 is a **new, separate input contract**, not a relaxation of review schema 1. A record requires `scene`, `video_sha256`, `image_size`, `annotation_source: human`, `review_status: reviewed`, a nonempty `reviewer`, `goals` and/or `endlines`. A goal polygon is ordered left foot, left top, right top, right foot; `mouth_line` joins the two foot points. It must be finite, convex, in bounds and nondegenerate. Aut lines need two distinct source points. Automatic suggestions are drafts until a named human explicitly checks them. Old `manual_draft` exports cannot be imported as reviewed geometry.

Current `evaluate_review.py` remains restricted to actual human labels and original detector/selected-ball cache evaluation. `evaluate_clip_proxy.py` is separate: it reports AI temporal coverage and review workload, never human precision/recall. Multiple hypotheses per frame are not a single selected ball. Team mappings are only supported by the event function; no team labels were invented for these clips. See `reports/v06_validation.md` for failures and scope limits.
