# v0.4.1 review usability work (2026-11-08)

Base: c58c58d, feature/tracking-v02. Prepared for the user-authorized feature-branch checkpoint.

Proposals start pending in the browser. Pending proposals are omitted from label writes and restored as suggestions on reload. The persisted v1 schema is unchanged. Completeness cannot be asserted while player suggestions are pending. Selecting a numbered ball accepts exactly that candidate and rejects the other ball proposals; review status remains a separate explicit checkbox. No-visible-ball defaults to uncertain, not a negative visibility label.

Added smallest-area hit testing, zoom, keyboard navigation/Delete/S, reviewer name persistence, visible save feedback, mobile sizing and collapsed event controls. Previous-frame copying and a full event tab are not implemented.

Validation: existing 55 unittest tests passed after installing runtime dependencies in a local venv. Added a pending-proposal persistence regression. Mock-API Chromium test at 390x844 checks one accepted ball and no automatic people-completeness assertion. Reviewer entry plus three clicks saves a ball review; measured automation times 0.308–0.610 seconds, not human annotation timing or real-video accuracy. Test source: tests/review_ui_browser.cjs. Browser tooling is development-only; runtime requirements unchanged.

Uploaded montage: 640x360, 29.970 FPS, 4388 frames exported into 9 clips with all decoded counts checked. Boundaries remain heuristic. Local H.264/yuv420p/faststart viewing copies and index.html are under results/scene_clips; ffmpeg was already installed. No media is staged or tracked. Downloadable local bundle: results/kesit-inceleme.zip.

No human annotations were created, no accuracy metrics measured, no player rating or match totals inferred. Short-video tracking attempted but official model download returned HTTP 503; existing model weights/caches were absent from this checkout. This blocks real detection-window replay, not scene export or UI testing. Public/phone hosting is not configured; the local HTML bundle is not a public URL.

Follow-up: official Ultralytics Hugging Face YOLO11 mirror download succeeded. Short video processed successfully into results/short_v041 (450 source frames); three 40-frame review windows prepared in results/review_short_v041. All 56 tests passed including the new regression. All nine H.264 clips actually played in Chromium at 390x844. Short inference is an operational run, not ground-truth validation. No long-montage inference was performed.

Standalone viewing artifact: results/kesit-inceleme.html embeds all nine H.264 clips; no ZIP extraction or relative media files required. It is local, not a public hosting URL.
