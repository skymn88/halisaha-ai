# v0.4.2 per-clip operational measurements

User requested actual measurements for every uploaded montage clip. This explicitly scoped run processes all nine exported clips once at stride 1, imgsz 640, YOLO11n, default pitch-ball and player tracking settings. Future tuning should reuse these raw caches; do not routinely reinfer the montage.

`measured_report.py` exports a standalone HTML with embedded H.264 annotated videos, image-coordinate person observation density and a JSON sidecar. It checks decoded video and ball-state counts against each clip manifest. It reports raw ball proposals separately from selected observations. Control-duration values are the existing unvalidated Possession heuristic, with unknown intervals retained and state reset at scene boundaries. They are not team possession or human-confirmed control.

Person density includes every exported person observation, including unassigned boxes. Duplicate observations can bias it. Heat scales normalize within each clip and cannot compare intensity across cameras. Temporary IDs and possible transitions are diagnostics, not player counts or measured ID switches. Source clip frames map to the montage via the manifest.

No human labels, calibrated distances, verified event counts or player ratings were produced. Low ball coverage and fragmented person IDs currently block reliable player statistics. Event statistics remain null, never zero. No model, filtering threshold or identity logic was changed just to produce larger counts.

## Completed run

9 clips, 4,388 decoded/processed frames. Selected ball frames per clip: 39, 0, 39, 22, 0, 7, 146, 0, 0. Total 253 selected frames; not verified ball accuracy. JSON assertions verified all nine records, total frame count and state-duration sums. Four clips have zero selected ball frames. No thresholds were relaxed.
