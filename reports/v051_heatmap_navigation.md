# v0.5.1 — script-independent navigation and schematic density

## Navigation finding

The user's preview environment is not directly inspectable. Reproduced the reported symptoms by opening the previous artifact with JavaScript disabled: only one scene was visible, zero heatmaps were visible. Previous browser tests only covered JavaScript-enabled Chromium.

New report uses native fragment links for all nine scene sections and heatmap headings. All scene sections and heatmap images are present and visible without scripting. Optional event-time seeking progressively enhances an anchor to the video; when scripts are blocked it still scrolls to the video. Native details exposes source geometry. No external asset or JavaScript download required. Versioned output: results/halisaha-analiz-v051.html.

## Dimensions and mapping

Research accessed 2026-11-08:
- The FA, Small-sided Law 1: recommends 25–50 m length and 16–35 m width for 5/6-a-side; 50–60 by 30–40 m for 7-a-side. https://www.thefa.com/football-rules-governance/lawsandrules/laws/football-5-5/law-1---the-playing-area
- Nurteks, manufacturer guidance: gives multiple facility sizes including 25×45 m, 28×48 m and 30×50 m; dimensions vary with site. https://nurteks.com.tr/haberler/blog/hali-saha-yapimi-ve-hali-saha-olculeri-hakkinda-her-sey

45×25 m is an illustrative drawing template, not a measurement of any uploaded venue or a claim of a mandatory standard. Goal/penalty-area artwork is schematic. No metric trajectory, speed or distance is inferred.

`schematic_heatmap.py` maps observed person foot points from the largest inscribed quadrilateral of the visible grass hull onto normalized coordinates. This is NOT independently verified full-field homography: cropped or missed field corners, lens distortion and viewpoint can distort locations/orientation. The source quadrilateral is exported and visible in the report. Outside-transform observations are discarded and their counts are explicitly shown. In some clips a substantial fraction is excluded; displayed density does not describe the entire match or unseen pitch.

Gaussian-smoothed observation-time bins are normalized to probability mass. Each clip uses its own peak color scale; colors compare locations within a clip, not total activity across clips. Heatmaps are rendered on an artificial pitch with no underlying camera image. No new YOLO run or event-statistics changes.

## Verification

69 unittest tests passed (4 new tests: ordered corner mapping, degenerate geometry, empty density and hotspot rendering). Mobile Chromium at 390×844 was tested with JavaScript disabled and enabled: 9 scene links, 9 heatmap links, all images visible and loaded. With scripting enabled, 9 videos played and event timestamps were sought successfully. Desktop screenshot inspected at 1440×1000. User-side HTML preview policy may differ, but all information remains accessible by scrolling even if scripts do not execute.
