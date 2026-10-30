# Detector comparison gate and licensing options

No detector was changed/downloaded, dependency installed, model trained or project LICENSE
selected. This documents choices, not legal clearance. Official sources were checked
2026-10-30; pin actual implementation/weights before a future decision by the owner.

| Option | Code / weights distinction | What remains |
|---|---|---|
| Larger YOLO11 | Ultralytics lists AGPL-3.0 / Enterprise; size does not change those options. | Current Ultralytics tracker dependency |
| Original RT-DETR | Official code repository Apache-2.0; verify exact checkpoint separately. Ultralytics wrapper is another implementation route. | Current player tracker still imports Ultralytics |
| RF-DETR | Package and current Nano/Small/Medium/Large detection weights Apache-2.0; Atto/Femto/Pico/XL/2XL detection/Plus components PML. | New adapter needed; tracker unchanged unless separately replaced |
| Original ByteTrack / BoT-SORT | Original repositories MIT; current Ultralytics implementations carry AGPL notices. Optional appearance weights/dependencies have separate terms. | Backend replacement requires integration/regression, not an import rename |
| Future multi-frame ball method | Identify exact code/checkpoint/data terms separately; no family-wide assumption. | Independent ball path, no prerequisite to player statistics |

Primary sources:

- [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics tracker](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/bot_sort.py)
- [RT-DETR LICENSE](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
- [RF-DETR component/weight table](https://github.com/roboflow/rf-detr#license)
- [ByteTrack LICENSE](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE)
- [BoT-SORT LICENSE](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE)

An adapter does not relicense code, weights or dependencies. This work neither determines
AGPL obligations for a future distributed product nor assigns an application license.

## Measurement gate

Prioritize alternate detectors only after reviewed labels quantify visible_without_raw_candidate.
If correct proposals exist but selection fails, use replay/filter measurement first.
No human truth exists now, so the alternate-detector gate is not met.

Freeze identical source-resolution clips/labels, scene split and tolerance. Separate person
detection, ball coverage, selection and tracking. Tune operating points on dev and freeze for
eval. Compare PR curves and equal false-alarm budgets: identical numeric confidence is not
necessarily comparable. Also compare quality/runtime tradeoffs on identical hardware, recording
input shapes, preprocessing, precision, threads, crop passes, memory and decoding inclusion.
Published COCO AP/FPS is not a local football benchmark.

Record implementation version/commit, weights SHA256, class mapping, resolution/resize,
previous/future frames and postprocessing. Future online metadata records artifact/version
and zero detector temporal context; future-frame offline references are separate.
Historical missing hashes stay unknown; today's local weights are not retroactive proof.

Report angle/scene and ball size, blur/fast image movement, occlusion, marking, stationary and
person-overlap groups; FP groups include markings, clothing, other, unknown. Retain sample
counts. Temporal methods reset at cuts and disclose look-ahead. No training plan is part of
this milestone.
