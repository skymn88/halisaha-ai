"""Persist weak/strong overlapping box pairs as review hints, never suppress detections."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from player_tracking import box_iou
from review_schema import flag, human_reviewed, read_rows, write_rows, load_dataset


def audit(path, output, strong_conf=.25, weak_floor=.10, overlap_iou=.5, dataset=None):
    if not 0 <= weak_floor < strong_conf <= 1 or not 0 < overlap_iou <= 1:
        raise ValueError('Invalid confidence/IoU thresholds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, labels = defaultdict(list), {}
    for row in read_rows(path):
        if row['class_name'] in ('person', 'player') and int(row['track_id']) >= 0:
            groups[(int(row.get('scene_id') or 0), int(row['frame']))].append(row)
    if dataset:
        data = load_dataset(dataset)
        matching_sources = [s['source_id'] for s in data['sources']
                            if (Path(s['run'])/'detections.csv').resolve() == Path(path).resolve()]
        if len(matching_sources) != 1:
            raise ValueError('Audit CSV must match exactly one dataset source')
        complete = {}
        for row in read_rows(Path(dataset)/'frame_labels.csv'):
            if row['source_id'] == matching_sources[0] and human_reviewed(row) and flag(row['people_complete']):
                complete.setdefault((int(row['scene_id']), int(row['frame'])), row['window_id'])
        labels = {key:[] for key in complete}
        for row in read_rows(Path(dataset)/'detection_labels.csv'):
            key = (int(row['scene_id']), int(row['frame']))
            if complete.get(key) == row['window_id'] and human_reviewed(row) and row['object_type'] == 'player' and row['decision'] == 'accepted':
                labels.setdefault(key, []).append(row)
    pairs, weak_count = [], 0
    box = lambda row: [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
    for (scene, frame), rows in groups.items():
        weak = [r for r in rows if weak_floor <= float(r['confidence']) < strong_conf]
        strong = [r for r in rows if float(r['confidence']) >= strong_conf]
        weak_count += len(weak)
        for low in weak:
            for high in strong:
                if low['track_id'] == high['track_id']:
                    continue
                overlap = box_iou(box(low), box(high))
                if overlap <= overlap_iou:
                    continue
                interpretation = 'unreviewed_overlap_not_proven_duplicate'
                truth = labels.get((scene, frame))
                if truth is not None:
                    def best(annotation):
                        choices = [(box_iou(box(annotation), box(r)), r['annotation_id']) for r in truth]
                        score, identifier = max(choices, default=(0, ''))
                        return identifier if score >= .5 else ''
                    a, b = best(low), best(high)
                    interpretation = ('both_boxes_match_same_human_box' if a and a == b else
                                      'boxes_match_distinct_human_boxes' if a and b else 'reviewed_but_unmatched')
                pairs.append(dict(scene_id=scene, frame=frame, time_s=low['time_s'],
                                  weak_id=low['track_id'], strong_id=high['track_id'],
                                  weak_confidence=low['confidence'], strong_confidence=high['confidence'],
                                  iou=overlap, **{'weak_'+k: low[k] for k in ('x1', 'y1', 'x2', 'y2')},
                                  **{'strong_'+k: high[k] for k in ('x1', 'y1', 'x2', 'y2')},
                                  interpretation=interpretation))
    fields = ['scene_id', 'frame', 'time_s', 'weak_id', 'strong_id', 'weak_confidence',
              'strong_confidence', 'iou']+['weak_'+k for k in ('x1','y1','x2','y2')]+[
                  'strong_'+k for k in ('x1','y1','x2','y2')]+['interpretation']
    write_rows(output/'overlap_pairs.csv', fields, pairs)
    report = dict(weak_associated_rows=weak_count, overlapping_pairs=len(pairs),
                  weak_rows_with_overlap=len({(r['scene_id'], r['frame'], r['weak_id']) for r in pairs}),
                  overlap_frames=len({(r['scene_id'], r['frame']) for r in pairs}),
                  per_scene=dict(Counter(str(r['scene_id']) for r in pairs)),
                  review_interpretations=dict(Counter(r['interpretation'] for r in pairs)),
                  parameters=dict(strong_conf=strong_conf, weak_floor=weak_floor, overlap_iou=overlap_iou),
                  reviewed_complete_frames_available=len(labels),
                  note='High IoU is a diagnostic, not proof of duplicate players. No boxes are suppressed.')
    (output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', default='results/v031_final/detections.csv')
    parser.add_argument('--output', default='results/v04_player_audit')
    parser.add_argument('--strong-conf', type=float, default=.25)
    parser.add_argument('--weak-floor', type=float, default=.10)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    print(json.dumps(audit(args.detections, args.output, args.strong_conf, args.weak_floor,
                          args.iou, args.dataset), indent=2))
