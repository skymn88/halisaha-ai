"""Evaluate only explicitly human-reviewed labels; empty/AI-only sets produce null metrics."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from ball_cache import sha256_file
from player_tracking import box_iou
from review_schema import flag, human_reviewed, load_dataset, read_rows, validate_frame


def match_pairs(left, right, eligible, cost):
    """Maximum-cardinality bipartite matching, deterministic nearest-first augmenting paths."""
    edges = {i: sorted([j for j in range(len(right)) if eligible(a, right[j])],
                       key=lambda j: (cost(a, right[j]), j)) for i, a in enumerate(left)}
    assigned = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(left)):
        visit(i, set())
    return sorted((i, j) for j, i in assigned.items())


def counts(tp=0, fp=0, fn=0):
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def bbox(row):
    return [float(row[k]) for k in ('x1','y1','x2','y2')]


def center(row):
    if row.get('center_x') not in (None, ''):
        return float(row['center_x']), float(row['center_y'])
    box = bbox(row)
    return (box[0]+box[2])/2, (box[1]+box[3])/2


def distance(a, b):
    x, y = center(a), center(b)
    return math.hypot(x[0]-y[0], x[1]-y[1])


def evaluate(directory, split='eval', output=None):
    directory = Path(directory)
    dataset = load_dataset(directory)
    config_path = directory/'evaluation_config.json'
    if sha256_file(config_path) != dataset['evaluation_config_sha256']:
        raise ValueError('Evaluation configuration changed after dataset creation; version the dataset/config explicitly')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    tolerance, person_iou = config['ball_center_tolerance_px'], config['person_iou']
    if not math.isfinite(tolerance) or tolerance <= 0 or not 0 < person_iou <= 1:
        raise ValueError('Invalid fixed evaluation thresholds')
    windows = {w['window_id']: w for w in dataset['windows'] if split == 'all' or w['split'] == split}
    sources = {s['source_id']: s for s in dataset['sources']}
    frames = defaultdict(dict)
    for row in read_rows(directory/'frame_labels.csv'):
        if row['window_id'] in windows:
            frames[row['window_id']][int(row['frame'])] = row
    labels, all_labels, identities = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in read_rows(directory/'detection_labels.csv'):
        all_labels[(row['window_id'], int(row['frame']))].append(row)
        if human_reviewed(row) and row['decision'] == 'accepted':
            labels[(row['window_id'], int(row['frame']))].append(row)
    for row in read_rows(directory/'identity_labels.csv'):
        identities[(row['window_id'], int(row['frame']))].append(row)
    cache = {}
    for source_id, source in sources.items():
        run = Path(source['run'])
        raw = run/'raw_detections.csv'
        people, balls, selected, tracked = defaultdict(list), defaultdict(list), {}, defaultdict(list)
        raw_rows = read_rows(raw)
        for row in raw_rows:
            (people if row['class_name'] in ('person','player') else balls)[int(row['frame'])].append(row)
        if not raw.exists():
            for row in read_rows(run/'ball_candidates.csv'):
                balls[int(row['frame'])].append(row)
        for row in read_rows(run/'detections.csv'):
            if row['class_name'] == 'person':
                tracked[int(row['frame'])].append(row)
        timeline = read_rows(run/'ball_track.csv')
        for row in timeline:
            if row['state'] == 'observed':
                selected[int(row['frame'])] = row
        cache[source_id] = dict(people=people, balls=balls, selected=selected,
                                tracked=tracked, processed={int(r['frame']) for r in timeline},
                                raw_people_available=raw.exists())
    per_window = []
    for window_id, window in windows.items():
        if window['kind'] != 'detections':
            continue
        cached = cache[window['source_id']]
        selected_counts, person_counts, tracked_counts = Counter(), Counter(), Counter()
        positives = raw_hits = ball_frames = human_frames = skipped = selection_misses = 0
        errors, all_errors, groups = [], [], defaultdict(Counter)
        false_groups = Counter()
        complete_identity_frames = 0
        person_frames = 0
        for frame in range(window['start_frame'], window['end_frame']):
            status = frames[window_id].get(frame)
            if not status or not human_reviewed(status):
                continue
            validate_frame(status, all_labels[(window_id, frame)], identities[(window_id, frame)],
                           sources[window['source_id']]['image_size'])
            human_frames += 1
            if frame not in cached['processed']:
                skipped += 1
                continue
            gt = labels[(window_id, frame)]
            if flag(status['people_complete']):
                person_frames += 1
                truth = [r for r in gt if r['object_type'] == 'player']
                for key, predictions, tally in (
                        ('raw', cached['people'][frame], person_counts),
                        ('tracked', cached['tracked'][frame], tracked_counts)):
                    if key == 'raw' and not cached['raw_people_available']:
                        continue
                    matches = match_pairs(predictions, truth,
                                          lambda a, b: box_iou(bbox(a), bbox(b)) >= person_iou,
                                          lambda a, b: -box_iou(bbox(a), bbox(b)))
                    tally.update(tp=len(matches), fp=len(predictions)-len(matches), fn=len(truth)-len(matches))
                if flag(status['identities_complete']):
                    complete_identity_frames += 1
            state = status['ball_state']
            if state in ('uncertain', 'unreviewed'):
                continue
            truth = [r for r in gt if r['object_type'] == 'ball']
            if state in ('visible', 'partially_occluded') and len(truth) != 1:
                raise ValueError('Reviewed visible frame needs exactly one human ball: '+window_id+':'+str(frame))
            ball_frames += 1
            predicted = cached['selected'].get(frame)
            if state in ('visible', 'partially_occluded'):
                positives += 1
                target = truth[0]
                hit = any(distance(r, target) <= tolerance for r in cached['balls'][frame])
                raw_hits += hit
                error = distance(predicted, target) if predicted else None
                good = error is not None and error <= tolerance
                selection_misses += int(hit and not good)
                selected_counts.update(tp=int(good), fp=int(predicted is not None and not good), fn=int(not good))
                if error is not None:
                    all_errors.append(error)
                    if good:
                        errors.append(error)
                tags = ['state:'+state]+['tag:'+t for t in target['error_groups'].split(';') if t]
                for tag in tags:
                    groups[tag].update(visible_frames=1, raw_candidate_hits=int(hit), selected_hits=int(good))
            else:
                selected_counts.update(fp=int(predicted is not None))
                groups['state:'+state].update(negative_frames=1, selected_false_observations=int(predicted is not None))
            if predicted and (not truth or distance(predicted, truth[0]) > tolerance):
                rejected = [r for r in all_labels[(window_id, frame)] if human_reviewed(r) and
                            r['decision'] == 'rejected' and r['object_type'] == 'ball' and r['fp_group']]
                matched = [r for r in rejected if distance(predicted, r) <= tolerance]
                false_groups[matched[0]['fp_group'] if matched else 'unknown'] += 1
        mean = lambda xs: sum(xs)/len(xs) if xs else None
        per_window.append(dict(window_id=window_id, source_id=window['source_id'], scene_id=window['scene_id'],
            angle=window['angle'], angle_reviewed=window.get('angle_reviewed', False),
            split=window['split'], selection=window['selection'],
            window_frames=window['end_frame']-window['start_frame'], human_reviewed_frames=human_frames,
            evaluated_person_frames=person_frames,
            reviewed_but_unprocessed_frames=skipped, evaluated_ball_frames=ball_frames,
            visible_ball_frames=positives, raw_ball_candidate_hits=raw_hits,
            visible_without_raw_candidate=positives-raw_hits,
            selection_misses_despite_raw_candidate=selection_misses,
            raw_ball_candidate_recall=raw_hits/positives if positives else None,
            selected_ball=counts(**{k:selected_counts[k] for k in ('tp','fp','fn')}) if ball_frames else None,
            center_error_matched_mean_px=mean(errors), center_error_all_selected_visible_mean_px=mean(all_errors),
            center_error_sample_count=len(all_errors), ball_groups={k:dict(v) for k,v in groups.items()},
            selected_false_positive_groups=dict(false_groups),
            raw_person=counts(**{k:person_counts[k] for k in ('tp','fp','fn')}) if person_counts else None,
            tracked_person=counts(**{k:tracked_counts[k] for k in ('tp','fp','fn')}) if tracked_counts else None,
            complete_identity_frames=complete_identity_frames,
            identity_metrics=None,
            identity_note='Use export_trackeval.py; no homemade IDF1/HOTA. Incomplete windows are ineligible.'))
    # Event coverage is only scored after the reviewer explicitly audits the full window.
    completed = {r['window_id'] for r in read_rows(directory/'window_reviews.csv')
                 if human_reviewed(r) and flag(r['events_complete'])}
    event_reviews = [r for r in read_rows(directory/'event_labels.csv') if human_reviewed(r)]
    event_labels = [r for r in event_reviews if
                    r['decision'] in ('approved','corrected') and not r['duplicate_of']]
    predictions = read_rows(directory/'event_candidates.csv')
    events = []
    canonical = lambda row: 'shot' if row['event_type'] == 'shot_candidate' else row['event_type']
    for window_id, window in windows.items():
        if window['kind'] != 'events':
            continue
        source = sources[window['source_id']]
        if window_id not in completed:
            events.append(dict(window_id=window_id, metrics=None, reason='Full event window not human-reviewed'))
            continue
        truth = [r for r in event_labels if r['window_id'] == window_id and r['event_type'] not in ('unknown','shot_candidate')]
        guess = [r for r in predictions if r['window_id'] == window_id and r['context_status'] != 'cut_limited_unevaluable']
        # Exclude explicitly unassessable boundary intervals from both sides.
        exclusions = [r for r in read_rows(directory/'unevaluable_intervals.csv')
                      if r['source_id'] == source['source_id'] and int(r['scene_id']) == window['scene_id']]
        uncertain = [r for r in event_reviews if r['window_id'] == window_id and
                     r['decision'] in ('uncertain','unevaluable')]
        for row in uncertain:
            padding = math.ceil(max(config['event_tolerances_s'])*source['fps'])
            exclusions.append(dict(start_frame=int(row.get('interval_start_frame') or max(
                window['start_frame'], int(row['frame'])-padding)), end_frame=int(
                row.get('interval_end_frame') or min(window['end_frame'], int(row['frame'])+padding+1))))
        valid = lambda r: not any(int(e['start_frame']) <= int(r['frame']) < int(e['end_frame']) for e in exclusions)
        truth, guess = [r for r in truth if valid(r)], [r for r in guess if valid(r)]
        metrics = {}
        for seconds in config['event_tolerances_s']:
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Invalid event tolerance')
            delta = lambda a,b: abs(int(a['frame'])-int(b['frame']))/source['fps']
            for typed in (True, False):
                matches = match_pairs(guess, truth, lambda a,b: delta(a,b) <= seconds and (
                    not typed or canonical(a) == canonical(b)), delta)
                metrics[f"{'typed' if typed else 'proposal_coverage'}_at_{seconds}s"] = dict(
                    **counts(len(matches), len(guess)-len(matches), len(truth)-len(matches)),
                    matched_time_errors_s=[delta(guess[i],truth[j]) for i,j in matches],
                    actor_labeled=sum(bool(truth[j]['actor_id']) for _,j in matches),
                    actor_accuracy=None, actor_note='No verified tracker-to-human identity mapping; not scored')
                if typed:
                    per_type = {}
                    for kind in sorted({canonical(r) for r in truth+guess}):
                        left = [r for r in guess if canonical(r) == kind]
                        right = [r for r in truth if canonical(r) == kind]
                        pairs = match_pairs(left, right, lambda a,b: delta(a,b) <= seconds, delta)
                        per_type[kind] = counts(len(pairs),len(left)-len(pairs),len(right)-len(pairs))
                    metrics[f'typed_at_{seconds}s']['per_type'] = per_type
        events.append(dict(window_id=window_id, source_id=window['source_id'], scene_id=window['scene_id'],
                           angle=window['angle'], selection=window['selection'],
                           human_events=len(truth), proposed_events=len(guess),
                           excluded_uncertain_intervals=len(uncertain), metrics=metrics))
    report = dict(schema_version=1, split=split, configuration=config,
                  evaluation_config_sha256=sha256_file(config_path),
                  source_provenance=[dict(source_id=s['source_id'], model=s['model'],
                    source_sha256=s['source_sha256'], cache_sha256=s['cache_sha256'],
                    invocation=s['invocation']) for s in dataset['sources']],
                  per_window=per_window, event_windows=events,
                  human_reviewed_frame_rows=sum(human_reviewed(r) for rows in frames.values() for r in rows.values()),
                  ai_reviewed_excluded=sum(r.get('annotation_source') == 'ai_reviewed'
                                          for rows in frames.values() for r in rows.values()),
                  human_confirmed_event_counts=dict(Counter(r['event_type'] for r in event_labels
                                                            if r['window_id'] in windows and
                                                            r['event_type'] not in ('unknown','shot_candidate'))),
                  claim='No human labels means no accuracy estimate; counts and replay matches are not accuracy.',
                  alternate_detector_gate='needs_human_visible_ball_labels' if not any(
                      w['visible_ball_frames'] for w in per_window) else 'inspect_candidate_misses_by_angle_before_deciding',
                  limitations=dataset['limitations'])
    path = Path(output) if output else directory/f'evaluation_{split}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='results/review_v04')
    parser.add_argument('--split', choices=['dev','eval','all'], default='eval')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = evaluate(args.dataset, args.split, args.output)
    print(json.dumps(dict(split=report['split'], human_reviewed_frame_rows=report['human_reviewed_frame_rows'],
                          human_confirmed_events=report['human_confirmed_event_counts'],
                          claim=report['claim']), indent=2))
