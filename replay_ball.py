"""Build full precision evidence from saved proposals, or replay it without video/YOLO."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import cv2
from ball_cache import (FIELDS, collect_evidence, evidence_row, read_evidence,
                        sha256_file, write_metadata)
from ball_tracking import PitchBallTracker
from calibration import load_geometries
from detection_io import read_detections
from pitch_context import PitchBackground


CANDIDATE_FIELDS = ['frame', 'confidence', 'x1', 'y1', 'x2', 'y2', 'ball_id', 'status',
                    'white_marking_cue', 'scene_id', 'background_marking_fraction',
                    'foreground_fraction', 'innovation_px']
TRACK_FIELDS = ['frame', 'time_s', 'scene_id', 'ball_id', 'state', 'confidence', 'center_x', 'center_y']


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Use a fresh output directory: '+str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_cache(run, output, video=None):
    """Recompute image invariants using existing raw scores and saved backgrounds, never YOLO."""
    run, output = Path(run), fresh_output(output)
    raw_meta = json.loads((run/'raw_detections.meta.json').read_text(encoding='utf-8'))
    parameters = json.loads((run/'summary.json').read_text(encoding='utf-8'))['parameters']
    if parameters.get('ball_mode', 'pitch') != 'pitch':
        raise ValueError('Evidence cache supports pitch mode only')
    records = defaultdict(list)
    for record in read_detections(run/'raw_detections.csv'):
        records[record.frame].append(record)
    timeline = [dict(frame=int(r['frame']), scene_id=int(r['scene_id']), time_s=float(r['time_s']))
                for r in read_csv(run/'ball_track.csv')]
    geometry_path = run/'camera_config.json'
    geometries = load_geometries(geometry_path if geometry_path.exists() else None,
                                raw_meta['image_size'], {r['scene_id'] for r in timeline})
    backgrounds, references = {}, {}
    for scene in {r['scene_id'] for r in timeline}:
        path = run/'pitch_context'/f'scene_{scene:03d}_background.png'
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Missing saved reference: '+str(path))
        backgrounds[scene] = PitchBackground(image, parameters['background_difference'],
                                             parameters['foreground_min'])
        references[str(scene)] = dict(background_sha256=sha256_file(path))
    cap = cv2.VideoCapture(str(video or raw_meta['source_video']))
    if not cap.isOpened():
        raise ValueError('Cannot open raw video; image evidence cannot be reconstructed from rounded legacy columns alone')
    try:
        with (output/'ball_evidence.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            for item in timeline:
                frame, scene = item['frame'], item['scene_id']
                balls = [d for d in records[frame] if not d.is_person]
                if not balls:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode raw source frame {frame}')
                if (image.shape[1], image.shape[0]) != tuple(raw_meta['image_size']):
                    raise ValueError('Raw image dimensions do not match cache')
                if any(d.scene_id != scene for d in records[frame]):
                    raise ValueError('Raw detection scene does not match timeline')
                evidence = collect_evidence([(d.box, d.confidence) for d in balls], image,
                                           backgrounds[scene], geometries.get(scene),
                                           [d.box for d in records[frame] if d.is_person])
                writer.writerows(evidence_row(frame, scene, i, row) for i, row in enumerate(evidence))
    finally:
        cap.release()
    write_metadata(output/'ball_evidence.meta.json', raw_meta, parameters, timeline, references,
                   dict(kind='reconstructed_from_raw_and_saved_reference', detector_rerun=False,
                        raw_sha256=sha256_file(run/'raw_detections.csv'),
                        source_video_sha256=sha256_file(video or raw_meta['source_video']),
                        geometry_sha256=sha256_file(geometry_path) if geometry_path.exists() else None,
                        historical_detector=raw_meta.get('detector', {'version': 'not_recorded'})))
    return output/'ball_evidence.csv'


def build_csv_cache(run, output, feature_version):
    """Import saved fractions and derive overlap from raw person boxes, without opening images."""
    from ball_cache import FEATURE_VERSION
    if feature_version != FEATURE_VERSION:
        raise ValueError('Explicitly identify the historical feature definition; v0.3 mean fractions are incompatible')
    run, output = Path(run), fresh_output(output)
    raw_meta = json.loads((run/'raw_detections.meta.json').read_text(encoding='utf-8'))
    parameters = json.loads((run/'summary.json').read_text(encoding='utf-8'))['parameters']
    if parameters.get('ball_mode', 'pitch') != 'pitch':
        raise ValueError('Legacy v02 features are not pitch evidence')
    raw, people = defaultdict(list), defaultdict(list)
    for detection in read_detections(run/'raw_detections.csv'):
        (people if detection.is_person else raw)[detection.frame].append(detection)
    geometry_path = run/'camera_config.json'
    geometries = load_geometries(geometry_path if geometry_path.exists() else None, raw_meta['image_size'],
                                {s['scene_id'] for s in raw_meta['scenes']})
    saved = defaultdict(list)
    for row in read_csv(run/'ball_candidates.csv'):
        saved[int(row['frame'])].append(row)
    used = set()
    with (output/'ball_evidence.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for frame, observations in raw.items():
            for index, detection in enumerate(observations):
                matches = [(i, r) for i, r in enumerate(saved[frame]) if (frame, i) not in used and
                           tuple(float(r[k]) for k in ('x1','y1','x2','y2')) == detection.box and
                           abs(float(r['confidence'])-detection.confidence) <= .000051]
                if len(matches) != 1:
                    raise ValueError('Raw/candidate correspondence is missing or ambiguous')
                i, row = matches[0]
                used.add((frame, i))
                if any(row.get(k) in (None, '') for k in ('background_marking_fraction','foreground_fraction','white_marking_cue')):
                    raise ValueError('Saved image invariants are incomplete')
                center = ((detection.box[0]+detection.box[2])/2, (detection.box[1]+detection.box[3])/2)
                geometry = geometries.get(detection.scene_id)
                boxes = [d.box for d in people[frame]]
                evidence = dict(box=detection.box, confidence=detection.confidence,
                    marking_fraction=float(row['background_marking_fraction']),
                    foreground_fraction=float(row['foreground_fraction']),
                    static_marking=bool(int(row['white_marking_cue'])),
                    outside_pitch=geometry is not None and not geometry.contains(center),
                    body_overlap=any(p[0] <= center[0] <= p[2] and p[1] <= center[1] <= p[1]+.8*(p[3]-p[1]) for p in boxes),
                    person_overlap=any(p[0] <= center[0] <= p[2] and p[1] <= center[1] <= p[3] for p in boxes))
                writer.writerow(evidence_row(frame, detection.scene_id, index, evidence))
    if len(used) != sum(len(rows) for rows in saved.values()):
        raise ValueError('Not all saved proposals matched raw observations')
    timeline = [dict(frame=int(r['frame']), scene_id=int(r['scene_id']), time_s=float(r['time_s']))
                for r in read_csv(run/'ball_track.csv')]
    path = output/'ball_evidence.meta.json'
    write_metadata(path, raw_meta, parameters, timeline, provenance=dict(
        kind='csv_only_legacy_import', raw_sha256=sha256_file(run/'raw_detections.csv'),
        candidates_sha256=sha256_file(run/'ball_candidates.csv'),
        feature_definition_explicitly_supplied=feature_version, images_opened=False,
        limitation='Fractions were rounded to 4 decimals by the historical exporter; equivalence must be checked.'))
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata['feature_contract']['precision'] = 'historical_fractions_rounded_4_decimals; raw_scores_full_precision'
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return output/'ball_evidence.csv'


def replay(path, output, reference_run=None):
    metadata, groups = read_evidence(path)
    output = fresh_output(output)
    parameters = metadata['parameters']
    tracker, previous_scene = None, None
    candidates, states = [], []
    for item in metadata['timeline']:
        frame, scene, time_s = item['frame'], item['scene_id'], item['time_s']
        if scene != previous_scene:
            tracker = PitchBallTracker(parameters['ball_hits'], parameters['ball_gap'],
                                       parameters['ball_radius'], parameters.get('ball_reacquire_hits', 3))
            previous_scene = scene
        accepted = tracker.update_evidence(frame, groups.get(frame, []))
        for row in tracker.last_diagnostics:
            candidates.append([frame, round(row['confidence'], 4), *row['box'], row['ball_id'],
                               row['status'], int(row['static_marking']), scene,
                               round(row['marking_fraction'], 4), round(row['foreground_fraction'], 4),
                               row['innovation_px']])
        if accepted:
            box, score, tid, _, _ = accepted[0]
            states.append([frame, time_s, scene, tid, 'observed', round(score, 4),
                           (box[0]+box[2])/2, (box[1]+box[3])/2])
        else:
            active = tracker.active
            states.append([frame, time_s, scene, active['id'] if active else -1,
                           tracker.last_state, '', '', ''])
    for name, fields, rows in [('ball_candidates.csv', CANDIDATE_FIELDS, candidates),
                               ('ball_track.csv', TRACK_FIELDS, states)]:
        with (output/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    report = dict(processed_frames=len(states), candidates=len(candidates),
                  accepted_ball_rows=sum(r[4] == 'observed' for r in states),
                  evidence_sha256=sha256_file(path), image_or_detector_used=False,
                  feature_contract=metadata['feature_contract'],
                  note='Replay equivalence is not a measure of ball accuracy.')
    if reference_run:
        mismatches = []
        for name in ('ball_candidates.csv', 'ball_track.csv'):
            expected, actual = read_csv(Path(reference_run)/name), read_csv(output/name)
            if len(expected) != len(actual):
                mismatches.append(dict(file=name, reason='row_count', expected=len(expected), actual=len(actual)))
                continue
            for i, (a, b) in enumerate(zip(expected, actual)):
                for key in a:
                    if a[key] == b.get(key):
                        continue
                    try:
                        equal = abs(float(a[key])-float(b[key])) <= 1e-9
                    except (ValueError, TypeError):
                        equal = False
                    if not equal:
                        mismatches.append(dict(file=name, row=i, column=key, expected=a[key], actual=b.get(key)))
        report['reference_match'] = not mismatches
        report['mismatch_count'] = len(mismatches)
        report['mismatch_examples'] = mismatches[:20]
    (output/'replay_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if reference_run and not report['reference_match']:
        raise ValueError('Replay differs from reference; see replay_report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--run', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--video')
    build.add_argument('--legacy-features', action='store_true', help='CSV-only import; requires --feature-version')
    build.add_argument('--feature-version', help='Historical feature definition, explicitly supplied for CSV-only import')
    play = sub.add_parser('replay')
    play.add_argument('--cache', required=True)
    play.add_argument('--output', required=True)
    play.add_argument('--reference-run')
    args = parser.parse_args()
    if args.command == 'build':
        if args.legacy_features:
            print(build_csv_cache(args.run, args.output, args.feature_version))
        else:
            print(build_cache(args.run, args.output, args.video))
    else:
        print(json.dumps(replay(args.cache, args.output, args.reference_run), indent=2))


if __name__ == '__main__':
    main()
