"""Versioned image evidence; no tracker state or inferred ball coordinates are cached."""
import csv
import hashlib
import json
import math
from pathlib import Path


FEATURE_VERSION = 'pitch_max_core_v1'
FIELDS = ['frame', 'scene_id', 'proposal_index', 'confidence', 'x1', 'y1', 'x2', 'y2',
          'background_marking_fraction', 'foreground_fraction', 'white_marking_cue',
          'body_overlap', 'person_overlap', 'outside_pitch']


def sha256_file(path):
    """Hash an existing artifact without treating a file name as model identity."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def feature_contract(difference=30., foreground_min=.25):
    return dict(version=FEATURE_VERSION, difference_bgr_max=difference,
                foreground_min=foreground_min, marking_min=.25,
                white_hsv_s_lt=80, white_hsv_v_gt=140,
                region='max(full_box,central_quarter_to_three_quarters)',
                body_height_fraction=.8, coordinates='source_pixels',
                geometry='cached center containment; rebuild when geometry changes',
                precision='unrounded Python float', temporal_features_cached=False)


def collect_evidence(candidates, image, background, geometry=None, people=()):
    records = []
    for box, confidence in candidates:
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        records.append(dict(box=tuple(box), confidence=float(confidence),
                            **background.evidence(image, box),
                            outside_pitch=geometry is not None and not geometry.contains(center),
                            body_overlap=any(p[0] <= center[0] <= p[2] and
                                             p[1] <= center[1] <= p[1]+.8*(p[3]-p[1]) for p in people),
                            person_overlap=any(p[0] <= center[0] <= p[2] and
                                               p[1] <= center[1] <= p[3] for p in people)))
    return records


def evidence_row(frame, scene_id, index, record):
    return [frame, scene_id, index, record['confidence'], *record['box'],
            record['marking_fraction'], record['foreground_fraction'],
            int(record['static_marking']), int(record['body_overlap']),
            int(record['person_overlap']), int(record['outside_pitch'])]


def read_evidence(path):
    """Reject incomplete historical caches rather than silently assuming missing context."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1 or metadata.get('feature_contract', {}).get('version') != FEATURE_VERSION:
        raise ValueError('Unsupported ball evidence schema/feature version')
    expected = feature_contract(metadata['parameters'].get('background_difference', 30.),
                                metadata['parameters'].get('foreground_min', .25))
    if any(metadata['feature_contract'].get(k) != v for k, v in expected.items() if k != 'precision'):
        raise ValueError('Feature contract/parameters differ; rebuild image invariants')
    groups, indices = {}, set()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Incomplete ball evidence columns')
        for row in reader:
            frame, scene, index = (int(row[k]) for k in ('frame', 'scene_id', 'proposal_index'))
            key = (scene, frame, index)
            if key in indices or min(key) < 0:
                raise ValueError('Duplicate or invalid evidence index')
            indices.add(key)
            values = [float(row[k]) for k in ('confidence', 'x1', 'y1', 'x2', 'y2',
                                             'background_marking_fraction', 'foreground_fraction')]
            if not all(math.isfinite(x) for x in values) or not all(0 <= values[i] <= 1 for i in (0, 5, 6)):
                raise ValueError('Invalid evidence values')
            if values[3] <= values[1] or values[4] <= values[2]:
                raise ValueError('Degenerate evidence box')
            flags = {}
            for key_name in ('white_marking_cue', 'body_overlap', 'person_overlap', 'outside_pitch'):
                if row[key_name] not in ('0', '1'):
                    raise ValueError('Invalid evidence flag')
                flags[key_name] = row[key_name] == '1'
            groups.setdefault(frame, []).append((index, scene, dict(
                confidence=values[0], box=tuple(values[1:5]), marking_fraction=values[5],
                foreground_fraction=values[6], static_marking=flags['white_marking_cue'],
                **{k: flags[k] for k in ('body_overlap', 'person_overlap', 'outside_pitch')})))
    timeline = metadata['timeline']
    if not timeline or any(b['frame'] <= a['frame'] for a, b in zip(timeline, timeline[1:])):
        raise ValueError('Timeline must contain increasing processed source frames')
    frames = {r['frame']: r['scene_id'] for r in timeline}
    for frame, rows in groups.items():
        if frame not in frames or any(scene != frames[frame] for _, scene, _ in rows):
            raise ValueError('Evidence/timeline scene mismatch')
        rows.sort(key=lambda item: item[0])
    return metadata, {frame: [r for _, _, r in rows] for frame, rows in groups.items()}


def write_metadata(path, raw_meta, parameters, timeline, references=None, provenance=None):
    result = dict(schema_version=1, feature_contract=feature_contract(
        parameters.get('background_difference', 30.), parameters.get('foreground_min', .25)),
        source=raw_meta, parameters=parameters, timeline=timeline,
        references=references or {}, provenance=provenance or {})
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
