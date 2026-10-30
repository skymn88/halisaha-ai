"""Review schema v1: proposals, AI observations and human labels remain distinct."""
import csv
import json
import math
import os
from pathlib import Path


SCHEMA_VERSION = 1
BALL_STATES = ('visible', 'partially_occluded', 'fully_occluded', 'out_of_frame', 'uncertain', 'unreviewed')
ERROR_GROUPS = ('small_far', 'fast_blurred', 'on_line', 'person_overlap', 'stationary')
FP_GROUPS = ('white_marking', 'clothing_footwear', 'other', 'unknown')
SOURCES = ('proposal', 'ai_reviewed', 'human')
EVENT_TYPES = ('pass_completed', 'pass_attempt', 'shot_candidate', 'shot',
               'control_change', 'ball_recovery', 'unknown')
COMMON = ['window_id', 'source_id', 'scene_id', 'frame']
PROVENANCE = ['review_status', 'annotation_source', 'reviewer', 'notes']
FRAME_FIELDS = COMMON + ['ball_state', 'people_complete', 'identities_complete'] + PROVENANCE
DETECTION_FIELDS = COMMON + ['annotation_id', 'object_type', 'decision', 'x1', 'y1', 'x2', 'y2',
                             'center_x', 'center_y', 'proposal_id', 'error_groups', 'fp_group'] + PROVENANCE
IDENTITY_FIELDS = COMMON + ['annotation_id', 'identity_id', 'team_id', 'role', 'track_hint'] + PROVENANCE
WINDOW_FIELDS = ['window_id', 'events_complete', 'review_status', 'annotation_source', 'reviewer', 'notes']
EVENT_FIELDS = ['event_id', 'window_id', 'source_id', 'scene_id', 'frame', 'event_type',
                'actor_id', 'recipient_id', 'team_id', 'decision', 'candidate_id', 'duplicate_of',
                'interval_start_frame', 'interval_end_frame'] + PROVENANCE
TABLES = {'frame_labels.csv': FRAME_FIELDS, 'detection_labels.csv': DETECTION_FIELDS,
          'identity_labels.csv': IDENTITY_FIELDS, 'event_labels.csv': EVENT_FIELDS,
          'window_reviews.csv': WINDOW_FIELDS}


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fields, rows):
    """Replace a CSV atomically on one filesystem; never truncate it before serialization."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def human_reviewed(row):
    return row.get('annotation_source') == 'human' and row.get('review_status') == 'reviewed'


def flag(value):
    return str(value).lower() in ('1', 'true')


def provenance(row):
    if row.get('annotation_source') not in SOURCES or row.get('review_status') not in ('reviewed', 'unreviewed'):
        raise ValueError('Invalid review provenance')
    if row['review_status'] == 'reviewed' and (
            row['annotation_source'] == 'proposal' or not row.get('reviewer', '').strip()):
        raise ValueError('Reviewed labels need human/ai_reviewed source and named reviewer')


def validate_frame(frame, detections, identities, size):
    """Completeness is an explicit reviewer assertion, never inferred from proposals."""
    provenance(frame)
    if frame['ball_state'] not in BALL_STATES:
        raise ValueError('Invalid ball state')
    accepted, ids = [], set()
    width, height = size
    for row in detections:
        provenance(row)
        if any(str(row[k]) != str(frame[k]) for k in COMMON):
            raise ValueError('Detection/frame scope mismatch')
        if row['annotation_id'] in ids:
            raise ValueError('Duplicate annotation ID')
        ids.add(row['annotation_id'])
        if row['object_type'] not in ('player', 'ball') or row['decision'] not in ('accepted', 'rejected'):
            raise ValueError('Invalid annotation type/decision')
        values = [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2', 'center_x', 'center_y')]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite label coordinates')
        x1, y1, x2, y2, cx, cy = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 0 <= cx <= width and 0 <= cy <= height):
            raise ValueError('Label outside source image or degenerate')
        if set(filter(None, row.get('error_groups', '').split(';')))-set(ERROR_GROUPS):
            raise ValueError('Unknown error group')
        if row.get('fp_group') and row['fp_group'] not in FP_GROUPS:
            raise ValueError('Unknown false positive group')
        if row['decision'] == 'accepted':
            accepted.append(row)
    balls = [r for r in accepted if r['object_type'] == 'ball']
    if len(balls) > 1:
        raise ValueError('At most one ground-truth football per frame')
    if frame['review_status'] == 'reviewed':
        if frame['ball_state'] in ('visible', 'partially_occluded') and len(balls) != 1:
            raise ValueError('Visible ball needs one explicitly corrected/accepted ball label')
        if frame['ball_state'] in ('fully_occluded', 'out_of_frame') and balls:
            raise ValueError('Do not invent observed coordinates for hidden/out-of-frame balls')
        if any(r['annotation_source'] != frame['annotation_source'] or r['review_status'] != 'reviewed'
               for r in detections):
            raise ValueError('Reviewed frame and detections must have consistent provenance')
    person_ids = {r['annotation_id'] for r in accepted if r['object_type'] == 'player'}
    identity_map = {}
    for row in identities:
        provenance(row)
        if any(str(row[k]) != str(frame[k]) for k in COMMON):
            raise ValueError('Identity/frame scope mismatch')
        if row['annotation_id'] not in person_ids or row['annotation_id'] in identity_map:
            raise ValueError('Identity must reference exactly one accepted player annotation')
        identity_map[row['annotation_id']] = row
    if flag(frame.get('identities_complete')):
        if not flag(frame.get('people_complete')) or set(identity_map) != person_ids:
            raise ValueError('Complete identities require complete player boxes and identity rows')
        values = [r['identity_id'].strip() for r in identity_map.values()]
        if not all(values) or len(set(values)) != len(values):
            raise ValueError('Complete identities need distinct nonempty IDs in each frame')
        if any(r['annotation_source'] != frame['annotation_source'] or
               r['review_status'] != frame['review_status'] for r in identity_map.values()):
            raise ValueError('Identity provenance differs from frame')


def load_dataset(directory):
    directory = Path(directory)
    data = json.loads((directory/'dataset.json').read_text(encoding='utf-8'))
    if data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported review dataset version')
    seen = set()
    groups = {}
    scenes = {}
    sources = {s['source_id']: s for s in data['sources']}
    if len(sources) != len(data['sources']):
        raise ValueError('Duplicate source ID')
    for source in sources.values():
        if not math.isfinite(source['fps']) or source['fps'] <= 0 or min(source['image_size']) <= 0:
            raise ValueError('Invalid source FPS/image size')
    for window in data['windows']:
        if window['window_id'] in seen or window['split'] not in ('dev', 'eval'):
            raise ValueError('Duplicate window or unknown split')
        seen.add(window['window_id'])
        if window['end_frame'] <= window['start_frame']:
            raise ValueError('Empty window')
        group = window['split_group']
        if group in groups and groups[group] != window['split']:
            raise ValueError('Split leakage: a scene/group belongs to dev and eval')
        groups[group] = window['split']
        source = sources[window['source_id']]
        bounds = [s for s in source['scenes'] if s['scene_id'] == window['scene_id']]
        if len(bounds) != 1 or not (bounds[0]['start_frame'] <= window['start_frame'] <
                                   window['end_frame'] <= bounds[0]['end_frame']):
            raise ValueError('Review window crosses or misidentifies a source scene')
        key = (window['source_id'], window['scene_id'])
        if key in scenes and scenes[key] != window['split']:
            raise ValueError('Same source scene appears in both splits')
        scenes[key] = window['split']
    return data
