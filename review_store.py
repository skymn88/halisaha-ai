"""Small CSV-backed review store with explicit provenance and append-only revisions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import cv2
from player_tracking import box_iou
from review_schema import (TABLES, EVENT_TYPES, flag, load_dataset, provenance,
                           read_rows, validate_frame, write_rows)


class ReviewStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.dataset = load_dataset(directory)
        self.windows = {w['window_id']: w for w in self.dataset['windows']}
        self.sources = {s['source_id']: s for s in self.dataset['sources']}
        self.proposals = {}
        for source in self.sources.values():
            run = Path(source['run'])
            path = run/'raw_detections.csv'
            rows = read_rows(path if path.exists() else run/'detections.csv')
            tracked = {}
            for row in read_rows(run/'detections.csv'):
                if row['class_name'] == 'person' and int(row['track_id']) >= 0:
                    tracked.setdefault(int(row['frame']), []).append(row)
            by_frame = {}
            for i, row in enumerate(rows):
                frame = int(row['frame'])
                box = [float(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
                person = row['class_name'] in ('person', 'player')
                hint = ''
                if person:
                    choices = [(box_iou(box, [float(t[k]) for k in ('x1', 'y1', 'x2', 'y2')]), t)
                               for t in tracked.get(frame, [])]
                    if choices:
                        score, match = max(choices, key=lambda item: item[0])
                        if score >= .5:
                            hint = match['track_id']
                by_frame.setdefault(frame, []).append(dict(
                    annotation_id=f"p_{source['source_id']}_{i}", object_type='player' if person else 'ball',
                    decision='accepted', x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    center_x=(box[0]+box[2])/2, center_y=(box[1]+box[3])/2,
                    proposal_id=str(i), error_groups='', fp_group='', track_hint=hint,
                    identity_id='', team_id='', role='unknown', score=row['confidence']))
            self.proposals[source['source_id']] = by_frame

    def window(self, window_id, frame=None):
        window = self.windows[window_id]
        if frame is not None and not window['start_frame'] <= frame < window['end_frame']:
            raise ValueError('Frame outside review window')
        return window

    def frame(self, window_id, frame):
        window = self.window(window_id, frame)
        match = lambda row: row['window_id'] == window_id and int(row['frame']) == frame
        frames = [r for r in read_rows(self.directory/'frame_labels.csv') if match(r)]
        detections = [r for r in read_rows(self.directory/'detection_labels.csv') if match(r)]
        identity = {r['annotation_id']: r for r in read_rows(self.directory/'identity_labels.csv') if match(r)}
        if not frames:
            record = dict(window_id=window_id, source_id=window['source_id'], scene_id=window['scene_id'],
                          frame=frame, ball_state='unreviewed', people_complete=0, identities_complete=0,
                          review_status='unreviewed', annotation_source='proposal', reviewer='', notes='')
        else:
            record = frames[0]
        if not detections and record['review_status'] == 'unreviewed':
            detections = [dict(row) for row in self.proposals[window['source_id']].get(frame, [])]
        for row in detections:
            row.update({k: v for k, v in identity.get(row['annotation_id'], {}).items()
                        if k in ('identity_id', 'team_id', 'role', 'track_hint')})
        return dict(frame=record, detections=detections, window=window,
                    source=self.sources[window['source_id']])

    def image(self, window_id, frame):
        window = self.window(window_id, frame)
        path = self.directory/'frames'/window_id/f'{frame:06d}.png'
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            cap = cv2.VideoCapture(self.sources[window['source_id']]['video'])
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Cannot decode raw source frame')
            finally:
                cap.release()
        return image

    def revise(self, operation, tables, details):
        before = {name: read_rows(self.directory/name) for name in tables}
        def key(row):
            return tuple(str(row.get(k, '')) for k in ('window_id','frame','annotation_id','event_id'))
        changed_before, changed_after = {}, {}
        for name, rows in tables.items():
            old = {key(r):r for r in before[name]}
            new = {key(r):r for r in rows}
            changed = {k for k in set(old)|set(new) if
                       {a:str(b) for a,b in old.get(k, {}).items()} !=
                       {a:str(b) for a,b in new.get(k, {}).items()}}
            changed_before[name] = [old[k] for k in sorted(changed) if k in old]
            changed_after[name] = [new[k] for k in sorted(changed) if k in new]
        revision = dict(revision_id=uuid.uuid4().hex,
                        time_utc=datetime.now(timezone.utc).isoformat(),
                        operation=operation, details=details, format='row_deltas_v1',
                        before=changed_before, after=changed_after)
        # Journal first so a failed/partial multi-file write has recoverable evidence.
        with (self.directory/'review_history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(revision, ensure_ascii=False)+'\n')
        for name, rows in tables.items():
            write_rows(self.directory/name, TABLES[name], rows)
        return revision['revision_id']

    def save_frame(self, payload):
        frame = int(payload['frame'])
        window = self.window(payload['window_id'], frame)
        source = payload['annotation_source']
        reviewed = bool(payload.get('reviewed'))
        common = dict(window_id=window['window_id'], source_id=window['source_id'],
                      scene_id=window['scene_id'], frame=frame, annotation_source=source,
                      review_status='reviewed' if reviewed else 'unreviewed',
                      reviewer=payload.get('reviewer', '').strip(), notes=payload.get('notes', ''))
        record = dict(common, ball_state=payload['ball_state'],
                      people_complete=int(bool(payload.get('people_complete'))),
                      identities_complete=int(bool(payload.get('identities_complete'))))
        detections, identities = [], []
        for item in payload['detections']:
            row = {k: item.get(k, '') for k in TABLES['detection_labels.csv']}
            row.update(common)
            detections.append(row)
            if item['object_type'] == 'player' and item['decision'] == 'accepted':
                identities.append(dict(common, annotation_id=item['annotation_id'],
                                       identity_id=item.get('identity_id', ''), team_id=item.get('team_id', ''),
                                       role=item.get('role', 'unknown'), track_hint=item.get('track_hint', '')))
        validate_frame(record, detections, identities, self.sources[window['source_id']]['image_size'])
        tables = {}
        for name, replacement in [('frame_labels.csv', [record]), ('detection_labels.csv', detections),
                                  ('identity_labels.csv', identities)]:
            old = [r for r in read_rows(self.directory/name)
                   if not (r['window_id'] == window['window_id'] and int(r['frame']) == frame)]
            tables[name] = old+replacement
        return self.revise('save_frame', tables, common)

    def save_event(self, payload):
        window = self.window(payload['window_id'], int(payload['frame']))
        if payload['event_type'] not in EVENT_TYPES or payload['decision'] not in (
                'approved', 'rejected', 'corrected', 'uncertain', 'unevaluable'):
            raise ValueError('Invalid event type/decision')
        row = {k: payload.get(k, '') for k in TABLES['event_labels.csv']}
        row.update(event_id=payload.get('event_id') or uuid.uuid4().hex,
                   source_id=window['source_id'], scene_id=window['scene_id'],
                   review_status='reviewed')
        provenance(row)
        start, end = row['interval_start_frame'], row['interval_end_frame']
        if start != '' or end != '':
            if not (start != '' and end != '' and window['start_frame'] <= int(start) < int(end) <= window['end_frame']):
                raise ValueError('Uncertain interval must be a nonempty [start,end) inside the window')
        if row['duplicate_of']:
            existing = {r['event_id'] for r in read_rows(self.directory/'event_labels.csv')}
            if row['duplicate_of'] == row['event_id'] or row['duplicate_of'] not in existing:
                raise ValueError('Duplicate must refer to a different existing reviewed event')
        old = [r for r in read_rows(self.directory/'event_labels.csv') if r['event_id'] != row['event_id']]
        return self.revise('save_event', {'event_labels.csv': old+[row]}, row)

    def complete_window(self, payload):
        window = self.window(payload['window_id'])
        row = dict(window_id=window['window_id'], events_complete=int(bool(payload['events_complete'])),
                   annotation_source=payload['annotation_source'], review_status='reviewed',
                   reviewer=payload.get('reviewer', '').strip(), notes=payload.get('notes', ''))
        provenance(row)
        old = [r for r in read_rows(self.directory/'window_reviews.csv') if r['window_id'] != row['window_id']]
        return self.revise('review_event_window', {'window_reviews.csv': old+[row]}, row)
