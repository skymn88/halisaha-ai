"""Unvalidated possession/event proposals in source time; never automatic match statistics."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from review_schema import human_reviewed, load_dataset, read_rows, write_rows


DEFAULTS = dict(enter_s=.15, release_s=.10, max_ball_gap_s=.11, control_radius_heights=.60,
                rival_margin_heights=.15, min_person_conf=.25, min_ball_conf=.15,
                motion_diagonals_per_s=.5, shot_separation_s=.7, context_before_s=2., context_after_s=3.)


class Possession:
    """Hysteresis over observed evidence; missing ball always exports unknown, not a held label."""
    def __init__(self, config=None):
        self.config = dict(DEFAULTS, **(config or {}))
        if set(self.config)-set(DEFAULTS):
            raise ValueError('Unknown possession parameter')
        if any(not math.isfinite(v) or v < 0 for v in self.config.values()):
            raise ValueError('Possession thresholds must be finite and nonnegative')
        if min(self.config[k] for k in ('max_ball_gap_s','control_radius_heights','enter_s','release_s')) <= 0:
            raise ValueError('Gap, entry/release durations and control radius must be positive')
        self.owner = None
        self.last_support = None
        self.pending, self.pending_since = None, None
        self.last_time = None

    def update(self, time_s, ball, people):
        config = self.config
        if self.last_time is not None:
            if time_s <= self.last_time:
                raise ValueError('Possession input time must increase within a scene')
            if time_s-self.last_time > config['max_ball_gap_s']:
                self.pending, self.pending_since, self.owner = None, None, None
        self.last_time = time_s
        nearby, state, proposed = [], 'unknown', None
        if ball is not None and ball['confidence'] >= config['min_ball_conf']:
            for person in people:
                x1, y1, x2, y2 = person['box']
                distance = math.hypot(ball['x']-(x1+x2)/2, ball['y']-y2)/max(1., y2-y1)
                if distance <= config['control_radius_heights']:
                    nearby.append((distance, person))
            nearby.sort(key=lambda item: item[0])
            if nearby:
                best = nearby[0][1]
                if len(nearby) > 1 and nearby[1][0]-nearby[0][0] <= config['rival_margin_heights']:
                    state = 'contested'
                elif best['track_id'] < 0 or best['confidence'] < config['min_person_conf']:
                    state = 'unknown'
                else:
                    proposed = best['track_id']
            elif people:
                state = 'free_ball'
        if proposed is not None:
            if proposed == self.owner:
                self.last_support = time_s
                self.pending, self.pending_since = None, None
                return dict(state='player', owner_id=self.owner, evidence='observed_near_foot_hypothesis')
            if proposed != self.pending:
                self.pending, self.pending_since = proposed, time_s
            # A long missing interval cannot contribute to the entry duration.
            if time_s-self.pending_since >= config['enter_s'] and (
                    self.owner is None or self.last_support is None or
                    time_s-self.last_support >= config['release_s']):
                self.owner, self.last_support = proposed, time_s
                self.pending, self.pending_since = None, None
                return dict(state='player', owner_id=self.owner, evidence='sustained_near_foot_hypothesis')
            state = 'contested' if self.owner is not None else 'unknown'
        else:
            self.pending, self.pending_since = None, None
        if self.last_support is not None and time_s-self.last_support >= config['release_s']:
            self.owner = None
        return dict(state=state, owner_id='', evidence='heuristic_not_ground_truth')


def generate(directory, config=None):
    directory = Path(directory)
    dataset = load_dataset(directory)
    config = dict(DEFAULTS, **(config or {}))
    states, candidates, exclusions = [], [], []
    for source in dataset['sources']:
        run = Path(source['run'])
        people = defaultdict(list)
        for row in read_rows(run/'detections.csv'):
            if row['class_name'] != 'person':
                continue
            people[(int(row.get('scene_id') or 0), int(row['frame']))].append(dict(
                box=[float(row[k]) for k in ('x1','y1','x2','y2')],
                track_id=int(row['track_id']), confidence=float(row['confidence'])))
        diagonal = math.hypot(*source['image_size'])
        tracker, previous_scene, last_ball, last_owner, last_shot = None, None, None, None, -float('inf')
        source_windows = [w for w in dataset['windows'] if w['source_id'] == source['source_id'] and w['kind'] == 'events']
        for row in read_rows(run/'ball_track.csv'):
            frame, scene, now = int(row['frame']), int(row['scene_id']), int(row['frame'])/source['fps']
            if scene != previous_scene:
                tracker, last_ball, last_owner, last_shot = Possession(config), None, None, -float('inf')
                previous_scene = scene
            ball = dict(x=float(row['center_x']), y=float(row['center_y']), confidence=float(row['confidence']),
                        ball_id=row['ball_id']) if row['state'] == 'observed' else None
            owned = tracker.update(now, ball, people[(scene, frame)])
            states.append(dict(source_id=source['source_id'], scene_id=scene, frame=frame, time_s=now,
                               **owned, annotation_source='proposal'))
            windows = [w for w in source_windows if w['scene_id'] == scene and w['start_frame'] <= frame < w['end_frame']]
            if not windows:
                last_ball = None
                last_owner = None
                continue
            window = windows[0]
            scene_info = next(s for s in source['scenes'] if s['scene_id'] == scene)
            clipped = frame/source['fps']-config['context_before_s'] < scene_info['start_frame']/source['fps'] or (
                frame/source['fps']+config['context_after_s'] >= scene_info['end_frame']/source['fps'])
            def emit(event_type, reason, actor='', recipient='', score=''):
                identifier = f"{source['source_id']}_{scene}_{frame}_{event_type}"
                candidates.append(dict(candidate_id=identifier, window_id=window['window_id'],
                    source_id=source['source_id'], scene_id=scene, frame=frame, time_s=now,
                    event_type=event_type, actor_id=actor, recipient_id=recipient, team_id='',
                    reason=reason, diagnostic_score=score, status='human_review_required',
                    context_status='cut_limited_unevaluable' if clipped else 'within_scene',
                    evidence_clip=window['clip'], evidence_start_frame=window['start_frame'],
                    evidence_end_frame=window['end_frame'], annotation_source='proposal'))
            if owned['state'] == 'player':
                if last_owner is not None and last_owner['id'] != owned['owner_id'] and now-last_owner['time'] <= 2.:
                    emit('control_change', 'different_track_control_hypotheses; teams_and_identity_unverified',
                         last_owner['id'], owned['owner_id'])
                last_owner = dict(id=owned['owner_id'], time=now)
            if ball is not None:
                if last_ball is not None:
                    gap = now-last_ball['time']
                    if 0 < gap <= config['max_ball_gap_s'] and ball['ball_id'] == last_ball['ball_id']:
                        speed = math.hypot(ball['x']-last_ball['x'], ball['y']-last_ball['y'])/gap/diagonal
                        if speed >= config['motion_diagonals_per_s'] and now-last_shot >= config['shot_separation_s']:
                            emit('shot_candidate', 'rapid_image_motion; goal_context_unknown; not_a_shot_decision',
                                 score=speed)
                            last_shot = now
                last_ball = dict(ball, time=now)
            elif last_ball and now-last_ball['time'] > config['max_ball_gap_s']:
                last_ball = None
        for scene in source['scenes']:
            exclusions.append(dict(source_id=source['source_id'], scene_id=scene['scene_id'],
                start_frame=scene['start_frame'], end_frame=min(scene['end_frame'],
                scene['start_frame']+round(config['context_before_s']*source['fps'])),
                reason='cut_or_recording_start_context_unavailable'))
            exclusions.append(dict(source_id=source['source_id'], scene_id=scene['scene_id'],
                start_frame=max(scene['start_frame'], scene['end_frame']-round(config['context_after_s']*source['fps'])),
                end_frame=scene['end_frame'], reason='cut_or_recording_end_context_unavailable'))
    write_rows(directory/'possession_candidates.csv',
               ['source_id','scene_id','frame','time_s','state','owner_id','evidence','annotation_source'], states)
    fields = ['candidate_id','window_id','source_id','scene_id','frame','time_s','event_type','actor_id',
              'recipient_id','team_id','reason','diagnostic_score','status','context_status','evidence_clip',
              'evidence_start_frame','evidence_end_frame','annotation_source']
    write_rows(directory/'event_candidates.csv', fields, candidates)
    write_rows(directory/'unevaluable_intervals.csv',
               ['source_id','scene_id','start_frame','end_frame','reason'], exclusions)
    report = dict(parameters=config, candidates=len(candidates), per_type=dict(Counter(
        r['event_type'] for r in candidates)), human_confirmed=0,
        note='All outputs are unvalidated hypotheses. No pixel speed is a physical ball speed.',
        cuts_reset_state=True, team_classification=False,
        random_audit_windows=[w['window_id'] for w in dataset['windows'] if w['selection'] == 'random_audit'])
    (directory/'event_generation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='results/review_v04')
    parser.add_argument('--config', help='Optional JSON of unvalidated proposal thresholds')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else None
    print(json.dumps(generate(args.dataset, config), indent=2))
