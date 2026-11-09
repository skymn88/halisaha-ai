"""Scene-scoped event hypotheses from observed alternative ball trajectories."""
import math
import cv2
import numpy as np
from event_candidates import Possession
from scene_analytics import crossing


def distance(a, b): return math.hypot(a['x']-b['x'], a['y']-b['y'])


def directed_to_mouth(a, b, line, margin):
    start, end = np.asarray(line[0], float), np.asarray(line[1], float)
    unit = (end-start)/max(1., np.linalg.norm(end-start))
    origin = np.array([a['x'], a['y']]); direction = np.array([b['x']-a['x'], b['y']-a['y']])
    length = np.linalg.norm(direction)
    if length < 1e-6: return False
    return crossing(origin, origin+direction/length*10000., start-margin*unit, end+margin*unit)


def transfer_kind(actor, recipient, teams):
    a,b = teams.get(str(actor)),teams.get(str(recipient))
    if a in (None,'unknown') or b in (None,'unknown'): return 'control_change_unknown_team'
    return 'completed_pass_candidate' if a == b else 'ball_recovery_candidate'


def generate_events(observations, people, geometry, fps, config, teams=None):
    """No predictions become observations; weak gaps never produce strong-tier shots."""
    tracks = {}
    for p in observations: tracks.setdefault(p['trajectory_id'], []).append(p)
    events = []
    for tid, points in tracks.items():
        points.sort(key=lambda p:p['frame'])
        possession = Possession({'enter_s':config['control_enter_s'], 'control_radius_heights':config['control_radius_heights']})
        owned = None; release = None; previous = None; flight = []; observed_buffer = []
        def emit(kind, tier, p, reasons, **extra):
            events.append(dict(type=kind, tier=tier, frame=p['frame'], time_s=p['frame']/fps,
                               trajectory_id=tid, source='proposal', review_status='unreviewed',
                               geometry_source=geometry['geometry_source'], reasons=reasons, **extra))
        for p in points:
            now = p['frame']/fps
            observed_buffer = [q for q in observed_buffer if now-q['frame']/fps <= config['event_window_s']] + [p]
            ps = people.get(p['frame'], [])
            pinput = [dict(track_id=int(r['track_id']),confidence=float(r['confidence']),box=[float(r[k]) for k in ('x1','y1','x2','y2')]) for r in ps]
            state = possession.update(now, dict(x=p['x'],y=p['y'],confidence=.15), pinput)
            if owned and now-owned['time_s'] > config['event_window_s']: owned=None;release=None;flight=[]
            if release and now-release['time_s'] > config['event_window_s']: release=None;owned=None;flight=[]
            if owned and not release and state['state']=='free_ball' and distance(owned['point'],p)/owned['height'] > config['control_radius_heights']:
                release=dict(owned, point=p, time_s=now)
                flight=[q for q in observed_buffer if q['frame'] >= owned['point']['frame']]
            elif release: flight.append(p)
            entry=False;line_cross=False;direction=False
            if previous:
                line_cross=any(crossing((previous['x'],previous['y']),(p['x'],p['y']),*line) for line in geometry['endlines'])
                for goal in geometry['goals']:
                    poly=np.array(goal['polygon'],np.float32)
                    entry |= cv2.pointPolygonTest(poly,(previous['x'],previous['y']),False)<0 <= cv2.pointPolygonTest(poly,(p['x'],p['y']),False)
                if line_cross or entry:
                    emit('goal_entry_candidate' if entry else 'endline_crossing', 'B', p,
                         ['geometry_intersection_hypothesis','not_a_goal','interpolated_evidence' if p['frame']-previous['frame']>1 else 'consecutive_observations'])
            if release and len(flight)>=2:
                start=flight[0]; height=release['height']
                gaps=any(b['frame']-a['frame']>1 for a,b in zip(flight,flight[1:]))
                length=sum(distance(a,b) for a,b in zip(flight,flight[1:])); travel=distance(start,p)/height
                straightness=distance(start,p)/max(1e-6,length)
                speed=distance(start,p)/max(1e-6,(p['frame']-start['frame'])/fps)/height
                direction=any(directed_to_mouth(start,p,g['mouth_line'],config['goal_margin_heights']*height) for g in geometry['goals'])
                near_goal=any(math.dist((start['x'],start['y']),np.mean(g['mouth_line'],axis=0))/height <= config['goal_half_radius_heights'] for g in geometry['goals'])
                strong=not gaps and speed>=config['fast_heights_per_s'] and direction and straightness>=config['direction_consistency'] and (entry or line_cross)
                weak=gaps and near_goal and (entry or line_cross or state['state']=='player')
                if strong or weak:
                    # Recontrol near a goal is not a verified attacking half or a shot.
                    kind='shot_candidate' if strong or (direction and (entry or line_cross)) else 'release_recontrol_review'
                    emit(kind, 'A' if strong else 'B', p,
                         ['release','goal_direction','fast_normalized_motion','observed_geometry_intersection'] if strong else ['release','near_goal_proxy_not_calibrated_half','flight_not_fully_observed','human_review_required'],
                         actor=release['actor'], evidence='interpolated_evidence' if gaps else 'observed', travel_heights=travel)
                    release=None;flight=[]
                elif not geometry['goals'] and travel>=config['travel_heights'] and speed>=config['fast_heights_per_s'] and straightness>=config['direction_consistency']:
                    # No geometric evidence: deliberately not included as a shot in proxy coverage.
                    emit('fast_release_review', 'B', p, ['release','goal_context_unknown','could_be_pass_or_clearance'], actor=release['actor'])
                    release=None;flight=[]
                if release and state['state']=='player' and state['owner_id']!=release['actor']:
                    if travel>=config['travel_heights'] and straightness>=config['direction_consistency']:
                        emit(transfer_kind(release['actor'],state['owner_id'],teams or {}), 'B' if gaps else 'A', p,
                             ['release','new_control','team_known' if teams else 'team_unknown','interpolated_evidence' if gaps else 'observed'],
                             actor=release['actor'],recipient=state['owner_id'],recovery_type='unknown_contact')
                    release=None;flight=[]
            if state['state']=='player':
                owner=state['owner_id']; person=next((r for r in ps if int(r['track_id'])==owner),None)
                if person:
                    owned=dict(actor=owner,time_s=now,point=p,height=max(1.,float(person['y2'])-float(person['y1'])))
            previous=p
    # Deduplicate alternative hypotheses without converting them to confirmed counts.
    result=[]
    for event in sorted(events,key=lambda e:(e['frame'],e['tier'])):
        if any(e['type']==event['type'] and abs(e['time_s']-event['time_s'])<config['event_cooldown_s'] for e in result):continue
        result.append(event)
    return result
