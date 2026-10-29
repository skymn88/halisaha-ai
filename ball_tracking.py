"""Competing ball hypotheses with explicit confirmation and bounded reacquisition."""
import numpy as np


class PitchBallTracker:
    def __init__(self, hits=3, gap=8, radius=45., reacquire_hits=3):
        self.hits,self.gap,self.radius = hits,gap,radius
        self.reacquire_hits = max(3,reacquire_hits)
        self.active = None
        self.hypotheses = []
        self.next_id = 1
        self.last_diagnostics = []
        self.last_state = 'missing'
        self.last_selected = None

    def update(self, frame, candidates, image, background, geometry=None, people=()):
        self.hypotheses = [t for t in self.hypotheses if frame-t['frame']<=self.gap]
        previous = self.active
        if previous is not None and frame-previous['frame']>self.gap:
            previous = None
        self.active = previous
        self.last_diagnostics,self.last_selected = [],None
        used,ranked = set(),[]
        for box,confidence in sorted(candidates,key=lambda c:-c[1]):
            center = (np.asarray(box[:2])+np.asarray(box[2:]))/2
            evidence = background.evidence(image,box)
            status = 'candidate'
            if geometry is not None and not geometry.contains(center):
                status = 'outside_pitch'
            elif evidence['static_marking']:
                status = 'static_marking'
            distances = []
            for t in self.hypotheses:
                dt = frame-t['frame']
                prediction = t['center']+t['velocity']*min(dt,3)
                distance = float(min(np.linalg.norm(center-prediction),np.linalg.norm(center-t['center'])))
                gate = self.radius*(1+.25*min(max(dt-1,0),4))
                if t['id'] not in used and distance<=gate:
                    distances.append((distance,t))
            innovation = min((d for d,_ in distances),default=None)
            row = dict(box=box,confidence=confidence,status=status,ball_id=-1,innovation_px=innovation,**evidence)
            self.last_diagnostics.append(row)
            if status!='candidate':
                continue
            if distances:
                _,t = min(distances,key=lambda item:item[0])
                dt = frame-t['frame']
                velocity = (center-t['center'])/dt
                speed = np.linalg.norm(velocity)
                if speed>self.radius:
                    velocity *= self.radius/speed
                t['velocity'] = .5*t['velocity']+.5*velocity if speed>1 else np.zeros(2)
                t['center'],t['frame'],t['box'],t['confidence'] = center,frame,box,confidence
                t['hits'] += 1
            else:
                t = dict(id=self.next_id,center=center,frame=frame,box=box,velocity=np.zeros(2),
                         hits=1,reacquiring=previous is not None,confidence=confidence)
                self.next_id += 1
                self.hypotheses.append(t)
            used.add(t['id'])
            row['ball_id'] = t['id']
            challenger = previous is not None and previous['id']!=t['id']
            required = max(self.hits,self.reacquire_hits) if t['reacquiring'] or challenger else self.hits
            confirmed = t['hits']>=required
            row['status'] = 'accepted' if confirmed else ('motion_gate' if t['reacquiring'] or challenger else 'temporal_unconfirmed')
            body_overlap = any(p[0]<=center[0]<=p[2] and p[1]<=center[1]<=p[1]+.8*(p[3]-p[1]) for p in people)
            person_overlap = any(p[0]<=center[0]<=p[2] and p[1]<=center[1]<=p[3] for p in people)
            rank = confidence+.35*evidence['foreground_fraction']-(.3 if body_overlap else .15 if person_overlap else 0)
            if previous is not None and previous['id']==t['id']:
                rank += .15
            ranked.append((confirmed,rank,row,t))
        confirmed = [item for item in ranked if item[0]]
        choices = confirmed or ranked
        if not choices:
            self.last_state = 'occluded' if previous is not None else 'missing'
            return []
        _,_,row,selected = max(choices,key=lambda item:item[1])
        self.last_selected = row
        for ready,_,other,t in ranked:
            if other is not row and ready:
                other['status'] = 'alternative'
        if not confirmed and previous is not None and previous['id']!=selected['id']:
            self.last_state = 'occluded'
            return []
        self.active = selected
        if not confirmed:
            self.last_state = 'tentative'
            return []
        row['status'] = 'accepted'
        self.last_state = 'observed'
        return [(row['box'],row['confidence'],selected['id'],'accepted',False)]

    def prediction(self, frame, hold_frames=3):
        if self.active is None or self.active['hits']<self.hits:
            return None
        age = frame-self.active['frame']
        if not 0<age<=min(hold_frames,self.gap):
            return None
        delta = self.active['velocity']*min(age,3)
        box = np.asarray(self.active['box'])+np.tile(delta,2)
        return dict(box=tuple(box),track_id=self.active['id'],state='predicted',age_frames=age)
