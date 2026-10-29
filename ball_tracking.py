"""Single observed-ball selection with reviewable pitch/background evidence."""
import numpy as np


class PitchBallTracker:
    def __init__(self, hits=3, gap=8, radius=45.):
        self.hits, self.gap, self.radius = hits, gap, radius
        self.active = None
        self.next_id = 1
        self.last_diagnostics = []
        self.last_state = 'missing'

    def update(self, frame, candidates, image, background, geometry=None):
        if self.active is not None and frame-self.active['frame'] > self.gap:
            self.active = None
        scored = []
        self.last_diagnostics = []
        for box, confidence in candidates:
            center = (np.asarray(box[:2])+np.asarray(box[2:]))/2
            evidence = background.evidence(image, box)
            status = 'candidate'
            if geometry is not None and not geometry.contains(center):
                status = 'outside_pitch'
            elif evidence['static_marking']:
                status = 'static_marking'
            distance = None
            if self.active is not None:
                dt = frame-self.active['frame']
                prediction = self.active['center']+self.active['velocity']*min(dt,3)
                distance = float(np.linalg.norm(center-prediction))
                # Bound the motion gate; gaps do not permit arbitrary jumps.
                if status == 'candidate' and distance > self.radius*(1+.25*min(dt-1,4)):
                    status = 'motion_gate'
            row = dict(box=box, confidence=confidence, status=status, ball_id=-1,
                       innovation_px=distance, **evidence)
            self.last_diagnostics.append(row)
            if status == 'candidate':
                rank = confidence+.25*evidence['foreground_fraction']
                if distance is not None:
                    rank -= .35*min(distance/self.radius,2)
                scored.append((rank,len(self.last_diagnostics)-1,center))
        if not scored:
            self.last_state = 'occluded' if self.active is not None else 'missing'
            return []
        _, chosen, center = max(scored,key=lambda x:x[0])
        for _,idx,_ in scored:
            if idx != chosen:
                self.last_diagnostics[idx]['status'] = 'alternative'
        row = self.last_diagnostics[chosen]
        if self.active is None:
            self.active = dict(id=self.next_id, center=center, frame=frame,
                               velocity=np.zeros(2), hits=1)
            self.next_id += 1
        else:
            dt = frame-self.active['frame']
            displacement = (center-self.active['center'])/dt
            # Stop prediction drift when the observed ball stops.
            self.active['velocity'] = displacement if np.linalg.norm(displacement) > 1 else np.zeros(2)
            self.active['center'] = center
            self.active['frame'] = frame
            self.active['hits'] += 1
        row['ball_id'] = self.active['id']
        confirmed = self.active['hits'] >= self.hits
        row['status'] = 'accepted' if confirmed else 'temporal_unconfirmed'
        self.last_state = 'observed' if confirmed else 'tentative'
        return [(row['box'],row['confidence'],row['ball_id'],'accepted',False)] if confirmed else []
