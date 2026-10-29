"""Camera-independent pixel-space helpers; no pitch calibration assumptions."""
import cv2
import numpy as np


class TemporalBall:
    def __init__(self, hits=3, gap=8, radius=45., movement=4.):
        self.hits, self.gap, self.radius, self.movement = hits, gap, radius, movement
        self.tracks = []
        self.next_id = 1

    def update(self, frame, candidates, image):
        self.tracks = [t for t in self.tracks if frame-t['frame'] <= self.gap]
        used, output = set(), []
        # Stronger candidates get first choice; one observation per hypothesis/frame.
        for box, confidence in sorted(candidates, key=lambda c: -c[1]):
            center = (np.array(box[:2])+np.array(box[2:]))/2
            available = [t for t in self.tracks if t['id'] not in used and np.linalg.norm(center-t['center']) <= self.radius]
            t = min(available, key=lambda t: np.linalg.norm(center-t['center'])) if available else None
            if t is None:
                t = dict(id=self.next_id, center=center, origin=center.copy(), hits=0,
                         confirmed=False, moved=False, frame=frame)
                self.next_id += 1
                self.tracks.append(t)
            used.add(t['id'])
            t['hits'] += 1
            t['moved'] |= np.linalg.norm(center-t['origin']) >= self.movement
            t['center'], t['frame'] = center, frame
            marking = white_marking(box, image)
            # A confirmed ball remains valid when stationary, including on a line.
            t['confirmed'] |= t['hits'] >= self.hits and (t['moved'] or not marking)
            reason = 'accepted' if t['confirmed'] else ('marking_unconfirmed' if marking else 'temporal_unconfirmed')
            output.append((box, confidence, t['id'], reason, marking))
        return output


def white_marking(box, image):
    """White pixels connected beyond the box suggest a line/spot, not proof."""
    h, w = image.shape[:2]
    x1,y1,x2,y2 = map(int, box)
    pad = max(3, int(max(x2-x1,y2-y1)*0.5))
    a,b,c,d = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    crop = image[b:d,a:c]
    if crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[:,:,1] < 65) & (hsv[:,:,2] > 155)).astype(np.uint8)
    # White presence in both the object and surrounding ring is a marking cue.
    inner = white[max(0,y1-b):max(0,y2-b),max(0,x1-a):max(0,x2-a)]
    ring = white.copy()
    ring[max(0,y1-b):max(0,y2-b),max(0,x1-a):max(0,x2-a)] = 0
    return bool(inner.size and inner.mean() > .25 and ring.sum()/max(1,white.size-inner.size) > .12)


def jersey_feature(image, box):
    """Median torso Lab color for future unsupervised team grouping; no team labels."""
    h,w = image.shape[:2]
    x1,y1,x2,y2 = map(float,box)
    crop = image[max(0,int(y1+.2*(y2-y1))):min(h,int(y1+.55*(y2-y1))),
                 max(0,int(x1+.2*(x2-x1))):min(w,int(x2-.2*(x2-x1)))]
    if not crop.size: return ['', '', '']
    lab = cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    return [round(float(v),2) for v in np.median(lab.reshape(-1,3),axis=0)]


class IDTransitions:
    """Conservative spatial hints only; these are not ground-truth ID switches."""
    def __init__(self, gap=8, radius=35.):
        self.last = {}
        self.seen = set()
        self.gap, self.radius = gap, radius

    def update(self, frame, players):
        current = {tid for tid,box in players if tid >= 0}
        events, used = [], set()
        for tid, box in players:
            if tid < 0: continue
            center = (np.array(box[:2])+np.array(box[2:]))/2
            if tid not in self.seen:
                choices = [(np.linalg.norm(center-pos),old,frame-f) for old,(f,pos) in self.last.items()
                           if old not in current and old not in used and 0 < frame-f <= self.gap]
                if choices:
                    distance,old,gap = min(choices)
                    if distance <= self.radius:
                        events.append([frame,old,tid,gap,round(float(distance),2),'possible_spatial_transition'])
                        used.add(old)
            self.seen.add(tid)
            self.last[tid] = frame,center
        self.last = {tid: state for tid,state in self.last.items() if frame-state[0] <= self.gap}
        return events
