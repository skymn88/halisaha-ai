"""Fixed-camera person tracking adapter; accepts records from any detector."""
from types import SimpleNamespace
import numpy as np


class DetectionArray:
    def __init__(self, data):
        self.data = np.asarray(data,dtype=np.float32).reshape(-1,6)
        self.xyxy = self.data[:,:4]
        self.conf = self.data[:,4]
        self.cls = self.data[:,5]
        self.xywh = self.xyxy.copy()
        self.xywh[:,:2] = (self.xyxy[:,:2]+self.xyxy[:,2:])/2
        self.xywh[:,2:] = self.xyxy[:,2:]-self.xyxy[:,:2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return DetectionArray(self.data[index])


def box_iou(a,b):
    a,b = np.asarray(a),np.asarray(b)
    intersection = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union = np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
    return float(intersection/max(union,1e-9))


class PlayerTracker:
    def __init__(self, config, display_conf=.25, hold_frames=3):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker
        config = dict(config)
        if config.get('tracker_type') not in ('botsort','bytetrack'):
            raise ValueError('Only botsort and bytetrack backends are supported by this adapter')
        if config.get('with_reid',False) or config.get('gmc_method','none')!='none':
            raise ValueError('CSV replay adapter requires with_reid=false and gmc_method=none')
        backend = BOTSORT if config['tracker_type']=='botsort' else BYTETracker
        self.backend = backend(SimpleNamespace(**config))
        self.display_conf, self.hold_frames = display_conf,hold_frames
        self.last_seen = {}
        self.last_diagnostics = {}

    def update(self, frame, detections, geometry=None):
        raw = [d for d in detections if d.is_person]
        kept = [d for d in raw if geometry is None or geometry.contains(d.foot)]
        array = DetectionArray([[*d.box,d.confidence,0] for d in kept])
        tracks = self.backend.update(array)
        observed,used = [],set()
        for track in tracks:
            x1,y1,x2,y2,tid,score,_,index = track
            tid,index = int(tid),int(index)
            used.add(index)
            observed.append(dict(box=(float(x1),float(y1),float(x2),float(y2)),confidence=float(score),
                                 track_id=tid,state='observed',source_index=index))
            self.last_seen[tid] = frame
        for index,d in enumerate(kept):
            if index not in used and d.confidence>=self.display_conf:
                observed.append(dict(box=d.box,confidence=d.confidence,track_id=-1,
                                     state='detected_unassigned',source_index=index))
        predictions = []
        for track in self.backend.lost_stracks:
            age = frame-self.last_seen.get(track.track_id,-1000000)
            box = tuple(float(x) for x in track.xyxy)
            if 0<age<=self.hold_frames and not any(box_iou(box,o['box'])>.3 for o in observed):
                predictions.append(dict(box=box,track_id=int(track.track_id),state='predicted',age_frames=age))
        self.last_seen = {tid:f for tid,f in self.last_seen.items() if frame-f<=max(self.hold_frames,300)}
        self.last_diagnostics = dict(raw_persons=len(raw),outside_pitch=len(raw)-len(kept),
                                     associated=sum(o['state']=='observed' for o in observed),
                                     low_conf_associated=sum(o['state']=='observed' and o['confidence']<self.display_conf for o in observed),
                                     unassigned=sum(o['state']=='detected_unassigned' for o in observed),predicted=len(predictions))
        return observed,predictions
