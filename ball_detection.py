"""Periodic tiled acquisition and bounded ROI recovery using the same detector."""
import numpy as np
from player_tracking import box_iou


def tile_starts(length, size, overlap=.2):
    if length<=size:
        return [0]
    step = max(1,int(size*(1-overlap)))
    starts = list(range(0,length-size+1,step))
    if starts[-1]!=length-size:
        starts.append(length-size)
    return starts


def suppress_duplicates(records, threshold=.5):
    result = []
    for record in sorted(records,key=lambda d:-d.confidence):
        if not any(box_iou(record.box,other.box)>threshold for other in result):
            result.append(record)
    return result


class BallSearch:
    def __init__(self, detector, interval=15, tile_fraction=.75, roi_fraction=.35, imgsz=640):
        self.detector = detector
        self.interval,self.tile_fraction,self.roi_fraction,self.imgsz = interval,tile_fraction,roi_fraction,imgsz
        self.last_passes = 0

    def propose(self, image, frame, time_s, scene_id, full_candidates, tracker, background, confidence):
        h,w = image.shape[:2]
        candidates = list(full_candidates)
        self.last_passes = 0
        active = tracker.active
        hypotheses = [t for t in tracker.hypotheses if frame-t['frame']<=tracker.gap]
        hypotheses.sort(key=lambda t:(t is active,t['hits'],t.get('confidence',0)),reverse=True)
        searched_centers = []
        for hypothesis in hypotheses[:2]:
            gap = frame-hypothesis['frame']
            center = hypothesis['center']+hypothesis['velocity']*min(gap,3)
            plausible = any(not background.evidence(image,d.box)['static_marking'] and
                            np.linalg.norm((np.array(d.box[:2])+np.array(d.box[2:]))/2-center)<tracker.radius
                            for d in candidates)
            # Check a challenger even if a different active hypothesis has a plausible observation.
            if hypothesis is active and plausible:
                continue
            size = max(32,int(min(h,w)*self.roi_fraction))
            if any(np.linalg.norm(center-other)<size/2 for other in searched_centers):
                continue
            a = max(0,min(w-size,int(center[0]-size/2)))
            b = max(0,min(h-size,int(center[1]-size/2)))
            candidates.extend(self.detector.detect_ball_crop(image[b:min(h,b+size),a:min(w,a+size)],
                              frame,time_s,scene_id,confidence,self.imgsz,(a,b),'roi'))
            self.last_passes += 1
            searched_centers.append(center)
        if self.interval and frame%self.interval==0:
            size = max(32,int(min(h,w)*self.tile_fraction))
            for b in tile_starts(h,size):
                for a in tile_starts(w,size):
                    candidates.extend(self.detector.detect_ball_crop(image[b:min(h,b+size),a:min(w,a+size)],
                                      frame,time_s,scene_id,confidence,self.imgsz,(a,b),'tile'))
                    self.last_passes += 1
        return suppress_duplicates(candidates)
