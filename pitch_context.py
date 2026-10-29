"""Offline, per-shot background context for fixed-camera football footage."""
from collections import deque
from pathlib import Path
import cv2
import numpy as np


def scan_scenes(video, threshold=23., window=30, min_frames=60):
    """Detect hard cuts and gradual transitions using lagged thumbnail change."""
    cap = cv2.VideoCapture(str(video))
    history = deque(maxlen=window)
    previous = None
    changes, candidates = [], []
    frame = 0
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        while True:
            ok, image = cap.read()
            if not ok:
                break
            small = cv2.resize(image, (80, 45)).astype(np.float32)
            changes.append(float(np.abs(small-previous).mean()) if previous is not None else 0.)
            if len(history) == window:
                score = float(np.abs(small-history[0]).mean())
                if score > threshold:
                    candidates.append((frame, score))
            history.append(small)
            previous = small
            frame += 1
    finally:
        cap.release()
    groups = []
    for item in candidates:
        if not groups or item[0]-groups[-1][-1][0] > window:
            groups.append([])
        groups[-1].append(item)
    boundaries = [0]
    for group in groups:
        start = max(1, group[0][0]-window)
        end = group[-1][0]+1
        hard = start+int(np.argmax(changes[start:end]))
        boundary = hard if changes[hard] > threshold/2 else max(group, key=lambda x: x[1])[0]-window//2
        if boundary-boundaries[-1] >= min_frames and frame-boundary >= min_frames:
            boundaries.append(boundary)
    return [dict(scene_id=i, start_frame=a, end_frame=b) for i,(a,b) in enumerate(zip(boundaries, boundaries[1:]+[frame]))]


class PitchBackground:
    """Median reference is offline context, not annotated ground truth."""
    def __init__(self, background, difference=30., foreground_min=.25):
        self.background = background
        self.difference = difference
        self.foreground_min = foreground_min
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        self.white = ((hsv[:,:,1] < 80) & (hsv[:,:,2] > 140)).astype(np.uint8)

    @classmethod
    def from_video(cls, video, scene, output, samples=41, difference=30., foreground_min=.25):
        cap = cv2.VideoCapture(str(video))
        images = []
        # Avoid transition tails when the scene is long enough.
        margin = min(30, (scene['end_frame']-scene['start_frame'])//5)
        positions = np.unique(np.linspace(scene['start_frame']+margin,
                                         scene['end_frame']-1-margin, samples).astype(int))
        try:
            for frame in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, image = cap.read()
                if ok:
                    images.append(image)
        finally:
            cap.release()
        if not images:
            raise ValueError(f'No background samples for scene {scene["scene_id"]}')
        background = np.median(np.stack(images), axis=0).astype(np.uint8)
        result = cls(background, difference, foreground_min)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output/f'scene_{scene["scene_id"]:03d}_background.png'), background)
        cv2.imwrite(str(output/f'scene_{scene["scene_id"]:03d}_markings.png'), result.white*255)
        return result

    def evidence(self, image, box):
        h,w = image.shape[:2]
        x1,y1,x2,y2 = box
        a,b = max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
        c,d = min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
        if c <= a or d <= b:
            return dict(marking_fraction=0., foreground_fraction=0., static_marking=False)
        reference = self.background[b:d,a:c].astype(np.float32)
        current = image[b:d,a:c].astype(np.float32)
        foreground = float((np.max(np.abs(current-reference),axis=2) > self.difference).mean())
        marking = float(self.white[b:d,a:c].mean())
        return dict(marking_fraction=marking, foreground_fraction=foreground,
                    static_marking=marking >= .25 and foreground < self.foreground_min)
