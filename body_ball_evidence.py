"""Appearance penalties and relative motion evidence, not semantic ball verification."""
import math
import cv2
import numpy as np


def appearance(candidate, image, people, config):
    c = dict(candidate)
    nearest = []
    for p in people:
        x1, y1, x2, y2 = [float(p[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        height = max(1., y2-y1)
        nearest.append((math.hypot(c['x']-(x1+x2)/2, c['y']-y2)/height, p, height))
    c.update(person_id=None, person_height=40., inside_person=False, foot_zone=False, skin_fraction=0., roundness=None)
    if nearest:
        distance, p, height = min(nearest, key=lambda r: r[0])
        inside = float(p['x1']) <= c['x'] <= float(p['x2']) and float(p['y1']) <= c['y'] <= float(p['y2'])
        c.update(person_id=int(p['track_id']), person_height=height, foot_distance=distance,
                 person_center=[(float(p['x1'])+float(p['x2']))/2, (float(p['y1'])+float(p['y2']))/2],
                 inside_person=inside, foot_zone=inside and c['y'] >= float(p['y1'])+config['foot_zone_fraction']*height)
    x1,y1,x2,y2 = c['box']; h,w = image.shape[:2]
    patch = image[max(0,int(y1)):min(h,math.ceil(y2)), max(0,int(x1)):min(w,math.ceil(x2))]
    if patch.size:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        skin = ((hsv[:,:,0] <= 25) | (hsv[:,:,0] >= 170)) & (hsv[:,:,1] >= 45) & (hsv[:,:,1] <= 210) & (hsv[:,:,2] >= 60)
        c['skin_fraction'] = float(skin.mean())
        mask = ((hsv[:,:,2] > 160) & (hsv[:,:,1] < 100)).astype('uint8')*255
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea); perimeter = cv2.arcLength(contour, True)
            if perimeter: c['roundness'] = min(1., 4*math.pi*cv2.contourArea(contour)/perimeter**2)
    c['size_ratio'] = max(x2-x1,y2-y1)/c['person_height']
    # Tiny/blurred objects have unreliable color and shape; these cues only penalize.
    penalty = .12*int(c['inside_person'] and not c['foot_zone']) + .08*int(c['skin_fraction'] > config['skin_fraction_penalty'])
    penalty += .04*int(c['roundness'] is not None and c['roundness'] < .25)
    c['weight'] = max(.01, c.get('weight', .15)-penalty)
    return c


def body_attachment(points, config):
    recent = points[-max(4, int(config['body_min_observations'])):]
    if len(recent) < config['body_min_observations'] or any(p.get('person_id', -1) is None or p.get('person_id', -1) < 0 or 'person_center' not in p for p in recent):
        return False
    if len({p['person_id'] for p in recent}) != 1: return False
    heights = np.array([p['person_height'] for p in recent])
    ball = np.array([[p['x'],p['y']] for p in recent])
    person = np.array([p['person_center'] for p in recent])
    # Scale-normalized offsets tolerate modest zoom. A stationary person is not evidence of attachment.
    relative = (ball-person)/heights[:,None]
    residual = float(np.max(np.linalg.norm(relative-relative[0], axis=1)))
    bv, pv = np.diff(ball,axis=0).flatten(), np.diff(person,axis=0).flatten()
    denom = np.linalg.norm(bv)*np.linalg.norm(pv)
    correlation = float(np.dot(bv,pv)/denom) if denom > 1e-6 else 0.
    person_travel = np.linalg.norm(person[-1]-person[0])/float(np.median(heights))
    near_body = all(p.get('foot_distance', 99) < 1.3 for p in recent)
    return bool(near_body and person_travel > .04 and residual < config['body_residual_heights'] and correlation > config['body_motion_correlation'])
