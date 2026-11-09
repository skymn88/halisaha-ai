"""Hash-bound geometry accepted only after explicit named human review."""
import hashlib
import json
from pathlib import Path
import math
import cv2
import numpy as np


def validate_geometry(record, scene, video_sha256, image_size=(640, 360)):
    if record.get('schema_version') != 2 or record.get('scene') != scene:
        raise ValueError('Geometry schema or scene mismatch')
    if record.get('video_sha256') != video_sha256 or record.get('image_size') != list(image_size):
        raise ValueError('Geometry clip hash or image size mismatch')
    if record.get('annotation_source') != 'human' or record.get('review_status') != 'reviewed' or not str(record.get('reviewer', '')).strip():
        raise ValueError('Geometry needs explicit named human review')
    def point(p):
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
            raise ValueError('Invalid geometry point')
        if not (0 <= p[0] <= image_size[0] and 0 <= p[1] <= image_size[1]):
            raise ValueError('Point outside source image')
    goals = record.get('goals', [])
    lines = record.get('endlines', [])
    if not goals and not lines:
        raise ValueError('Empty reviewed geometry')
    for goal in goals:
        polygon = goal.get('polygon', [])
        if len(polygon) != 4:
            raise ValueError('Goal needs four ordered corners: left foot, left top, right top, right foot')
        for p in polygon: point(p)
        contour = np.asarray(polygon, np.float32)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 5:
            raise ValueError('Goal mouth must be a nondegenerate convex quadrilateral')
        if goal.get('mouth_line') != [polygon[0], polygon[3]]:
            raise ValueError('Goal line must join post feet')
    for line in lines:
        if len(line) != 2: raise ValueError('End line needs two points')
        for p in line: point(p)
        if math.dist(*line) < 5: raise ValueError('End line too short')
    return record


def resolve_geometry(video, human_dir=None, allow_ai=False):
    video = Path(video)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    if human_dir and (path := Path(human_dir) / (video.stem + '.json')).exists():
        record = validate_geometry(json.loads(path.read_text()), video.stem, digest)
        return dict(record, geometry_source='human_reviewed')
    if allow_ai:
        record = json.loads(Path('configs/pilot_goal_regions.json').read_text())['scenes'].get(video.stem)
        if record and record['video_sha256'] == digest:
            return dict(record, goals=[g for g in record['goals'] if not g.get('partial')],
                        geometry_source='ai_reviewed_exploratory')
    return dict(goals=[], endlines=[], video_sha256=digest, geometry_source='unavailable')
