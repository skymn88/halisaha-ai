"""Hash-bound geometry accepted only after explicit named human review."""
import hashlib
import json
from pathlib import Path
import math
import cv2
import numpy as np
from calibration import SceneGeometry


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
    validate_pitch_calibration(record, image_size)  # raises on malformed input; result is computed again on resolve
    return record


def _calibration_pairs(items, name, image_size, pitch_size, minimum):
    if not isinstance(items, list) or len(items) < minimum:
        raise ValueError(f'{name} needs at least {minimum} points')
    def vector(value):
        return (isinstance(value, list) and len(value) == 2 and
                all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))
    image, pitch = [], []
    for item in items:
        i, m = (item.get('image'), item.get('pitch_m')) if isinstance(item, dict) else (None, None)
        if not vector(i) or not vector(m):
            raise ValueError(f'{name}: each point needs finite "image" [x, y] and "pitch_m" [X, Y]')
        if not (0 <= i[0] <= image_size[0] and 0 <= i[1] <= image_size[1]):
            raise ValueError(f'{name}: image point outside source image')
        if not (0 <= m[0] <= pitch_size[0] and 0 <= m[1] <= pitch_size[1]):
            raise ValueError(f'{name}: pitch point outside the declared pitch_size_m')
        image.append([float(i[0]), float(i[1])]); pitch.append([float(m[0]), float(m[1])])
    return image, pitch


def validate_pitch_calibration(record, image_size=(640, 360)):
    """Optional human metric calibration with a held-out check; None when the record has none.

    Added in v0.6.1 because the existing reprojection error is measured on the fitting points and cannot
    validate a calibration. Contract (all supplied by a named human reviewer, never invented here):
      pitch_calibration = {
        pitch_size_m: [width, length]            # metres; x in [0, width], y in [0, length]
        fit_points:     [{image:[x,y], pitch_m:[X,Y]}, ...]   # >= 4, not collinear
        holdout_points: [{image:[x,y], pitch_m:[X,Y]}, ...]   # >= 2, NOT used for the fit
        max_holdout_error_m: <positive metres>  # the reviewer's declared acceptance limit
      }
    Malformed input raises ValueError. A well-formed calibration whose held-out error exceeds the declared
    limit is returned with status 'failed_holdout' and metric_enabled False (goals/end lines stay usable).
    A planar homography says nothing about an airborne ball's height; lens distortion is not corrected.
    """
    calibration = record.get('pitch_calibration')
    if calibration is None:
        return None
    if not isinstance(calibration, dict):
        raise ValueError('pitch_calibration must be an object')
    size = calibration.get('pitch_size_m')
    if (not isinstance(size, list) or len(size) != 2 or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in size)):
        raise ValueError('pitch_size_m must be [width, length] in positive finite metres')
    limit = calibration.get('max_holdout_error_m')
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
        raise ValueError('max_holdout_error_m must be declared by the reviewer (positive metres)')
    fit_image, fit_pitch = _calibration_pairs(calibration.get('fit_points'), 'fit_points', image_size, size, 4)
    hold_image, hold_pitch = _calibration_pairs(calibration.get('holdout_points'), 'holdout_points', image_size, size, 2)
    if {tuple(p) for p in fit_image} & {tuple(p) for p in hold_image}:
        raise ValueError('Held-out points must be different from the fitting points')
    # Reuses the existing fit; raises for duplicate/collinear points or an invalid homography.
    geometry = SceneGeometry(dict(image_points=fit_image, pitch_points_m=fit_pitch, pitch_size_m=size))
    errors = []
    for image, pitch in zip(hold_image, hold_pitch):
        projected = geometry.project(image)
        errors.append(math.inf if projected[0] == '' else math.dist(projected, pitch))
    maximum = max(errors)
    status = 'validated' if maximum <= limit else 'failed_holdout'
    return dict(status=status, metric_enabled=status == 'validated', source='human_reviewed',
                pitch_size_m=[float(v) for v in size], homography=geometry.homography.tolist(),
                fit_points=len(fit_image), fit_rmse_px=geometry.reprojection_rmse_px,
                holdout_points=len(errors), holdout_max_error_m=maximum,
                holdout_rmse_m=math.sqrt(sum(e*e for e in errors)/len(errors)) if math.isfinite(maximum) else math.inf,
                max_holdout_error_m=float(limit))


def project_to_pitch(result, point):
    """Image point -> pitch metres, only for a held-out-validated ground-plane point; else None."""
    if not result or not result.get('metric_enabled'):
        return None
    vector = np.asarray(result['homography'], float) @ np.array([point[0], point[1], 1.])
    if abs(vector[2]) < 1e-9:
        return None
    return [float(vector[0]/vector[2]), float(vector[1]/vector[2])]


def resolve_geometry(video, human_dir=None, allow_ai=False):
    video = Path(video)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    if human_dir and (path := Path(human_dir) / (video.stem + '.json')).exists():
        record = validate_geometry(json.loads(path.read_text()), video.stem, digest)
        return dict(record, geometry_source='human_reviewed',
                    pitch_calibration_result=validate_pitch_calibration(record, (640, 360)))
    if allow_ai:
        record = json.loads(Path('configs/pilot_goal_regions.json').read_text())['scenes'].get(video.stem)
        if record and record['video_sha256'] == digest:
            return dict(record, goals=[g for g in record['goals'] if not g.get('partial')],
                        geometry_source='ai_reviewed_exploratory')
    return dict(goals=[], endlines=[], video_sha256=digest, geometry_source='unavailable')
