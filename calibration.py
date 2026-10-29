"""Optional per-scene geometry; metric coordinates require supplied measurements."""
import json
from pathlib import Path
import cv2
import numpy as np


class SceneGeometry:
    def __init__(self, config):
        polygon = config.get('pitch_polygon', [])
        self.polygon = np.asarray(polygon, dtype=np.float32).reshape(-1,2)
        if len(self.polygon) and (len(self.polygon) < 3 or not np.isfinite(self.polygon).all() or abs(cv2.contourArea(self.polygon)) < 1):
            raise ValueError('pitch_polygon must contain at least three finite non-collinear image points')
        self.homography = None
        self.reprojection_rmse_px = None
        image = np.asarray(config.get('image_points', []),dtype=np.float64).reshape(-1,2)
        pitch = np.asarray(config.get('pitch_points_m', []),dtype=np.float64).reshape(-1,2)
        if len(image) or len(pitch):
            if len(image) != len(pitch) or len(image) < 4 or not np.isfinite(image).all() or not np.isfinite(pitch).all():
                raise ValueError('Calibration needs at least four paired finite image_points and pitch_points_m')
            if len(np.unique(image,axis=0)) != len(image) or len(np.unique(pitch,axis=0)) != len(pitch):
                raise ValueError('Calibration points must be distinct')
            if cv2.contourArea(cv2.convexHull(image.astype(np.float32))) < 1 or cv2.contourArea(cv2.convexHull(pitch.astype(np.float32))) < .001:
                raise ValueError('Calibration points must not be collinear')
            h, _ = cv2.findHomography(image,pitch,method=0)
            if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
                raise ValueError('Invalid homography')
            inverse = np.linalg.inv(h)
            reprojected = cv2.perspectiveTransform(pitch[None].astype(np.float64),inverse)[0]
            self.reprojection_rmse_px = float(np.sqrt(np.mean(np.sum((reprojected-image)**2,axis=1))))
            limit = float(config.get('max_reprojection_rmse_px',5.))
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError('max_reprojection_rmse_px must be finite and positive')
            if not np.isfinite(self.reprojection_rmse_px) or self.reprojection_rmse_px > limit:
                raise ValueError('Calibration reprojection error exceeds configured limit')
            self.homography = h

    def contains(self, point):
        return not len(self.polygon) or cv2.pointPolygonTest(self.polygon,tuple(map(float,point)),False) >= 0

    def project(self, point):
        if self.homography is None:
            return ['', '']
        vector = self.homography @ np.array([*point,1.],dtype=float)
        if abs(vector[2]) < 1e-9:
            return ['', '']
        return [round(float(x),4) for x in vector[:2]/vector[2]]


def load_geometries(path, image_size, scene_ids):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('image_size') != list(image_size):
        raise ValueError('Camera configuration image_size must match source video [width, height]')
    result = {}
    for key, config in data.get('scenes',{}).items():
        scene_id = int(key)
        if scene_id not in scene_ids:
            raise ValueError(f'Configured scene {scene_id} is absent from detected scenes')
        result[scene_id] = SceneGeometry(config)
    return result
