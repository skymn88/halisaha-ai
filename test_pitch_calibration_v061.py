"""v0.6.1: human pitch calibration with a held-out check (reviewed_geometry.validate_pitch_calibration)."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from reviewed_geometry import project_to_pitch, resolve_geometry, validate_geometry, validate_pitch_calibration

SIZE = [20.0, 40.0]  # metres, width x length (synthetic; not a real pitch)
# Synthetic camera: a known homography from pitch metres to a 640x360 image.
H = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [20, 0], [20, 40], [0, 40]]), np.float32([[100, 320], [540, 320], [450, 120], [190, 120]]))


def to_image(point):
    return cv2.perspectiveTransform(np.float32([[point]]), H)[0][0].tolist()


def pair(point, shift=(0., 0.)):
    image = to_image(point)
    return dict(image=[image[0]+shift[0], image[1]+shift[1]], pitch_m=list(point))


def calibration(**changes):
    value = dict(pitch_size_m=SIZE,
                 fit_points=[pair(p) for p in ([0, 0], [20, 0], [20, 40], [0, 40])],
                 holdout_points=[pair([10, 20]), pair([5, 30])],
                 max_holdout_error_m=0.5)
    value.update(changes)
    return value


def record(**changes):
    base = dict(schema_version=2, scene='scene_000', video_sha256='abc', image_size=[640, 360],
                annotation_source='human', review_status='reviewed', reviewer='tester',
                goals=[dict(polygon=[[10, 50], [10, 10], [50, 10], [50, 50]], mouth_line=[[10, 50], [50, 50]])],
                endlines=[], pitch_calibration=calibration())
    base.update(changes)
    return base


class PitchCalibrationTests(unittest.TestCase):
    def test_exact_synthetic_camera_validates_and_projects(self):
        result = validate_pitch_calibration(record())
        self.assertEqual(result['status'], 'validated')
        self.assertTrue(result['metric_enabled'])
        self.assertLess(result['holdout_max_error_m'], 0.05)
        x, y = project_to_pitch(result, to_image([8, 12]))
        self.assertAlmostEqual(x, 8, places=2)
        self.assertAlmostEqual(y, 12, places=2)

    def test_bad_holdout_disables_metrics_but_keeps_record_usable(self):
        # A holdout point 40 px off the true projection is metres of error: calibration must not be trusted.
        bad = calibration(holdout_points=[pair([10, 20], shift=(40., 0.)), pair([5, 30])])
        result = validate_pitch_calibration(record(pitch_calibration=bad))
        self.assertEqual(result['status'], 'failed_holdout')
        self.assertFalse(result['metric_enabled'])
        self.assertIsNone(project_to_pitch(result, to_image([8, 12])))
        r = record(pitch_calibration=bad)
        self.assertIs(validate_geometry(r, 'scene_000', 'abc'), r)  # goals stay usable

    def test_missing_calibration_returns_none(self):
        r = record(); del r['pitch_calibration']
        self.assertIsNone(validate_pitch_calibration(r))
        self.assertIsNone(project_to_pitch(None, [1, 1]))

    def test_malformed_input_is_rejected(self):
        cases = {
            'no declared limit': calibration(max_holdout_error_m=None),
            'too few fit points': calibration(fit_points=calibration()['fit_points'][:3]),
            'too few holdout points': calibration(holdout_points=calibration()['holdout_points'][:1]),
            'nonpositive size': calibration(pitch_size_m=[0, 40]),
            'pitch point outside size': calibration(holdout_points=[dict(image=[300, 200], pitch_m=[25, 10]),
                                                                    pair([5, 30])]),
            'image point outside frame': calibration(holdout_points=[dict(image=[900, 200], pitch_m=[10, 10]),
                                                                     pair([5, 30])]),
            'nan point': calibration(holdout_points=[dict(image=[float('nan'), 200], pitch_m=[10, 10]),
                                                     pair([5, 30])]),
            'holdout reuses fit point': calibration(holdout_points=[calibration()['fit_points'][0], pair([5, 30])]),
        }
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_pitch_calibration(record(pitch_calibration=value))

    def test_collinear_fit_points_are_rejected(self):
        line = [pair([0, 0]), pair([5, 0]), pair([10, 0]), pair([20, 0])]
        with self.assertRaises(ValueError):
            validate_pitch_calibration(record(pitch_calibration=calibration(fit_points=line)))

    def test_malformed_calibration_rejects_whole_geometry(self):
        with self.assertRaises(ValueError):
            validate_geometry(record(pitch_calibration=calibration(max_holdout_error_m=-1)), 'scene_000', 'abc')

    def test_resolve_attaches_result_and_requires_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'scene_000.mp4'; video.write_bytes(b'synthetic')
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            (root / 'geometry').mkdir()
            (root / 'geometry' / 'scene_000.json').write_text(json.dumps(record(video_sha256=digest)))
            geometry = resolve_geometry(video, root / 'geometry')
            self.assertEqual(geometry['geometry_source'], 'human_reviewed')
            self.assertTrue(geometry['pitch_calibration_result']['metric_enabled'])
            (root / 'geometry' / 'scene_000.json').write_text(json.dumps(record(video_sha256='wrong')))
            with self.assertRaises(ValueError):
                resolve_geometry(video, root / 'geometry')

    def test_unreviewed_geometry_never_carries_calibration(self):
        r = record(annotation_source='ai_reviewed')
        with self.assertRaises(ValueError):
            validate_geometry(copy.deepcopy(r), 'scene_000', 'abc')


if __name__ == '__main__':
    unittest.main()
