"""Behavioral regression tests for pitch-aware filtering and scene calibration."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from ball_tracking import PitchBallTracker
from pitch_context import PitchBackground, scan_scenes
from calibration import SceneGeometry


class PitchBallTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.zeros((100,140,3),np.uint8)
        self.reference[:] = (30,110,30)
        self.reference[35:45,35:45] = 255
        self.background = PitchBackground(self.reference)

    def test_jittering_stationary_spot_rejected(self):
        tracker = PitchBallTracker()
        for frame in range(10):
            x = 35+frame%3
            self.assertEqual(tracker.update(frame,[([x,35,x+8,43],.9)],self.reference,self.background),[])
            self.assertEqual(tracker.last_diagnostics[0]['status'],'static_marking')

    def test_real_ball_can_start_stationary(self):
        tracker = PitchBallTracker()
        image = self.reference.copy()
        image[60:68,70:78] = 240
        rows = []
        for frame in range(6):
            rows = tracker.update(frame,[([70,60,78,68],.7)],image,self.background)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][3],'accepted')

    def test_ball_moves_then_stops_on_marking(self):
        tracker = PitchBallTracker(hits=2)
        # Dark detail differentiates a real ball from a white reference marking.
        for frame,x in enumerate([25,30,35,35,35]):
            image = self.reference.copy()
            image[35:43,x:x+8] = 60
            rows = tracker.update(frame,[([x,35,x+8,43],.8)],image,self.background)
            if frame:
                self.assertEqual(len(rows),1)
        self.assertEqual(tracker.active['velocity'].tolist(),[0.,0.])

    def test_revalidation_blocks_confirmed_false_marking(self):
        tracker = PitchBallTracker(hits=1)
        image = self.reference.copy()
        image[35:43,35:43] = 60
        self.assertTrue(tracker.update(0,[([35,35,43,43],.8)],image,self.background))
        self.assertEqual(tracker.update(1,[([35,35,43,43],.8)],self.reference,self.background),[])
        self.assertEqual(tracker.last_state,'occluded')

    def test_missing_frames_do_not_fabricate_and_expire(self):
        tracker = PitchBallTracker(hits=1,gap=2)
        image = self.reference.copy()
        image[60:68,70:78] = 240
        first = tracker.update(0,[([70,60,78,68],.8)],image,self.background)[0][2]
        self.assertEqual(tracker.update(1,[],image,self.background),[])
        self.assertEqual(tracker.last_state,'occluded')
        self.assertEqual(tracker.update(3,[],image,self.background),[])
        self.assertEqual(tracker.last_state,'missing')
        second = tracker.update(4,[([70,60,78,68],.8)],image,self.background)[0][2]
        self.assertNotEqual(first,second)

    def test_single_ball_and_motion_gate(self):
        tracker = PitchBallTracker(hits=1,radius=12)
        image = self.reference.copy()
        image[60:68,20:28] = 240
        image[60:68,110:118] = 240
        self.assertEqual(len(tracker.update(0,[([20,60,28,68],.8),([110,60,118,68],.6)],image,self.background)),1)
        self.assertEqual(tracker.update(1,[([110,60,118,68],.99)],image,self.background),[])
        self.assertEqual(tracker.last_diagnostics[0]['status'],'motion_gate')


class GeometryTests(unittest.TestCase):
    def test_planar_projection_and_roi(self):
        g = SceneGeometry(dict(pitch_polygon=[[0,0],[100,0],[100,100],[0,100]],
                               image_points=[[0,0],[100,0],[100,100],[0,100]],
                               pitch_points_m=[[0,0],[10,0],[10,20],[0,20]]))
        self.assertEqual(g.project((50,50)),[5.,10.])
        self.assertTrue(g.contains((50,50)))
        self.assertFalse(g.contains((101,50)))
        self.assertLess(g.reprojection_rmse_px,1e-5)

    def test_collinear_calibration_rejected(self):
        with self.assertRaises(ValueError):
            SceneGeometry(dict(image_points=[[0,0],[1,0],[2,0],[3,0]],pitch_points_m=[[0,0],[1,0],[2,0],[3,0]]))

    def test_empty_template_has_no_metric_coordinates(self):
        self.assertEqual(SceneGeometry({}).project((50,50)),['',''])

    def test_hard_cut_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cuts.mp4'
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(80,48))
            self.assertTrue(writer.isOpened())
            for f in range(150):
                writer.write(np.full((48,80,3),30 if f<75 else 200,np.uint8))
            writer.release()
            scenes = scan_scenes(path,window=10,min_frames=30)
            self.assertEqual([s['start_frame'] for s in scenes],[0,75])


if __name__ == '__main__':
    unittest.main()
