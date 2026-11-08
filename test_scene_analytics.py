import unittest
import numpy as np
from auto_pitch import infer_geometry
from scene_analytics import inside_person,crossing

class AnalyticsTests(unittest.TestCase):
    def test_any_person_overlap_removed_without_history(self):
        p=dict(x1=10,y1=10,x2=30,y2=50)
        self.assertTrue(inside_person(20,49,[p]))
        self.assertTrue(inside_person(10,10,[p]))
        self.assertFalse(inside_person(31,49,[p]))
    def test_finite_segment_crossing(self):
        self.assertTrue(crossing((0,0),(10,10),(0,10),(10,0)))
        self.assertFalse(crossing((0,0),(2,2),(0,10),(10,0)))
        self.assertFalse(crossing((0,0),(10,0),(0,1),(10,1)))
    def test_no_pitch_or_goals_on_black_input(self):
        g=infer_geometry(np.zeros((100,160,3),np.uint8))
        self.assertFalse(g['pitch_polygon']);self.assertFalse(g['goal_candidates'])
    def test_green_field_is_only_uncalibrated_proposal(self):
        im=np.zeros((100,160,3),np.uint8);im[20:90,10:150]=(40,110,40)
        g=infer_geometry(im)
        self.assertTrue(g['pitch_polygon']);self.assertFalse(g['metric_calibration'])
        self.assertEqual(g['review_status'],'unreviewed')
