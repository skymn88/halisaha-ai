import unittest
import numpy as np
from auto_pitch import infer_geometry
from scene_analytics import inside_person,crossing,shot_exit

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

    def test_shot_needs_departure_speed_and_actual_crossing(self):
        history=[dict(frame=i,x=10+10*i,y=20) for i in range(3)]
        line=[[25,0],[25,40]]
        self.assertTrue(shot_exit(history,30,('player',0),line))
        self.assertFalse(shot_exit(history,30,None,line))
        self.assertFalse(shot_exit(history,30,('player',0),[[40,0],[40,40]]))
        self.assertFalse(shot_exit(history,1,('player',0),line))
