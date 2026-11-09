import unittest
from motion_consistency import MotionConsistentBall


def candidate(x,y=20,size=4,weight=.9):
    return dict(x=x,y=y,box=[x-size/2,y-size/2,x+size/2,y+size/2],weight=weight,source='test')

class MotionConsistencyTests(unittest.TestCase):
    def test_static_high_score_never_selected(self):
        t=MotionConsistentBall()
        for f in range(15):self.assertIsNone(t.update(f,[candidate(20)]))
    def test_small_stationary_jitter_not_motion(self):
        t=MotionConsistentBall()
        for f,x in enumerate([20,21,20,19]*4):self.assertIsNone(t.update(f,[candidate(x)]))
    def test_fast_linear_ball_selected_after_real_support(self):
        t=MotionConsistentBall()
        self.assertIsNone(t.update(0,[candidate(10)]));self.assertIsNone(t.update(1,[candidate(35)]))
        b=t.update(2,[candidate(60)]);self.assertIsNotNone(b);self.assertEqual(b['step_px'],25)
    def test_size_jump_cannot_continue_trajectory(self):
        t=MotionConsistentBall()
        t.update(0,[candidate(10,size=3)]);t.update(1,[candidate(15,size=3)])
        self.assertIsNone(t.update(2,[candidate(20,size=10)]))
    def test_expired_history_cannot_bridge_occlusion(self):
        t=MotionConsistentBall();t.update(0,[candidate(10)]);t.update(1,[candidate(15)])
        self.assertIsNone(t.update(8,[candidate(50)]))
    def test_stop_is_not_extrapolated_as_observation(self):
        t=MotionConsistentBall()
        for f,x in enumerate([10,15,20]):t.update(f,[candidate(x)])
        self.assertIsNone(t.update(3,[candidate(20)]))
    def test_equal_competing_tracks_remain_unknown(self):
        t=MotionConsistentBall()
        for f in range(4):self.assertIsNone(t.update(f,[candidate(10+5*f,20),candidate(10+5*f,80)]))
    def test_duplicate_detectors_are_one_observation(self):
        t=MotionConsistentBall()
        for f in range(3):b=t.update(f,[candidate(10+5*f),candidate(11+5*f,weight=.3)])
        self.assertIsNotNone(b)

    def test_invalid_thresholds_rejected(self):
        for kwargs in ({'min_hits':1},{'max_step':float('nan')},{'max_size_ratio':.5},{'max_gap':0}):
            with self.assertRaises(ValueError):MotionConsistentBall(**kwargs)
