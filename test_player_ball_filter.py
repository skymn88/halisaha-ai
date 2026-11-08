import unittest
from player_ball_filter import TorsoFilter

class TorsoTests(unittest.TestCase):
    def person(self):return dict(track_id='1',confidence='.9',x1=10,y1=10,x2=30,y2=50)
    def candidate(self,f,x=20,y=25):return dict(frame=f,proposal_id=1,center_x=x,center_y=y)
    def test_three_consecutive_frames_required_and_release(self):
        d=TorsoFilter()
        self.assertEqual([d.classify(self.candidate(f),[self.person()])[0] for f in range(3)],['retained','retained','suppressed'])
        self.assertEqual(d.classify(self.candidate(3,x=40),[self.person()])[0],'retained')
    def test_feet_and_missing_frame_not_suppressed(self):
        d=TorsoFilter()
        for f in range(6):self.assertEqual(d.classify(self.candidate(f,y=48),[self.person()])[0],'retained')
        d=TorsoFilter()
        for f in (0,2,4,6):self.assertEqual(d.classify(self.candidate(f),[self.person()])[0],'retained')
    def test_relative_motion_breaks_support(self):
        d=TorsoFilter()
        for f,x in enumerate((11,25,12,26)):
            self.assertEqual(d.classify(self.candidate(f,x=x),[self.person()])[0],'retained')
