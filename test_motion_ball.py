import unittest
import cv2
import numpy as np
from motion_ball import WhiteMotionCandidates

class MotionTests(unittest.TestCase):
    def test_moving_small_white_region_and_stationary_continuation(self):
        d=WhiteMotionCandidates();bg=np.full((60,100),40,np.uint8)
        rows=[]
        for f,x in enumerate([10,13,16,16,16]):
            im=np.full((60,100,3),40,np.uint8);cv2.circle(im,(x,30),2,(240,240,240),-1)
            rows=d.update(im,bg,f)
        self.assertTrue(any(r['support']>=3 for r in rows))
    def test_static_background_white_marking_is_not_proposed(self):
        im=np.full((60,100,3),40,np.uint8);cv2.circle(im,(30,30),3,(240,240,240),-1)
        d=WhiteMotionCandidates();bg=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        for f in range(5):self.assertEqual(d.update(im,bg,f),[])
