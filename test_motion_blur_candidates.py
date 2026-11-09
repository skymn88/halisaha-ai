import unittest
import cv2
import numpy as np
from motion_blur_candidates import MotionBlurCandidates

class BlurCandidateTests(unittest.TestCase):
    def test_stationary_background_produces_no_candidates(self):
        im=np.full((60,100,3),40,np.uint8);cv2.circle(im,(30,30),3,(180,180,180),-1)
        d=MotionBlurCandidates(im)
        for _ in range(5):self.assertEqual(d.update(im),[])
    def test_dim_moving_highlight_is_only_a_proposal(self):
        bg=np.full((60,100,3),40,np.uint8);d=MotionBlurCandidates(bg);d.update(bg)
        im=bg.copy();cv2.rectangle(im,(30,30),(35,32),(110,110,110),-1)
        rows=d.update(im);self.assertTrue(rows)
        self.assertTrue(all(r['source']=='motion_blur' and r['support']==1 for r in rows))
