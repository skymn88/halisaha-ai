import unittest
import cv2
import numpy as np
from schematic_heatmap import visible_quad,render

class SchematicTests(unittest.TestCase):
    def test_quad_preserves_order_and_maps_corners(self):
        q=visible_quad([[10,10],[150,10],[150,90],[10,90]])
        expected=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
        h=cv2.getPerspectiveTransform(q,expected)
        mapped=cv2.perspectiveTransform(q.reshape(-1,1,2),h).reshape(-1,2)
        np.testing.assert_allclose(mapped,expected,atol=1e-6)
    def test_degenerate_geometry_rejected(self):
        with self.assertRaises(ValueError):visible_quad([[0,0],[1,1],[2,2],[3,3]])
    def test_empty_density_does_not_create_heat(self):
        im=render(np.zeros((50,90),np.float32),0)
        self.assertEqual(im.shape,(610,1010,3))
        self.assertTrue(np.isfinite(im).all())
    def test_hotspot_changes_schematic_without_video_background(self):
        empty=render(np.zeros((50,90),np.float32),1)
        grid=np.zeros((50,90),np.float32);grid[15:20,25:30]=1
        hot=render(grid,1)
        self.assertGreater(np.abs(hot.astype(float)-empty).sum(),10000)
