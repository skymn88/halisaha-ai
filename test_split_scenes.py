"""Clip-boundary and no-overwrite regressions for the scene export utility."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from split_scenes import split_video


class SceneExportTests(unittest.TestCase):
    def make_video(self, directory):
        path = Path(directory)/'source.mp4'
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
        self.assertTrue(writer.isOpened())
        for frame in range(60):
            writer.write(np.full((48,64,3),30 if frame<30 else 200,np.uint8))
        writer.release()
        return path

    def test_boundaries_mapping_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            output = Path(directory)/'clips'
            scenes = [dict(scene_id=0,start_frame=0,end_frame=30),dict(scene_id=1,start_frame=30,end_frame=60)]
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = split_video(source,output,scenes)
            self.assertEqual([r['frame_count'] for r in manifest['clips']],[30,30])
            self.assertEqual(manifest['clips'][1]['source_start_frame'],30)
            for scene_id,expected in [(0,30),(1,200)]:
                cap = cv2.VideoCapture(str(output/f'scene_{scene_id:03d}.mp4'))
                try:
                    for frame in [0,29]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                        ok,image = cap.read()
                        self.assertTrue(ok)
                        self.assertLess(abs(float(image.mean())-expected),12)
                finally:
                    cap.release()
            with self.assertRaises(ValueError):
                split_video(source,output,scenes)

    def test_invalid_manifest_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            output = Path(directory)/'bad_clips'
            with self.assertRaises(ValueError):
                split_video(source,output,[dict(scene_id=0,start_frame=1,end_frame=60)])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
