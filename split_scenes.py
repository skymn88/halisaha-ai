"""Export scene clips without YOLO inference; retain source-frame mapping."""
import argparse
import json
from pathlib import Path
import cv2
from pitch_context import scan_scenes


def split_video(video, output, scenes=None, scene_ids=None):
    video, output = Path(video), Path(output)
    if scenes is None:
        scenes = scan_scenes(video)
    cap = cv2.VideoCapture(str(video))
    records = []
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(3)), int(cap.get(4)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or not scenes:
            raise ValueError('Source must have positive FPS and at least one scene')
        previous_end = 0
        ids = set()
        for scene in scenes:
            scene_id, start, end = (scene[k] for k in ('scene_id','start_frame','end_frame'))
            if any(type(v) is not int for v in (scene_id,start,end)):
                raise ValueError('Scene IDs and frame boundaries must be integers')
            if scene_id in ids or start != previous_end or end <= start or end > frame_count:
                raise ValueError('Scene manifest must contain unique IDs and contiguous, valid boundaries')
            ids.add(scene_id)
            previous_end = end
        if previous_end != frame_count:
            raise ValueError('Scene manifest must cover the complete source video')
        selected = ids if scene_ids is None else set(scene_ids)
        if not selected or not selected <= ids:
            raise ValueError('Requested scene IDs must exist in the manifest')
        destinations = [output/f'scene_{s["scene_id"]:03d}.mp4' for s in scenes if s['scene_id'] in selected]
        if (output/'manifest.json').exists() or any(p.exists() for p in destinations):
            raise ValueError('Output already exists; choose a fresh directory to preserve prior clips')
        output.mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            scene_id, start, end = (scene[k] for k in ('scene_id','start_frame','end_frame'))
            if scene_id not in selected:
                continue
            path = output/f'scene_{scene_id:03d}.mp4'
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
            try:
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot write clip: {path}')
                cap.set(cv2.CAP_PROP_POS_FRAMES,start)
                written = 0
                for frame in range(start,end):
                    ok,image = cap.read()
                    if not ok:
                        raise RuntimeError(f'Source ended unexpectedly at frame {frame}')
                    writer.write(image)
                    written += 1
            finally:
                writer.release()
            check = cv2.VideoCapture(str(path))
            decoded = 0
            try:
                while True:
                    ok,_ = check.read()
                    if not ok:
                        break
                    decoded += 1
            finally:
                check.release()
            if decoded != written:
                raise RuntimeError(f'Clip frame verification failed: {path}')
            records.append(dict(file=path.name,parent_scene_id=scene_id,
                                source_start_frame=start,source_end_frame_exclusive=end,
                                source_start_time_s=round(start/fps,6),frame_count=written,
                                decoded_frames=decoded,duration_s=round(written/fps,6)))
        manifest = dict(source_video=str(video.resolve()),source_fps=fps,image_size=list(size),
                        codec='mp4v',audio_preserved=False,
                        note='Clips are re-encoded, local frame 0 maps to source_start_frame. Scene boundaries are heuristic; source is unchanged.',
                        clips=records)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(manifest,indent=2))
        return manifest
    finally:
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scenes',help='Existing scenes.json from tracking; avoids rescanning transitions')
    parser.add_argument('--scene-ids',nargs='+',type=int,help='Only export selected parent scene IDs')
    args = parser.parse_args()
    scene_data = json.loads(Path(args.scenes).read_text(encoding='utf-8-sig')) if args.scenes else None
    split_video(args.video,args.output,scene_data,args.scene_ids)
