"""Compare v0.2 ball postprocessing against saved v0.3 proposals without rerunning YOLO."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import cv2
from tracking_helpers import TemporalBall


def compare(video, directory, review_region=None, review_scene=0):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    parameters = summary['parameters']
    tracker = TemporalBall(parameters['ball_hits'],parameters['ball_gap'],
                           parameters['ball_radius'],parameters['ball_movement'])
    proposals = defaultdict(list)
    scene_for_frame = {}
    with (directory/'ball_candidates.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame'])
            scene_for_frame[frame] = int(row['scene_id'])
            proposals[frame].append(([float(row[k]) for k in ('x1','y1','x2','y2')],float(row['confidence'])))
    accepted = []
    per_scene = defaultdict(lambda: Counter())
    region_counts = Counter({'v02_replay_accepted':0,'v03_accepted':0})
    states = Counter()
    with (directory/'ball_track.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            states[row['state']] += 1
            scene_for_frame[int(row['frame'])] = int(row['scene_id'])
    def in_region(box, scene_id):
        if review_region is None or scene_id != review_scene:
            return False
        x,y = (box[0]+box[2])/2,(box[1]+box[3])/2
        a,b,c,d = review_region
        return a <= x <= c and b <= y <= d
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    frame = 0
    try:
        while True:
            ok,image = cap.read()
            if not ok:
                break
            if frame % parameters['stride'] == 0:
                scene = scene_for_frame.get(frame,0)
                for box,score,tid,status,_ in tracker.update(frame,proposals[frame],image):
                    if status == 'accepted':
                        accepted.append([frame,scene,tid,round(score,4),*box])
                        per_scene[scene]['v02_replay_ball_rows'] += 1
                        if in_region(box,scene):
                            region_counts['v02_replay_accepted'] += 1
            frame += 1
    finally:
        cap.release()
    with (directory/'detections.csv').open(encoding='utf-8') as stream:
        players = defaultdict(set)
        for row in csv.DictReader(stream):
            scene = int(row['scene_id'])
            if row['object_type'] == 'ball':
                per_scene[scene]['v03_ball_rows'] += 1
                if in_region([float(row[k]) for k in ('x1','y1','x2','y2')],scene):
                    region_counts['v03_accepted'] += 1
            elif int(row['track_id']) >= 0:
                players[scene].add(int(row['track_id']))
        for scene,ids in players.items():
            per_scene[scene]['unique_person_ids'] = len(ids)
    with (directory/'v02_ball_replay.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame','scene_id','ball_id','confidence','x1','y1','x2','y2'])
        writer.writerows(accepted)
    cap = cv2.VideoCapture(str(directory/'tracked.mp4'))
    decoded = 0
    try:
        while True:
            ok,_ = cap.read()
            if not ok:
                break
            decoded += 1
    finally:
        cap.release()
    if decoded != summary['processed_frames']:
        raise ValueError(f'Output video frame count mismatch: {decoded} vs {summary["processed_frames"]}')
    result = dict(source_decoded_frames=frame,output_decoded_frames=decoded,
                  v02_replay_ball_rows=len(accepted),v03_ball_rows=summary['ball_detections'],
                  ball_states=dict(states),per_scene={str(k):dict(v) for k,v in sorted(per_scene.items())},
                  review_region=review_region,review_scene=review_scene,review_region_counts=dict(region_counts),
                  note='Same saved YOLO proposals; original v0.2 filter replayed without resets across scenes. Counts are not accuracy scores.')
    (directory/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--output',required=True,help='Completed v0.3 output directory')
    parser.add_argument('--review-region',nargs=4,type=float,metavar=('X1','Y1','X2','Y2'))
    parser.add_argument('--review-scene',type=int,default=0)
    args = parser.parse_args()
    compare(args.video,args.output,args.review_region,args.review_scene)
