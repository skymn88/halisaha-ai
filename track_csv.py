"""Replay person tracking from detector-neutral CSV, without YOLO or a required video."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import cv2
from detection_io import BASE_FIELDS, read_detections
from player_tracking import PlayerTracker
from tracking_helpers import IDTransitions, jersey_feature
from calibration import load_geometries
from annotations import draw_object


def replay(args):
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    records = read_detections(args.detections)
    meta_path = Path(args.metadata) if args.metadata else Path(args.detections).with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig')) if meta_path.exists() else {}
    fps = args.fps if args.fps is not None else meta.get('fps')
    if fps is None or not math.isfinite(fps) or fps<=0:
        raise ValueError('Provide --fps or a detection metadata sidecar with positive fps')
    stride = args.stride if args.stride is not None else meta.get('stride',1)
    frame_count = args.frame_count if args.frame_count is not None else meta.get('frame_count',max((d.frame for d in records),default=-1)+1)
    start = args.start_frame if args.start_frame is not None else (0 if meta else min((d.frame for d in records),default=0))
    if stride<1 or frame_count<=start or start<0 or args.hold_frames<0 or args.id_gap<1 or not math.isfinite(args.id_radius) or args.id_radius<=0:
        raise ValueError('Invalid frame range, stride or hold duration')
    if not 0<=args.person_floor<=args.conf<=1:
        raise ValueError('person-floor must be in [0,conf]')
    stored_floor = meta.get('person_det_conf')
    if stored_floor is not None and args.person_floor<stored_floor:
        raise ValueError('Requested floor is below the cached detector floor; missing observations require new detections')
    grouped = defaultdict(list)
    frame_scenes = {}
    for d in records:
        if d.frame<start or d.frame>=frame_count:
            raise ValueError('Detection frame falls outside requested frame range')
        if (d.frame-start)%stride:
            raise ValueError('Detection frame is incompatible with requested stride/start')
        if d.frame in frame_scenes and frame_scenes[d.frame]!=d.scene_id:
            raise ValueError('A source frame cannot belong to multiple scenes')
        frame_scenes[d.frame] = d.scene_id
        if d.is_person and d.confidence>=args.person_floor:
            grouped[d.frame].append(d)
    tracker_path = Path(args.tracker_config) if args.tracker_config else Path(args.detections).parent/'tracker.yaml'
    config = YAML.load(str(tracker_path)) if tracker_path.exists() else YAML.load(check_yaml(args.tracker))
    for key in ('track_buffer','track_high_thresh','track_low_thresh','new_track_thresh','match_thresh'):
        value = getattr(args,key)
        if value is not None:
            config[key] = value
    if config['track_buffer']<1 or not 0<=config['track_low_thresh']<=config['track_high_thresh']<=1 or any(not 0<=config[k]<=1 for k in ('new_track_thresh','match_thresh')):
        raise ValueError('Invalid resolved tracker thresholds or buffer')
    if config['tracker_type']=='botsort':
        config['gmc_method'],config['with_reid'] = 'none',False
    scenes = meta.get('scenes',[])
    if scenes:
        for detection in records:
            matching = [s for s in scenes if s['start_frame']<=detection.frame<s['end_frame']]
            if len(matching)!=1 or matching[0]['scene_id']!=detection.scene_id:
                raise ValueError('Detection scene/frame does not match the metadata boundaries')
    scene_ids = {d.scene_id for d in records}|{s['scene_id'] for s in scenes} or {0}
    image_size = args.image_size or meta.get('image_size')
    camera_path = args.camera_config or (str(Path(args.detections).parent/'camera_config.json') if (Path(args.detections).parent/'camera_config.json').exists() else None)
    cap,writer = None,None
    output = Path(args.output)
    if (output/'detections.csv').exists():
        raise ValueError('Choose a fresh output directory; existing detections will not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    try:
        if args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise ValueError('Optional review video cannot be opened')
            size = (int(cap.get(3)),int(cap.get(4)))
            if image_size and list(size)!=image_size:
                raise ValueError('Review video dimensions do not match cached detections')
            image_size = list(size)
            writer = cv2.VideoWriter(str(output/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/stride,size)
            if not writer.isOpened():
                raise RuntimeError('Cannot create review video')
        if camera_path and image_size is None:
            raise ValueError('Calibration needs image_size in metadata or an optional matching review video')
        geometries = load_geometries(camera_path,image_size,scene_ids)
        tracker = None
        previous_scene = None
        transitions = IDTransitions(args.id_gap,args.id_radius)
        ids,totals = set(),defaultdict(int)
        with (output/'detections.csv').open('w',newline='',encoding='utf-8') as f, (output/'track_states.csv').open('w',newline='',encoding='utf-8') as t, (output/'player_diagnostics.csv').open('w',newline='',encoding='utf-8') as g, (output/'id_changes.csv').open('w',newline='',encoding='utf-8') as e:
            dw,tw,gw,ew = csv.writer(f),csv.writer(t),csv.writer(g),csv.writer(e)
            dw.writerow(BASE_FIELDS+['object_type','jersey_l','jersey_a','jersey_b','scene_id','pitch_x_m','pitch_y_m','tracking_state'])
            tw.writerow(['frame','time_s','scene_id','object_type','track_id','state','age_frames','confidence','x1','y1','x2','y2'])
            gw.writerow(['frame','time_s','scene_id','raw_persons','outside_pitch','associated','low_conf_associated','unassigned','predicted'])
            ew.writerow(['frame','old_id','new_id','gap_frames','distance_px','evidence','scene_id'])
            current_scene = records[0].scene_id if records else min(scene_ids)
            for frame in range(start,frame_count,stride):
                if scenes:
                    current_scene = next(s['scene_id'] for s in scenes if s['start_frame']<=frame<s['end_frame'])
                elif frame in frame_scenes:
                    current_scene = frame_scenes[frame]
                if current_scene!=previous_scene:
                    tracker = PlayerTracker(config,args.conf,args.hold_frames)
                    transitions = IDTransitions(args.id_gap,args.id_radius)
                    previous_scene = current_scene
                geometry = geometries.get(current_scene)
                observed,predicted = tracker.update(frame,grouped[frame],geometry)
                diag = tracker.last_diagnostics
                gw.writerow([frame,round(frame/fps,4),current_scene,*diag.values()])
                for key,value in diag.items():
                    totals[key] += value
                image = None
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
                    ok,image = cap.read()
                    if not ok:
                        raise RuntimeError(f'Review video lacks frame {frame}')
                players = []
                for o in observed:
                    tid,score,box,state = o['track_id'],o['confidence'],o['box'],o['state']
                    x1,y1,x2,y2 = map(int,box)
                    foot = ((x1+x2)/2,y2)
                    feature = jersey_feature(image,box) if image is not None else ['','','']
                    pitch = geometry.project(foot) if geometry else ['','']
                    dw.writerow([frame,round(frame/fps,4),tid,'person',round(score,4),x1,y1,x2,y2,*foot,'player',*feature,current_scene,*pitch,state])
                    tw.writerow([frame,round(frame/fps,4),current_scene,'player',tid,state,0,round(score,4),*box])
                    players.append((tid,box))
                    if tid>=0:
                        ids.add((current_scene,tid))
                    if image is not None:
                        draw_object(image,box,f'P{tid if tid>=0 else "?"} {score:.2f}',state)
                for o in predicted:
                    tw.writerow([frame,round(frame/fps,4),current_scene,'player',o['track_id'],'predicted',o['age_frames'],'',*o['box']])
                    if image is not None:
                        draw_object(image,o['box'],f'P{o["track_id"]} predicted {o["age_frames"]}f','predicted')
                events = transitions.update(frame,players)
                ew.writerows([event+[current_scene] for event in events])
                totals['possible_id_transitions'] += len(events)
                totals['processed_frames'] += 1
                if writer is not None:
                    writer.write(image)
        result = dict(unique_player_ids=len(ids),**dict(totals),parameters=vars(args),
                      source_record_kind=meta.get('record_kind','external_unspecified'),applied_camera_config=camera_path,
                      ignored_nonperson_rows=sum(not d.is_person for d in records),
                      note='No detector/model weights run. Predictions are separate from observations. External tracked CSV cannot recover already discarded detections.')
        (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        YAML.save(str(output/'tracker.yaml'),config)
        print(json.dumps(result,indent=2))
        return result
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detections',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--metadata')
    p.add_argument('--fps',type=float)
    p.add_argument('--frame-count',type=int)
    p.add_argument('--start-frame',type=int)
    p.add_argument('--stride',type=int)
    p.add_argument('--video',help='Optional original input, only for review rendering')
    p.add_argument('--camera-config')
    p.add_argument('--image-size',nargs=2,type=int,metavar=('WIDTH','HEIGHT'))
    p.add_argument('--id-gap',type=int,default=8)
    p.add_argument('--id-radius',type=float,default=35.)
    p.add_argument('--person-floor',type=float,default=.10)
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--hold-frames',type=int,default=3)
    p.add_argument('--tracker',choices=['botsort.yaml','bytetrack.yaml'],default='botsort.yaml')
    p.add_argument('--tracker-config')
    p.add_argument('--track-buffer',type=int,help='Override saved tracker configuration')
    p.add_argument('--track-high-thresh',type=float,help='Override saved tracker configuration')
    p.add_argument('--track-low-thresh',type=float,help='Override saved tracker configuration')
    p.add_argument('--new-track-thresh',type=float,help='Override saved tracker configuration')
    p.add_argument('--match-thresh',type=float,help='Override saved tracker configuration')
    return p


if __name__=='__main__':
    replay(parser().parse_args())
