"""Football tracking v0.3.1. See README.md for limitations and parameters."""
import argparse
import csv
import json
from pathlib import Path
import cv2
from tracking_helpers import TemporalBall, IDTransitions, jersey_feature
from ball_tracking import PitchBallTracker
from pitch_context import PitchBackground, scan_scenes
from calibration import load_geometries
from detection_io import RAW_FIELDS
from detector import YoloDetector
from player_tracking import PlayerTracker
from ball_detection import BallSearch
from annotations import draw_object


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--tracker', choices=['bytetrack.yaml','botsort.yaml'], default='botsort.yaml')
    p.add_argument('--conf', type=float, default=.25, help='Strong/unassigned person display threshold; raw floor is --person-det-conf')
    p.add_argument('--ball-conf', type=float, default=.15)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--ball-hits', type=int, default=3)
    p.add_argument('--ball-gap', type=int, default=8, help='Original source frames')
    p.add_argument('--ball-radius', type=float, default=45., help='Pixel association radius')
    p.add_argument('--ball-movement', type=float, default=4., help='Pixel displacement to clear marking caution')
    p.add_argument('--track-buffer', type=int, default=60)
    p.add_argument('--track-high-thresh', type=float, default=.25)
    p.add_argument('--track-low-thresh', type=float, default=.1)
    p.add_argument('--new-track-thresh', type=float, default=.25)
    p.add_argument('--match-thresh', type=float, default=.8)
    p.add_argument('--id-gap', type=int, default=8)
    p.add_argument('--id-radius', type=float, default=35.)
    p.add_argument('--ball-mode', choices=['pitch', 'v02', 'off'], default='pitch')
    p.add_argument('--background-samples', type=int, default=41)
    p.add_argument('--background-difference', type=float, default=30.)
    p.add_argument('--foreground-min', type=float, default=.25)
    p.add_argument('--scene-threshold', type=float, default=23.)
    p.add_argument('--scene-window', type=int, default=30)
    p.add_argument('--scene-min-frames', type=int, default=60)
    p.add_argument('--camera-config', help='Optional per-scene JSON geometry with measured correspondences')
    p.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads; avoids CPU oversubscription')
    p.add_argument('--person-det-conf', type=float, default=.10, help='Raw person confidence floor, including weak recovery observations')
    p.add_argument('--label-hold', type=int, default=3, help='Source frames for explicitly predicted labels; 0 disables')
    p.add_argument('--ball-search',action='store_true',help='Opt-in tiled/ROI experiment; no measured ROI recall gain on current validation clip')
    p.add_argument('--ball-search-interval', type=int, default=15, help='Source-frame interval for tiled acquisition; 0 disables')
    p.add_argument('--ball-tile-fraction', type=float, default=.75)
    p.add_argument('--ball-roi-fraction', type=float, default=.35)
    p.add_argument('--ball-crop-imgsz', type=int, default=640)
    p.add_argument('--ball-reacquire-hits', type=int, default=3)
    args = p.parse_args()
    if min(args.stride,args.ball_hits,args.ball_gap,args.track_buffer,args.id_gap) < 1:
        p.error('stride, hit counts, gaps and track buffer must be positive')
    if any(not 0 <= v <= 1 for v in [args.conf,args.ball_conf,args.track_high_thresh,args.track_low_thresh,args.new_track_thresh,args.match_thresh]):
        p.error('Confidence and tracker thresholds must be in [0,1]')
    if args.track_low_thresh > args.track_high_thresh: p.error('low threshold must not exceed high threshold')
    if min(args.ball_radius,args.ball_movement,args.id_radius,args.imgsz) <= 0: p.error('radii, movement and imgsz must be positive')
    if min(args.background_samples,args.scene_window,args.scene_min_frames,args.threads) < 1:
        p.error('background samples, scene window/minimum frames and threads must be positive')
    if not 0 < args.foreground_min <= 1 or args.background_difference <= 0 or args.scene_threshold <= 0:
        p.error('foreground fraction must be in (0,1]; difference and scene thresholds must be positive')
    if args.ball_mode == 'v02' and args.camera_config:
        p.error('camera geometry is unavailable in legacy v02 mode')
    import torch
    torch.set_num_threads(args.threads)
    if not 0<=args.person_det_conf<=args.conf<=1:
        p.error('person-det-conf must be in [0, conf]')
    if min(args.label_hold,args.ball_search_interval)<0 or min(args.ball_crop_imgsz,args.ball_reacquire_hits)<1:
        p.error('hold/search interval must be nonnegative; crop size/reacquisition hits positive')
    if not 0<args.ball_tile_fraction<=1 or not 0<args.ball_roi_fraction<=1:
        p.error('tile/ROI fractions must be in (0,1]')
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = YAML.load(check_yaml(args.tracker))
    for key in ('track_buffer','track_high_thresh','track_low_thresh','new_track_thresh','match_thresh'):
        config[key] = getattr(args,key)
    if config['tracker_type'] == 'botsort': config['gmc_method'] = 'none'  # fixed cameras
    tracker_path = out/'tracker.yaml'
    YAML.save(str(tracker_path),config)
    detector = YoloDetector(args.model,ball_enabled=args.ball_mode!='off')
    player_tracker = PlayerTracker(config,args.conf,args.label_hold)
    ball_search = BallSearch(detector,args.ball_search_interval,args.ball_tile_fraction,args.ball_roi_fraction,args.ball_crop_imgsz)
    cap = cv2.VideoCapture(args.video)
    writer = None
    count, ids, event_count, processed = 0,set(),0,0
    temporal = TemporalBall(args.ball_hits,args.ball_gap,args.ball_radius,args.ball_movement)
    ball_tracker = PitchBallTracker(args.ball_hits,args.ball_gap,args.ball_radius,args.ball_reacquire_hits)
    backgrounds = {}
    geometry = None
    scene_id = 0
    candidate_statuses = {}
    raw_candidates = 0
    transitions = IDTransitions(args.id_gap,args.id_radius)
    try:
        if not cap.isOpened(): raise RuntimeError('Video cannot be opened: '+args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        size = (int(cap.get(3)),int(cap.get(4)))
        scenes = scan_scenes(args.video,args.scene_threshold,args.scene_window,args.scene_min_frames) if args.ball_mode != 'v02' else [dict(scene_id=0,start_frame=0,end_frame=int(cap.get(7)))]
        if not scenes or scenes[-1]['end_frame'] == 0:
            raise RuntimeError('Video contains no decodable frames')
        geometries = load_geometries(args.camera_config,size,{s['scene_id'] for s in scenes})
        for scene in scenes:
            g = geometries.get(scene['scene_id'])
            scene['calibrated'] = g is not None and g.homography is not None
            scene['reprojection_rmse_px'] = g.reprojection_rmse_px if g else None
            scene['homography_image_to_pitch'] = g.homography.tolist() if g is not None and g.homography is not None else None
        if args.camera_config:
            (out/'camera_config.json').write_text(Path(args.camera_config).read_text(encoding='utf-8-sig'),encoding='utf-8')
        (out/'scenes.json').write_text(json.dumps(scenes,indent=2),encoding='utf-8')
        if args.ball_mode == 'pitch':
            for scene in scenes:
                backgrounds[scene['scene_id']] = PitchBackground.from_video(
                    args.video,scene,out/'pitch_context',args.background_samples,
                    args.background_difference,args.foreground_min)
        print(f'Prepared {len(scenes)} scene(s); processing {scenes[-1]["end_frame"]} frames',flush=True)
        writer = cv2.VideoWriter(str(out/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.stride,size)
        if not writer.isOpened(): raise RuntimeError('Video writer could not be opened')
        with (out/'detections.csv').open('w',newline='',encoding='utf-8') as f, (out/'ball_candidates.csv').open('w',newline='',encoding='utf-8') as b, (out/'id_changes.csv').open('w',newline='',encoding='utf-8') as e, (out/'ball_track.csv').open('w',newline='',encoding='utf-8') as t, (out/'raw_detections.csv').open('w',newline='',encoding='utf-8') as rd, (out/'player_diagnostics.csv').open('w',newline='',encoding='utf-8') as pd, (out/'track_states.csv').open('w',newline='',encoding='utf-8') as ts:
            csvw,bw,ew,tw = csv.writer(f),csv.writer(b),csv.writer(e),csv.writer(t)
            csvw.writerow(['frame','time_s','track_id','class_name','confidence','x1','y1','x2','y2','foot_x','foot_y','object_type','jersey_l','jersey_a','jersey_b','scene_id','pitch_x_m','pitch_y_m','tracking_state'])
            bw.writerow(['frame','confidence','x1','y1','x2','y2','ball_id','status','white_marking_cue','scene_id','background_marking_fraction','foreground_fraction','innovation_px'])
            ew.writerow(['frame','old_id','new_id','gap_frames','distance_px','evidence','scene_id'])
            tw.writerow(['frame','time_s','scene_id','ball_id','state','confidence','center_x','center_y'])
            raw_writer,person_diag,state_writer = csv.writer(rd),csv.writer(pd),csv.writer(ts)
            raw_writer.writerow(RAW_FIELDS)
            person_diag.writerow(['frame','time_s','scene_id','raw_persons','outside_pitch','associated','low_conf_associated','unassigned','predicted'])
            state_writer.writerow(['frame','time_s','scene_id','object_type','track_id','state','age_frames','confidence','x1','y1','x2','y2'])
            raw_count = 0
            low_recoveries,unassigned_count,prediction_count,crop_passes = 0,0,0,0
            frame = 0
            while True:
                ok,img = cap.read()
                if not ok: break
                if frame % args.stride:
                    frame += 1
                    continue
                new_scene = next(s['scene_id'] for s in scenes if s['start_frame'] <= frame < s['end_frame'])
                if new_scene != scene_id:
                    scene_id = new_scene
                    ball_tracker = PitchBallTracker(args.ball_hits,args.ball_gap,args.ball_radius,args.ball_reacquire_hits)
                    transitions = IDTransitions(args.id_gap,args.id_radius)
                    player_tracker = PlayerTracker(config,args.conf,args.label_hold)
                geometry = geometries.get(scene_id)
                processed += 1
                observations = detector.detect(img,frame,frame/fps,scene_id,args.person_det_conf,args.ball_conf,args.imgsz)
                person_observations = [d for d in observations if d.is_person]
                ball_observations = [d for d in observations if not d.is_person]
                if args.ball_mode=='pitch' and args.ball_search:
                    ball_observations = ball_search.propose(img,frame,frame/fps,scene_id,ball_observations,
                                                          ball_tracker,backgrounds[scene_id],args.ball_conf)
                    crop_passes += ball_search.last_passes
                for detection in person_observations+ball_observations:
                    raw_writer.writerow(detection.csv_row())
                    raw_count += 1
                person_rows,predictions = player_tracker.update(frame,person_observations,geometry)
                diag = player_tracker.last_diagnostics
                person_diag.writerow([frame,round(frame/fps,4),scene_id,*diag.values()])
                low_recoveries += diag['low_conf_associated']
                unassigned_count += diag['unassigned']
                prediction_count += len(predictions)
                rows,players,person_states = [],[],{}
                for observation in person_rows:
                    coords,score,tid = observation['box'],observation['confidence'],observation['track_id']
                    rows.append((coords,score,tid,'person'))
                    players.append((tid,coords))
                    person_states[(tid,tuple(coords))] = observation['state']
                    state_writer.writerow([frame,round(frame/fps,4),scene_id,'player',tid,observation['state'],0,round(score,4),*coords])
                    if tid>=0:
                        ids.add((scene_id,tid))
                for prediction in predictions:
                    state_writer.writerow([frame,round(frame/fps,4),scene_id,'player',prediction['track_id'],'predicted',prediction['age_frames'],'',*prediction['box']])
                events = transitions.update(frame,players)
                ew.writerows([event+[scene_id] for event in events])
                event_count += len(events)
                candidates = [(d.box,d.confidence) for d in ball_observations]
                raw_candidates += len(candidates)
                if args.ball_mode == 'pitch':
                    accepted = ball_tracker.update(frame,candidates,img,backgrounds[scene_id],geometry,[d.box for d in person_observations])
                    for diagnostic in ball_tracker.last_diagnostics:
                        status = diagnostic['status']
                        candidate_statuses[status] = candidate_statuses.get(status,0)+1
                        bw.writerow([frame,round(diagnostic['confidence'],4),*diagnostic['box'],diagnostic['ball_id'],status,
                                     int(diagnostic['static_marking']),scene_id,round(diagnostic['marking_fraction'],4),
                                     round(diagnostic['foreground_fraction'],4),diagnostic['innovation_px']])
                    if accepted:
                        box,score,tid,_,_ = accepted[0]
                        tw.writerow([frame,round(frame/fps,4),scene_id,tid,'observed',round(score,4),
                                     (box[0]+box[2])/2,(box[1]+box[3])/2])
                    else:
                        active = ball_tracker.active
                        tw.writerow([frame,round(frame/fps,4),scene_id,active['id'] if active else -1,
                                     ball_tracker.last_state,'','',''])
                elif args.ball_mode=='v02':
                    accepted = temporal.update(frame,candidates,img)
                    for box,score,tid,status,marking in accepted:
                        candidate_statuses[status] = candidate_statuses.get(status,0)+1
                        bw.writerow([frame,round(score,4),*box,tid,status,int(marking),scene_id,'','',''])
                    tw.writerow([frame,round(frame/fps,4),scene_id,-1,'v02_multi_hypothesis','','',''])
                else:
                    accepted = []
                    tw.writerow([frame,round(frame/fps,4),scene_id,-1,'disabled','','',''])
                for box,score,tid,status,marking in accepted:
                    if status == 'accepted':
                        rows.append((box,score,tid,'sports ball'))
                        count += 1
                annotated = img.copy()
                for box,score,tid,label in rows:
                    x1,y1,x2,y2 = map(int,box)
                    cx = (x1+x2)/2
                    feature = jersey_feature(img,box) if label == 'person' else ['','','']
                    state = person_states.get((tid,tuple(box)),'observed') if label=='person' else 'observed'
                    csvw.writerow([frame,round(frame/fps,4),tid,label,round(score,4),x1,y1,x2,y2,cx,y2,'player' if label=='person' else 'ball',*feature,scene_id,*(geometry.project((cx,y2)) if geometry is not None and label == 'person' else ['','']),state])
                    prefix = 'P' if label=='person' else 'BALL'
                    identity = str(tid) if tid>=0 else '?'
                    draw_object(annotated,box,f'{prefix}{identity} {score:.2f}',state,label!='person')
                for prediction in predictions:
                    draw_object(annotated,prediction['box'],f'P{prediction["track_id"]} predicted {prediction["age_frames"]}f','predicted')
                if args.ball_mode=='pitch' and not accepted:
                    selected = ball_tracker.last_selected
                    if selected is not None:
                        draw_object(annotated,selected['box'],f'BALL? {selected["confidence"]:.2f}','tentative',True)
                    predicted = ball_tracker.prediction(frame,args.label_hold)
                    if predicted is not None:
                        draw_object(annotated,predicted['box'],f'BALL{predicted["track_id"]} predicted {predicted["age_frames"]}f','predicted',True)
                        state_writer.writerow([frame,round(frame/fps,4),scene_id,'ball',predicted['track_id'],'predicted',predicted['age_frames'],'',*predicted['box']])
                if args.ball_mode == 'pitch':
                    cv2.putText(annotated,f'Scene {scene_id} | ball: {ball_tracker.last_state}',(6,size[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX,.45,(0,190,255),1)
                writer.write(annotated)
                if processed % 300 == 0:
                    print(f'Processed {processed} frames; observed ball rows: {count}',flush=True)
                frame += 1
        summary = dict(ball_detections=count,unique_player_ids=len(ids),possible_id_transitions=event_count,processed_frames=processed,raw_ball_candidates=raw_candidates,candidate_statuses=candidate_statuses,scene_count=len(scenes),raw_detection_rows=raw_count,low_conf_person_associations=low_recoveries,unassigned_person_observations=unassigned_count,predicted_person_labels=prediction_count,ball_crop_inference_passes=crop_passes,parameters=vars(args))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        metadata = dict(schema_version=1,record_kind='raw_model_observations',source_video=str(Path(args.video).resolve()),
                        fps=fps,image_size=list(size),frame_count=frame,stride=args.stride,scenes=scenes,
                        person_det_conf=args.person_det_conf,ball_conf=args.ball_conf)
        (out/'raw_detections.meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
    finally:
        cap.release()
        if writer is not None: writer.release()


if __name__ == '__main__': main()
