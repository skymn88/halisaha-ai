"""Football tracking v0.2. See README.md for limitations and parameters."""
import argparse
import csv
import json
from pathlib import Path
import cv2
from tracking_helpers import TemporalBall, IDTransitions, jersey_feature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--tracker', choices=['bytetrack.yaml','botsort.yaml'], default='botsort.yaml')
    p.add_argument('--conf', type=float, default=.25, help='Player confidence')
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
    args = p.parse_args()
    if min(args.stride,args.ball_hits,args.ball_gap,args.track_buffer,args.id_gap) < 1:
        p.error('stride, hit counts, gaps and track buffer must be positive')
    if any(not 0 <= v <= 1 for v in [args.conf,args.ball_conf,args.track_high_thresh,args.track_low_thresh,args.new_track_thresh,args.match_thresh]):
        p.error('Confidence and tracker thresholds must be in [0,1]')
    if args.track_low_thresh > args.track_high_thresh: p.error('low threshold must not exceed high threshold')
    if min(args.ball_radius,args.ball_movement,args.id_radius,args.imgsz) <= 0: p.error('radii, movement and imgsz must be positive')
    from ultralytics import YOLO
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
    model, ball_model = YOLO(args.model), YOLO(args.model)
    names = model.names
    persons = [int(k) for k,v in names.items() if v == 'person']
    balls = [int(k) for k,v in names.items() if v == 'sports ball']
    if not persons or not balls: raise RuntimeError('Model must include person and sports ball classes')
    cap = cv2.VideoCapture(args.video)
    writer = None
    count, ids, event_count, processed = 0,set(),0,0
    temporal = TemporalBall(args.ball_hits,args.ball_gap,args.ball_radius,args.ball_movement)
    transitions = IDTransitions(args.id_gap,args.id_radius)
    try:
        if not cap.isOpened(): raise RuntimeError('Video cannot be opened: '+args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        size = (int(cap.get(3)),int(cap.get(4)))
        writer = cv2.VideoWriter(str(out/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.stride,size)
        if not writer.isOpened(): raise RuntimeError('Video writer could not be opened')
        with (out/'detections.csv').open('w',newline='',encoding='utf-8') as f, (out/'ball_candidates.csv').open('w',newline='',encoding='utf-8') as b, (out/'id_changes.csv').open('w',newline='',encoding='utf-8') as e:
            csvw,bw,ew = csv.writer(f),csv.writer(b),csv.writer(e)
            csvw.writerow(['frame','time_s','track_id','class_name','confidence','x1','y1','x2','y2','foot_x','foot_y','object_type','jersey_l','jersey_a','jersey_b'])
            bw.writerow(['frame','confidence','x1','y1','x2','y2','ball_id','status','white_marking_cue'])
            ew.writerow(['frame','old_id','new_id','gap_frames','distance_px','evidence'])
            frame = 0
            while True:
                ok,img = cap.read()
                if not ok: break
                if frame % args.stride:
                    frame += 1
                    continue
                processed += 1
                result = model.track(img,persist=True,tracker=str(tracker_path),classes=persons,conf=args.conf,imgsz=args.imgsz,verbose=False)[0]
                raw = ball_model.predict(img,classes=balls,conf=args.ball_conf,imgsz=args.imgsz,verbose=False)[0]
                rows,players = [],[]
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().tolist()
                    tid = int(box.id.item()) if box.id is not None else -1
                    rows.append((coords,float(box.conf.item()),tid,'person'))
                    players.append((tid,coords))
                    if tid >= 0: ids.add(tid)
                events = transitions.update(frame,players)
                ew.writerows(events)
                event_count += len(events)
                candidates = [(box.xyxy[0].cpu().tolist(),float(box.conf.item())) for box in raw.boxes]
                for box,score,tid,status,marking in temporal.update(frame,candidates,img):
                    bw.writerow([frame,round(score,4),*box,tid,status,int(marking)])
                    if status == 'accepted':
                        rows.append((box,score,tid,'sports ball'))
                        count += 1
                annotated = img.copy()
                for box,score,tid,label in rows:
                    x1,y1,x2,y2 = map(int,box)
                    cx = (x1+x2)/2
                    feature = jersey_feature(img,box) if label == 'person' else ['','','']
                    csvw.writerow([frame,round(frame/fps,4),tid,label,round(score,4),x1,y1,x2,y2,cx,y2,'player' if label=='person' else 'ball',*feature])
                    color = (0,220,0) if label == 'person' else (0,160,255)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                    cv2.putText(annotated,f'{"P" if label=="person" else "BALL"}{tid} {score:.2f}',(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                    if label == 'person': cv2.circle(annotated,(int(cx),y2),3,color,-1)
                writer.write(annotated)
                frame += 1
        summary = dict(ball_detections=count,unique_player_ids=len(ids),possible_id_transitions=event_count,processed_frames=processed,parameters=vars(args))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
    finally:
        cap.release()
        if writer is not None: writer.release()


if __name__ == '__main__': main()
