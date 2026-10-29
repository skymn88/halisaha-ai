"""First-stage football tracking. Requires pip install -r requirements.txt.
Example: python run_tracking.py --video match.mp4 --output results --model yolo11n.pt
"""
import argparse
import csv
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='results')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--tracker', choices=['bytetrack.yaml','botsort.yaml'], default='botsort.yaml')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=960)
    p.add_argument('--stride', type=int, default=1, help='Frame sampling stride; use 1 for tracking continuity')
    args = p.parse_args()
    from ultralytics import YOLO
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError('Video cannot be opened: ' + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out/'tracked.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps / args.stride, (width,height))
    if not writer.isOpened(): raise RuntimeError('Video writer could not be opened')
    model = YOLO(args.model)
    # COCO person=0, sports ball=32. Generic sports-ball detection is not a reliable football tracker.
    names = model.names
    person_ids = [int(k) for k,v in names.items() if v == 'person']
    ball_ids = [int(k) for k,v in names.items() if v == 'sports ball']
    if not person_ids: raise RuntimeError('Model must include a person class')
    classes = person_ids + ball_ids
    with (out/'detections.csv').open('w',newline='',encoding='utf-8') as f:
        csvw=csv.writer(f)
        csvw.writerow(['frame','time_s','track_id','class_name','confidence','x1','y1','x2','y2','foot_x','foot_y'])
        frame=0
        while True:
            ok,img=cap.read()
            if not ok: break
            if frame % args.stride:
                frame+=1; continue
            result=model.track(img,persist=True,tracker=args.tracker,classes=classes,conf=args.conf,imgsz=args.imgsz,verbose=False)[0]
            annotated=img.copy()
            boxes=result.boxes
            if boxes is not None:
                xyxy=boxes.xyxy.cpu().numpy()
                cls=boxes.cls.cpu().numpy().astype(int)
                conf=boxes.conf.cpu().numpy()
                ids=boxes.id.int().cpu().tolist() if boxes.id is not None else [-1]*len(cls)
                for (x1,y1,x2,y2),class_id,score,tid in zip(xyxy,cls,conf,ids):
                    label=names[int(class_id)]
                    x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
                    cx=(x1+x2)/2; footy=y2
                    csvw.writerow([frame,round(frame/fps,4),tid,label,round(float(score),4),x1,y1,x2,y2,round(cx,2),footy])
                    color=(0,220,0) if label=='person' else (0,160,255)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                    cv2.putText(annotated,f'{"P" if label=="person" else "BALL"}{tid if tid>=0 else "?"}',(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                    if label=='person': cv2.circle(annotated,(int(cx),int(footy)),3,color,-1)
            writer.write(annotated)
            frame+=1
    cap.release(); writer.release()
    print('Saved:',out/'tracked.mp4',out/'detections.csv')
    print('NOTE: player IDs can switch after occlusions; no team, identity, or metric-distance claims yet.')

if __name__=='__main__': main()
