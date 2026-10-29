"""Small Ultralytics detector adapter; tracking consumes only Detection records."""
from detection_io import Detection


class YoloDetector:
    def __init__(self, weights, ball_enabled=True):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.person_classes = [int(k) for k,v in self.model.names.items() if v=='person']
        self.ball_classes = [int(k) for k,v in self.model.names.items() if v=='sports ball']
        if not self.person_classes or (ball_enabled and not self.ball_classes):
            raise ValueError('Model needs person class and, when enabled, sports ball class')
        self.ball_enabled = ball_enabled

    def detect(self, image, frame, time_s, scene_id, person_conf, ball_conf, imgsz):
        classes = self.person_classes+(self.ball_classes if self.ball_enabled else [])
        floor = min(person_conf,ball_conf) if self.ball_enabled else person_conf
        result = self.model.predict(image,classes=classes,conf=floor,imgsz=imgsz,verbose=False)[0]
        records = []
        for box in result.boxes:
            person = int(box.cls.item()) in self.person_classes
            score = float(box.conf.item())
            if score < (person_conf if person else ball_conf):
                continue
            records.append(Detection(frame,time_s,'person' if person else 'sports ball',score,
                                     tuple(box.xyxy[0].cpu().tolist()),scene_id,'full'))
        return records

    def detect_ball_crop(self, crop, frame, time_s, scene_id, confidence, imgsz, offset, source):
        result = self.model.predict(crop,classes=self.ball_classes,conf=confidence,imgsz=imgsz,verbose=False)[0]
        records = []
        dx,dy = offset
        for box in result.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().tolist()
            records.append(Detection(frame,time_s,'sports ball',float(box.conf.item()),
                                     (x1+dx,y1+dy,x2+dx,y2+dy),scene_id,source))
        return records
