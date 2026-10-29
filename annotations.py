"""Shared rendering; observations, unassigned detections and predictions look different."""
import cv2


def draw_object(image, box, label, state='observed', ball=False):
    x1,y1,x2,y2 = map(int,box)
    if state=='predicted':
        color = (180,180,180)
        for start,end in [((x1,y1),(x2,y1)),((x1,y2),(x2,y2)),((x1,y1),(x1,y2)),((x2,y1),(x2,y2))]:
            dx,dy = end[0]-start[0],end[1]-start[1]
            length = max(abs(dx),abs(dy),1)
            for i in range(0,length,10):
                j = min(i+5,length)
                cv2.line(image,(start[0]+int(dx*i/length),start[1]+int(dy*i/length)),
                         (start[0]+int(dx*j/length),start[1]+int(dy*j/length)),color,1)
    else:
        color = (0,190,255) if state!='observed' else ((0,160,255) if ball else (0,220,0))
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2 if state=='observed' else 1)
    cv2.putText(image,label,(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
