"""Low-contrast small moving highlights for trajectory verification, not ball decisions."""
import cv2
import numpy as np


class MotionBlurCandidates:
    def __init__(self, background):
        self.background=cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
        self.previous=None

    def update(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.previous is None:self.previous=gray;return []
        change=cv2.absdiff(gray,self.previous);self.previous=gray
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        local=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,np.ones((9,9),np.uint8))
        mask=((change>=8)&(cv2.absdiff(gray,self.background)>=14)&(local>=8)&(gray>=75)&(hsv[:,:,1]<125)).astype('uint8')
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        result=[]
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i])
            if not 2<=area<=100 or max(w,h)>24 or max(w,h)/max(1,min(w,h))>5:continue
            cx,cy=map(float,centers[i])
            result.append(dict(x=cx,y=cy,box=[x,y,x+w,y+h],support=1,moving=1,source='motion_blur',weight=.05))
        return result
