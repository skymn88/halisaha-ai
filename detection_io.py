"""Detector-neutral observation records and backward-compatible CSV transport."""
from dataclasses import dataclass
import csv
import math
from pathlib import Path


BASE_FIELDS = ['frame','time_s','track_id','class_name','confidence','x1','y1','x2','y2','foot_x','foot_y']
RAW_FIELDS = BASE_FIELDS+['scene_id','detector_pass']


@dataclass(frozen=True)
class Detection:
    frame: int
    time_s: float
    class_name: str
    confidence: float
    box: tuple
    scene_id: int = 0
    detector_pass: str = 'external'

    @property
    def is_person(self):
        return self.class_name in ('person','player')

    @property
    def foot(self):
        return ((self.box[0]+self.box[2])/2,self.box[3])

    def csv_row(self):
        return [self.frame,self.time_s,-1,self.class_name,self.confidence,*self.box,*self.foot,
                self.scene_id,self.detector_pass]


def read_detections(path):
    """Read original 11-column or extended CSV; input IDs are never trusted as new tracking IDs."""
    if Path(path).suffix.lower() in ('.parquet','.pq'):
        raise ValueError('Parquet input is not implemented yet; provide CSV with the documented schema')
    records = []
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(BASE_FIELDS)-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing detection columns: {sorted(missing)}')
        for line,row in enumerate(reader,2):
            try:
                frame = int(row['frame'])
                scene = int(row.get('scene_id') or 0)
                time_s = float(row['time_s'])
                score = float(row['confidence'])
                box = tuple(float(row[k]) for k in ('x1','y1','x2','y2'))
                if frame<0 or scene<0 or time_s<0 or not all(math.isfinite(x) for x in (*box,time_s,score)) or not 0<=score<=1 or box[2]<=box[0] or box[3]<=box[1]:
                    raise ValueError('invalid values or degenerate box')
                records.append(Detection(frame,time_s,row['class_name'],score,box,scene,row.get('detector_pass') or 'external'))
            except (ValueError,TypeError) as error:
                raise ValueError(f'Invalid detection at CSV line {line}: {error}') from error
    return records
