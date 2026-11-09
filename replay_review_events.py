"""Replay event semantics or reviewed geometry from cached observed hypotheses, without decoding."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from player_ball_filter import read
from review_pipeline_v06 import load_config
from reviewed_geometry import resolve_geometry
from tiered_events import generate_events


def replay(source,output,config_path='configs/review_v06.json',geometry_dir=None,reuse_geometry=False):
    config=load_config(config_path);out=Path(output)
    if out.exists():raise ValueError('Output must be fresh')
    roots=sorted(p for p in Path(source).glob('scene_*') if p.is_dir())
    if not roots:raise ValueError('No cached scenes')
    for root in roots:
        previous=json.loads((root/'summary.json').read_text())['config']
        if {k:v for k,v in previous.items() if k!='version'} != {k:v for k,v in config.items() if k!='version'}:
            raise ValueError('Numeric thresholds changed: replay the observation pipeline first')
    out.mkdir()
    for root in roots:
        target=out/root.name;shutil.copytree(root,target)
        summary=json.loads((target/'summary.json').read_text());geometry=json.loads((target/'geometry.json').read_text())
        if not reuse_geometry or geometry_dir:
            geometry=resolve_geometry(Path('results/scene_clips')/(root.name+'.mp4'),geometry_dir)
            # Existing overlay geometry would be stale. Raw video is honest for geometry-only replay.
            shutil.copyfile(Path('results/scene_clips')/(root.name+'.mp4'),target/'analysis.mp4')
            (target/'web.mp4').unlink(missing_ok=True)
            summary['video_overlay']='raw_clip_after_geometry_change'
        people=defaultdict(list)
        for row in read(Path('results/measured')/root.name/'detections.csv'):
            if row['class_name']=='person':people[int(row['frame'])].append(row)
        events=generate_events(json.loads((target/'ball_observations.json').read_text()),people,geometry,summary['fps'],config)
        summary.update(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                       geometry_source=geometry['geometry_source'],parent_observation_run=str(root.resolve()),
                       event_candidates=dict(Counter(e['type'] for e in events)),evidence_tiers=dict(Counter(e['tier'] for e in events)))
        for name,value in [('events',events),('summary',summary),('geometry',geometry)]:
            (target/f'{name}.json').write_text(json.dumps(value,indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--geometry-dir');p.add_argument('--reuse-geometry',action='store_true',help='Explicitly retain input geometry including AI exploratory provenance')
    a=p.parse_args();replay(a.input,a.output,geometry_dir=a.geometry_dir,reuse_geometry=a.reuse_geometry)
