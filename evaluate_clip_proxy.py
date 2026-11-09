"""AI temporal coverage and unmatched review burden, separate from human accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

TARGET_TYPES={'shot_candidate','goal_entry_candidate','endline_crossing'}

def score(events, anchor_s, duration_s, tolerance_s=2.):
    targets=[e for e in events if e['type'] in TARGET_TYPES]
    matched=[e for e in targets if abs(e['time_s']-anchor_s)<=tolerance_s]
    unmatched=[e for e in events if abs(e['time_s']-anchor_s)>tolerance_s or e['type'] not in TARGET_TYPES]
    return dict(ai_window_hit=bool(matched),target_candidates=len(targets),all_candidates=len(events),
                matched_candidates=len(matched),candidates_per_minute=len(events)*60/duration_s,
                unmatched_review_candidates_per_minute=len(unmatched)*60/duration_s,
                false_candidates_per_minute=None,human_precision=None,human_recall=None)


def evaluate(root, baseline='results/full_analysis_v052_release', labels='configs/ai_event_windows_v06.json'):
    records=json.loads(Path(labels).read_text());rows=[]
    for label in records['scenes']:
        scene=label['scene'];directory=Path(root)/scene
        summary=json.loads((directory/'summary.json').read_text())
        if summary['video_sha256']!=label['video_sha256']:raise ValueError('AI proxy clip hash mismatch')
        if label['annotation_source']!='ai_reviewed':raise ValueError('This evaluator accepts AI proxies only')
        fps=summary['fps'];anchor=label['anchor_frame']/fps
        events=json.loads((directory/'events.json').read_text())
        before=json.loads((Path(baseline)/scene/'events.json').read_text())
        rows.append(dict(scene=scene,anchor_s=anchor,uncertainty_s=[label['start_frame']/fps,label['end_frame']/fps],
                         annotation_source='ai_reviewed',notes=label['notes'],
                         anchor_evaluable=label['anchor_frame'] not in summary['unevaluable_frames'],
                         baseline=score(before,anchor,summary['duration_s'],records['matching_tolerance_s']),
                         current=score(events,anchor,summary['duration_s'],records['matching_tolerance_s'])))
    result=dict(rows=rows,baseline_hits=sum(r['baseline']['ai_window_hit'] for r in rows),
                current_hits=sum(r['current']['ai_window_hit'] for r in rows),clips=len(rows),
                human_labels=0,limitations=records['limitations']+['Temporal overlap does not establish correct object, shot or goal. Unmatched proposals may include real events outside the selected window.'],
                labels_sha256=hashlib.sha256(Path(labels).read_bytes()).hexdigest())
    (Path(root)/'clip_proxy.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();r=evaluate(a.root)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','limitations')},indent=2))
