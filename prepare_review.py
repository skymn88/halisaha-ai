"""Prepare unreviewed raw-image windows from local clips and existing CSVs; no inference."""
import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path
import random
import cv2
from ball_cache import sha256_file
from replay_ball import fresh_output
from review_schema import (SCHEMA_VERSION, TABLES, FRAME_FIELDS, read_rows, write_rows)
from split_scenes import split_video


def source_record(source_id, video, run):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Missing local raw video: '+str(video))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(cap.get(3)), int(cap.get(4))]
    finally:
        cap.release()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    raw_meta = run/'raw_detections.meta.json'
    metadata = json.loads(raw_meta.read_text(encoding='utf-8')) if raw_meta.exists() else {}
    return dict(source_id=source_id, video=str(video.resolve()), run=str(run.resolve()),
                fps=fps, frame_count=count, image_size=size, source_sha256=sha256_file(video),
                scenes=json.loads((run/'scenes.json').read_text(encoding='utf-8')),
                model=metadata.get('detector', dict(
                    implementation='historical_ultralytics', package_version='not_recorded_in_cache',
                    weights=summary['parameters'].get('model'), weights_sha256=None,
                    previous_frames=0, future_frames=0)),
                invocation=summary['parameters'],
                cache_sha256=sha256_file(run/'raw_detections.csv') if (run/'raw_detections.csv').exists()
                else sha256_file(run/'detections.csv'))


def prepare(output, short_video='ilk_15_saniye.mp4', montage='halisaha.mp4', seed=42):
    output = fresh_output(output)
    sources = [source_record('short', Path(short_video), Path('results/v031_final')),
               source_record('montage', Path(montage), Path('results/v03_long_sampled'))]
    windows = []
    def add(source, scene, start, length, kind, selection, split):
        end = min(start+length, scene['end_frame'])
        index = len(windows)
        windows.append(dict(window_id=f'w{index:03d}', source_id=source['source_id'],
                            scene_id=scene['scene_id'], start_frame=start, end_frame=end,
                            kind=kind, selection=selection, split=split,
                            split_group=f"{source['source_id']}:scene:{scene['scene_id']}",
                            angle=f"unverified_{source['source_id']}_scene_{scene['scene_id']}",
                            angle_reviewed=False))
    short = sources[0]
    for start, reason in [(30, 'ordinary_unreviewed'), (170, 'weak_person_overlap_example'),
                          (325, 'known_ball_clothing_and_recovery_example')]:
        add(short, short['scenes'][0], start, 40, 'detections', reason, 'dev')
    rng = random.Random(seed)
    for scene_id, split in ((2, 'dev'), (6, 'eval'), (8, 'eval')):
        scene = next(s for s in sources[1]['scenes'] if s['scene_id'] == scene_id)
        first = scene['start_frame']+min(30, (scene['end_frame']-scene['start_frame'])//8)
        last = scene['end_frame']-40
        for start in (first, max(first+40, last)):
            add(sources[1], scene, start, 40, 'detections', 'cross_view_unreviewed', split)
        length = min(int(round(6*sources[1]['fps'])), scene['end_frame']-scene['start_frame'])
        start = rng.randint(scene['start_frame'], scene['end_frame']-length)
        add(sources[1], scene, start, length, 'events', 'random_audit', split)
    # Include the entire short sample for event context, without declaring it labeled.
    add(short, short['scenes'][0], 0, short['frame_count'], 'events', 'known_failure_context', 'dev')
    frames, exported = [], 0
    for source in sources:
        cap = cv2.VideoCapture(source['video'])
        try:
            for window in [w for w in windows if w['source_id'] == source['source_id']]:
                start, end = window['start_frame'], window['end_frame']
                if window['kind'] == 'detections':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    directory = output/'frames'/window['window_id']
                    directory.mkdir(parents=True)
                    for frame in range(start, end):
                        ok, image = cap.read()
                        if not ok or not cv2.imwrite(str(directory/f'{frame:06d}.png'), image):
                            raise ValueError(f'Cannot export raw source frame {frame}')
                        frames.append(dict(window_id=window['window_id'], source_id=source['source_id'],
                                           scene_id=window['scene_id'], frame=frame, ball_state='unreviewed',
                                           people_complete=0, identities_complete=0, review_status='unreviewed',
                                           annotation_source='proposal', reviewer='', notes=''))
                        exported += 1
                bounds = sorted(set([0, start, end, source['frame_count']]))
                partitions = [dict(scene_id=i, start_frame=a, end_frame=b)
                              for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
                selected = next(s['scene_id'] for s in partitions if s['start_frame'] == start)
                directory = output/'clips'/window['window_id']
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = split_video(source['video'], directory, partitions, [selected])
                window['clip'] = str((directory/manifest['clips'][0]['file']).relative_to(output)).replace('\\', '/')
                window['clip_decoded_frames'] = manifest['clips'][0]['decoded_frames']
        finally:
            cap.release()
    config = dict(person_iou=.5, ball_center_tolerance_px=5., event_tolerances_s=[.5, 1.],
                  provenance='preset_before_human_review', source_coordinates=True,
                  note='5 px is an initial annotation tolerance, not a proven universal threshold.')
    (output/'evaluation_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    dataset = dict(schema_version=SCHEMA_VERSION, sources=sources, windows=windows, selection_seed=seed,
                   evaluation_config_sha256=sha256_file(output/'evaluation_config.json'),
                   labeling_status='unreviewed', exported_detection_frames=exported,
                   limitations=['Angle names require human verification.',
                                'Scene-disjoint split is not a held-out-match split; montage may contain repeats.',
                                'Known errors and random audit windows must be reported separately.',
                                'Historical montage uses stride 3/imgsz 640; short cache stride 1/imgsz 960.',
                                'Unprocessed frames have no proposal, not a negative annotation.'])
    (output/'dataset.json').write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, fields in TABLES.items():
        write_rows(output/name, fields, frames if name == 'frame_labels.csv' else [])
    print(json.dumps(dict(windows=len(windows), raw_png_frames=exported, human_labels=0,
                          output=str(output.resolve())), indent=2))
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/review_v04')
    parser.add_argument('--short-video', default='ilk_15_saniye.mp4')
    parser.add_argument('--montage', default='halisaha.mp4')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.output, args.short_video, args.montage, args.seed)
