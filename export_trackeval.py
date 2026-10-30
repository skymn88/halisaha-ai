"""Export only fully human-labeled contiguous windows to reference TrackEval MOT format."""
import argparse
import configparser
from pathlib import Path
from review_schema import flag, human_reviewed, load_dataset, read_rows, validate_frame
from replay_ball import fresh_output


def export(directory, output, split='eval'):
    directory, output = Path(directory), fresh_output(output)
    dataset = load_dataset(directory)
    sources = {s['source_id']:s for s in dataset['sources']}
    frames = {(r['window_id'],int(r['frame'])):r for r in read_rows(directory/'frame_labels.csv')}
    detections = read_rows(directory/'detection_labels.csv')
    identities = {(r['window_id'],int(r['frame']),r['annotation_id']):r
                  for r in read_rows(directory/'identity_labels.csv') if human_reviewed(r)}
    names, skipped = [], []
    for window in dataset['windows']:
        if window['kind'] != 'detections' or (split != 'all' and window['split'] != split):
            continue
        name = window['window_id']
        expected = range(window['start_frame'],window['end_frame'])
        if not all((name, frame) in frames and human_reviewed(frames[name,frame]) and
                   flag(frames[name,frame]['people_complete']) and flag(frames[name,frame]['identities_complete'])
                   for frame in expected):
            skipped.append(name)
            continue
        truth, remap = [], {}
        for row in detections:
            if row['window_id'] != name or not human_reviewed(row) or row['decision'] != 'accepted' or row['object_type'] != 'player':
                continue
            frame = int(row['frame'])
            identity = identities.get((name,frame,row['annotation_id']))
            if not identity or not identity['identity_id']:
                raise ValueError('Completeness claim lacks an identity: '+name)
            label = identity['identity_id']
            remap.setdefault(label,len(remap)+1)
            x1,y1,x2,y2 = [float(row[k]) for k in ('x1','y1','x2','y2')]
            truth.append([frame-window['start_frame']+1,remap[label],x1,y1,x2-x1,y2-y1,1,1,1])
        source = sources[window['source_id']]
        for frame in expected:
            validate_frame(frames[name,frame],
                [r for r in detections if r['window_id'] == name and int(r['frame']) == frame],
                [r for (wid,f,_),r in identities.items() if wid == name and f == frame],
                source['image_size'])
        predicted = []
        for row in read_rows(Path(source['run'])/'detections.csv'):
            frame = int(row['frame'])
            if row['class_name'] != 'person' or int(row['track_id']) < 0 or frame not in expected:
                continue
            x1,y1,x2,y2 = [float(row[k]) for k in ('x1','y1','x2','y2')]
            predicted.append([frame-window['start_frame']+1,int(row['track_id']),x1,y1,x2-x1,y2-y1,
                              float(row['confidence']),-1,-1,-1])
        gt = output/'gt'/name/'gt'
        gt.mkdir(parents=True)
        (gt/'gt.txt').write_text('\n'.join(','.join(map(str,r)) for r in sorted(truth))+'\n',encoding='utf-8')
        tracking = output/'trackers'/'cached'/'data'
        tracking.mkdir(parents=True,exist_ok=True)
        (tracking/f'{name}.txt').write_text('\n'.join(','.join(map(str,r)) for r in sorted(predicted))+'\n',encoding='utf-8')
        info = configparser.ConfigParser()
        info.optionxform = str
        info['Sequence'] = dict(name=name,imDir='img1',frameRate=str(source['fps']),
                                seqLength=str(window['end_frame']-window['start_frame']),
                                imWidth=str(source['image_size'][0]),imHeight=str(source['image_size'][1]),imExt='.png')
        with (gt.parent/'seqinfo.ini').open('w',encoding='utf-8') as stream:
            info.write(stream)
        names.append(name)
    (output/'seqmap.txt').write_text('name\n'+'\n'.join(names)+'\n',encoding='utf-8')
    import json
    report = dict(exported=names,skipped_incomplete=skipped,metrics=None,
                  note='No reference TrackEval run performed. Exports use 1-based local frames; no predictions or P? IDs.')
    (output/'export_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='results/review_v04')
    parser.add_argument('--output',default='results/v04_trackeval')
    parser.add_argument('--split',choices=['dev','eval','all'],default='eval')
    args=parser.parse_args()
    print(export(args.dataset,args.output,args.split))
