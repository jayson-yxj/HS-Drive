"""Compare static Drivable to a train-only BEV-location prior on identical masks."""
import argparse,json
from pathlib import Path
import numpy as np
from datasets.index import checked_json,safe_path
from datasets.drivable import validate_arrays
from simulator.recorder import atomic_json,file_hash


def labels(index,root,row,overlay=None):
    if overlay is not None:
        directory,manifest=overlay;ref=manifest['samples'][row['sample_id']];path=safe_path(directory,ref['path'])
        if file_hash(path)!=ref['sha256'] or row['raw_frame_sha256']!=ref['raw_frame_sha256']:raise ValueError('Road prior sample mismatch')
        with np.load(path,allow_pickle=False) as data:
            from datasets.road_supervision import validate_road_arrays
            validate_road_arrays(data);return data['drivable_static'],data['drivable_valid']
    target=checked_json(root,row['target']);ref=target['drivable']
    path=safe_path(index['raw_root'],ref['path'])
    if file_hash(path)!=ref['sha256']:raise ValueError('Label checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        label=data['drivable_static'];valid=data['drivable_valid'];rules=data['drivable_rule_valid']
    validate_arrays(label,valid,rules)
    return label,valid


def metrics(confusion):
    tn,fp,fn,tp=confusion.ravel().tolist();known=tn+fp+fn+tp
    return {'confusion_tn_fp_fn_tp':confusion.tolist(),'valid_cells':known,
        'iou_positive':tp/(tp+fp+fn) if tp+fp+fn else None,
        'all_positive_iou':(tp+fn)/known if known else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--model-metrics',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();manifest=json.loads((Path(a.run)/'manifest.json').read_text());root=Path(manifest['config']['index'])
    if file_hash(root/'index.json')!=manifest['index_sha256']:raise ValueError('Dataset changed')
    index=json.loads((root/'index.json').read_text());chosen=set(manifest['train_sample_ids'])
    overlay=None
    if manifest['config'].get('road_overlay'):
        directory=Path(manifest['config']['road_overlay']);om=json.loads((directory/'manifest.json').read_text())
        if file_hash(directory/'manifest.json')!=manifest['road_overlay_manifest_sha256'] or om['index_sha256']!=manifest['index_sha256']:raise ValueError('Road prior provenance mismatch')
        overlay=(directory,om)
    sums=np.zeros((160,120),np.float64);counts=np.zeros_like(sums);seen=set()
    for row in index['samples']:
        if row['sample_id'] not in chosen:continue
        if row['split']!='train':raise ValueError('Prior attempted to fit on held-out data')
        label,valid=labels(index,root,row,overlay);sums+=(label==1)&valid;counts+=valid;seen.add(row['sample_id'])
    if seen!=chosen or not counts.sum():raise ValueError('Missing training supervision')
    global_prior=float(sums.sum()/counts.sum())
    probability=np.divide(sums,counts,out=np.full_like(sums,global_prior),where=counts>0)
    pred=probability>=.5;confusion=np.zeros((2,2),np.int64);scenes={};sample_ids=[]
    for row in index['samples']:
        if row['split']!=a.split:continue
        label,valid=labels(index,root,row,overlay)
        matrix=np.bincount(label[valid].astype(int)*2+pred[valid],minlength=4).reshape(2,2)
        confusion+=matrix;scenes.setdefault(row['scene_id'],np.zeros((2,2),np.int64))[:]+=matrix
        sample_ids.append(row['sample_id'])
    model=json.loads(Path(a.model_metrics).read_text())
    if (model['sample_ids']!=sample_ids or model['index_sha256']!=manifest['index_sha256'] or
            model['checkpoint_sha256']!=manifest['best_sha256'] or model['drivable_static']['valid_cells']!=int(confusion.sum())):
        raise ValueError('Model/prior sample, mask or identity mismatch')
    report={'schema_version':'hsd-static-drivable-prior-v1','split':a.split,'samples':len(sample_ids),
        'prior':metrics(confusion),'prior_per_scene':{k:metrics(v) for k,v in scenes.items()},
        'model':model['drivable_static'],'model_beats_location_prior':model['drivable_static']['iou_positive']>metrics(confusion)['iou_positive'],
        'unseen_training_cells':int((counts==0).sum()),'training_positive_fraction':global_prior,
        'train_samples':len(chosen),'index_sha256':manifest['index_sha256'],'model_metrics_sha256':file_hash(a.model_metrics),
        'scope':'Fixed BEV-cell positive frequency from actual training subset only; unseen cells fall back to global train frequency; no RGB; identical evaluation masks'}
    atomic_json(a.output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
