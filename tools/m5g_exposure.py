"""Audit actual saved camera update schedules, split geometry, and old learning curves."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m5f_dataset import ROOT
from tools.m5f_common import pixel_label
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5g_training_exposure.json')
    if out.exists():raise FileExistsError(out)
    rows=json.loads(Path(ROOT,'index.json').read_text())['records'];meta=[];poses={};hashes={};records=[]
    for i,r in enumerate(rows):
        d=pixel_label(r);frame=json.loads(Path(r['raw_frame']).read_text());poses[r['group']]=dict(split=r['split'],map=r['group'].split('_')[0],pose=np.array(frame['T_world_from_ego'])[:3,3].tolist());hashes[r['sample_id']]=r['input_sha256']
        for c,name in enumerate(r['camera_names']):
            t=d['target'][c];v=d['valid'][c];meta.append(dict(index=i,camera_index=c,sample_id=r['sample_id'],group=r['group'],split=r['split'],configuration=r['configuration'],camera=name,positive=int(t.sum()),negative=int((v&~t).sum()),classes={str(k):int((t&(d['classes'][c]==k)).sum())for k in [1,2,3]},distance={f'{lo}_{hi}':int((t&(d['radial'][c]>=lo)&(d['radial'][c]<hi)).sum())for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]},small_actor_count=sum(3<=a['pixels']<=32 for a in r['actors'][c]['actors'])))
    lookup={(r['index'],r['camera_index']):r for r in meta};train=[r for r in meta if r['split']=='train']
    for seed in [101,102]:
        path=Path(f'runs/m5f_learning/m5f_schedule_{seed}.json');schedule=json.loads(path.read_text())['schedule'];counts=Counter(tuple(x)for x in schedule);selected=[lookup[tuple(x)]for x in schedule];assert all(r['split']=='train'for r in selected)
        records.append(dict(seed=seed,steps=len(schedule),available_views=len(train),unique_views=len(counts),unseen_views=len(train)-len(counts),coverage=len(counts)/len(train),repeated_view_updates=len(schedule)-len(counts),unique_layout_frames=len({r['sample_id']for r in selected}),positive_image_updates=sum(r['positive']>0 for r in selected),background_only_updates=sum(r['positive']==0 for r in selected),by_scene=dict(Counter(r['group']for r in selected)),by_camera={name:dict(updates=sum(r['camera']==name for r in selected),positive_updates=sum(r['camera']==name and r['positive']>0 for r in selected),available_positive_views=sum(r['camera']==name and r['positive']>0 for r in train))for name in rows[0]['camera_names']},class_updates={str(k):sum(r['classes'][str(k)]>0 for r in selected)for k in [1,2,3]},distance_updates={key:sum(r['distance'][key]>0 for r in selected)for key in selected[0]['distance']},small_actor_updates=sum(r['small_actor_count']>0 for r in selected),exposure=[dict(sample_id=r['sample_id'],camera=r['camera'],updates=counts[(r['index'],r['camera_index'])],positive=r['positive'])for r in train],schedule_sha256=file_hash(path)))
    distances=[]
    for a,pa in poses.items():
        for b,pb in poses.items():
            if pa['split']=='train'and pb['split']=='val'and pa['map']==pb['map']:distances.append(dict(train=a,val=b,distance_m=float(np.linalg.norm(np.array(pa['pose'])-pb['pose']))))
    old=json.loads(Path('reports/m5f_image_foreground_training.json').read_text());curves=[]
    for r in old['records']:
        h=[x for x in r['history']if 'validation'in x];curves.append(dict(name=r['name'],seed=r['seed'],history=r['history'],valAP=[dict(step=x['step'],meanAP=float(np.mean([s['histogram_ap']or 0 for s in x['validation']['scenes'].values()])))for x in h]))
    atomic_json(out,dict(kind='offline_actual_M5F_saved_schedules_and_labels',records=records,available_metadata=meta,split_pose_distances=distances,min_train_val_distance_m=min(r['distance_m']for r in distances),identical_cross_split_input_hashes=len(set(r['input_sha256']for r in rows if r['split']=='train')&set(r['input_sha256']for r in rows if r['split']=='val')),historical_curves=curves,source_sha256=file_hash(__file__),index_sha256=file_hash(Path(ROOT,'index.json')),limits='Schedules record updateexposure not independentdata. Oldcheckpoints onlybest, oldValLoss andTrainAP notrecorded cannot reconstruct perstep; newmatched reruns recordboth. OldM5F pixelPRfigure reversed histogram class order, JSONmetrics correct; M5G usespositivefirst. Sameassetlayouts reused acrossdifferentlocations, notindependentlayoutgeneralization.'))
    print([(r['seed'],r['unique_views'],r['unseen_views'],r['positive_image_updates'])for r in records],flush=True)

if __name__=='__main__':main()
