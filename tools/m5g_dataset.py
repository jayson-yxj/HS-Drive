"""Read-only adapter of recorded synchronized RGB/Depth/Instance into isolated pixels."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from datasets.m5b_pairs import camera_rays
from datasets.m5f_pixels import adapt,SCHEMA
from simulator.recorder import atomic_json,file_hash

ROOT='/root/gpufree-data/hs-drive-data/processed/m5g_pixels'


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',nargs='+',default=['/root/gpufree-data/hs-drive-data/processed/m5d_index','/root/gpufree-data/hs-drive-data/processed/m5e_index']);p.add_argument('--output',default=ROOT);p.add_argument('--splits',nargs='+',default=['train','val']);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);records=[];audits=[];assets={}
    for index in a.indices:
        rows=json.loads(Path(index,'index.json').read_text())['samples'];assets[str(Path(index,'index.json'))]=file_hash(Path(index,'index.json'))
        for row in rows:
            if row['split'] not in a.splits:continue
            raw=Path(row['raw_frame']).parent.parent;man=json.loads((raw/'manifest.json').read_text());cal=json.loads((raw/'calibration.json').read_text());frame=json.loads(Path(row['raw_frame']).read_text());arrays={};actors=[];stats=[]
            for f in [raw/'manifest.json',raw/'calibration.json',Path(row['raw_frame'])]:assets[str(f)]=file_hash(f)
            assert man['calibration_sha256']==file_hash(raw/'calibration.json')
            for cam in row['camera_names']:
                r,d=camera_rays(raw,man,frame,cam);K=cal['cameras'][cam]['K'];T=cal['cameras'][cam]['T_hs_ego_from_opt'];label,actor=adapt(*d,r['self_mask'],K,T,frame['actors'])
                for k in ['target','valid','classes','actor','depth','radial']:arrays.setdefault(k,[]).append(label[k])
                entry=frame['cameras'][cam];rgb=cv2.imread(str(raw/entry['rgb']));assert rgb.shape==(384,640,3) and d[0].shape==(384,640)
                for k in ['rgb','depth','instance']:assert file_hash(raw/entry[k])==entry[k+'_sha256'];assets[str(raw/entry[k])]=entry[k+'_sha256']
                # Edge alignment is a diagnostic proxy; texture and reflection can also create RGB edges.
                gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,60,120);near=cv2.dilate(edges,np.ones((5,5),np.uint8))>0;bound=~label['interior']&~r['self_mask'];known_bound=bound&np.isin(d[1],[12,13,14,15,16,17,18,19,21]);tags,cnt=np.unique(d[1],return_counts=True)
                stats.append(dict(camera=cam,valid=int(label['valid'].sum()),positive=int(label['target'].sum()),negative=int((label['valid']&~label['target']).sum()),self_ignored=int(r['self_mask'].sum()),boundary_ignored=int((~label['interior']).sum()),semantic_histogram=dict(zip(map(str,tags.tolist()),cnt.tolist())),target_boundary_pixels=int(known_bound.sum()),boundary_near_rgb_edge_fraction=float(near[known_bound].mean()) if known_bound.any() else None))
                actors.append(dict(camera=cam,actors=actor))
            path=out/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(path,**{k:np.array(v) for k,v in arrays.items()})
            row=dict(row,input=str(Path(index,row['input']).resolve()))
            rec=dict(row,pixel_label=str(path),pixel_sha256=file_hash(path),actors=actors,weather=man.get('weather'),geometry=man.get('geometry'));records.append(rec);audits.append(dict(sample_id=row['sample_id'],group=row['group'],split=row['split'],configuration=row['configuration'],cameras=stats,actors=actors));print(row['sample_id'],sum(s['positive'] for s in stats),flush=True)
    atomic_json(out/'index.json',dict(schema=SCHEMA,records=records,label_source_sha256=file_hash('datasets/m5f_pixels.py'),source_sha256=file_hash(__file__)))
    atomic_json(out/'quality.json',dict(kind='offline_actual_recorded_CARLA_pixel_label_audit',records=audits,source_assets_sha256=assets,contract=SCHEMA,sync='Capture code asserts each image.frame/timestamp equals selected snapshot; old files store shared frame timestamp, not all individual sensor stamps. Identical K/T/dimensions checked. No independent per-pixel alignment truth.',limits='RGBedge agreement only proxy. Transparent/reflection target semantics may differ from appearance; boundary1px ignored but interiorglass not fully separatelyidentified. AABB/renderID association conservativeproxy, not assumedActorID. Unknown/Static/Dynamic unrelated retained Ignore. Oldtests excluded.'))

if __name__=='__main__':main()
