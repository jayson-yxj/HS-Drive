"""Render RGB, versioned road support and directed lane labels for inspection."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from datasets.index import checked_json,safe_path
from datasets.road_supervision import validate_road_arrays
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True);p.add_argument('--overlay',required=True)
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.index);overlay=Path(a.overlay);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    index=json.loads((root/'index.json').read_text());manifest=json.loads((overlay/'manifest.json').read_text())
    if manifest['index_sha256']!=file_hash(root/'index.json'):raise ValueError('Overlay/index mismatch')
    rows=[r for r in index['samples'] if r['scene_id']==a.scene]
    if not rows:raise ValueError('Scene not found')
    # Select by annotation coverage only; this is a supervision preview, not model selection.
    best=None
    for row in rows:
        ref=manifest['samples'][row['sample_id']];path=safe_path(overlay,ref['path'])
        if file_hash(path)!=ref['sha256']:raise ValueError('Overlay checksum')
        with np.load(path,allow_pickle=False) as data:
            validate_road_arrays(data);score=int(data['junction_support'].sum())
            if best is None or score>best[0]:best=(score,row,{k:data[k] for k in data.files})
    score,row,data=best;record=checked_json(root,row['input']);ref=record['cameras']['front']['rgb']
    path=safe_path(index['raw_root'],ref['path'])
    if file_hash(path)!=ref['sha256']:raise ValueError('RGB checksum')
    rgb=cv2.imread(str(path));rgb=cv2.resize(rgb,(960,576))
    label=data['drivable_static'];color=np.full((*label.shape,3),65,np.uint8)
    color[label==0]=(40,40,210);color[label==1]=(50,180,50)
    color[data['junction_support']]=(220,190,20)
    color[data['vertical_ambiguity']]=(170,50,170)
    left=cv2.resize(color[::-1,::-1],(480,640),interpolation=cv2.INTER_NEAREST);right=left.copy()
    for x in range(4,160,8):
        for y in range(4,120,8):
            if not data['lane_direction_valid'][x,y]:continue
            dx,dy=data['lane_direction'][:,x,y];start=(int((119-y)*4+2),int((159-x)*4+2))
            end=(round(start[0]-dy*19),round(start[1]-dx*19))
            cv2.arrowedLine(right,start,end,(255,255,255),1,tipLength=.35)
    header=np.zeros((110,960,3),np.uint8)
    texts=[f"{row['sample_id']} | supervision only | junction cells={score}",
           'Green: static road | Cyan: junction road | Red: non-road | Gray: unknown',
           'Purple: vertical ambiguity | White arrows: valid lane direction',
           'BEV: forward up, ego-left left | static road support does not imply right-of-way']
    for i,line in enumerate(texts):cv2.putText(header,line,(10,24+i*25),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
    title=np.zeros((35,960,3),np.uint8)
    for x,text in [(8,'Road geometry (80m x 60m)'),(488,'Directed lanes; conflicts masked')]:
        cv2.putText(title,text,(x,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    canvas=np.concatenate([header,rgb,title,np.concatenate([left,right],axis=1)])
    out.mkdir(parents=True);cv2.imwrite(str(out/'road_supervision.png'),canvas)
    atomic_json(out/'manifest.json',{'sample_id':row['sample_id'],'index_sha256':file_hash(root/'index.json'),
        'overlay_manifest_sha256':file_hash(overlay/'manifest.json'),'junction_cells':score,
        'selection':'maximum junction label coverage within specified scene; no model outputs used',
        'image_sha256':file_hash(out/'road_supervision.png')})


if __name__=='__main__':main()
