"""Render actual saved training GT and actual preprocessed crops, no model."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from tools.m8j6_prepare_actual import draw_box


def main():
    root=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    manifest=json.loads((root/'manifest.json').read_text())
    out=root/'actual_training_views';out.mkdir(exist_ok=False)
    for frame in manifest['frames']:
        i=frame['index']
        with np.load(root/'rgb_input'/f'{i:02}.npz') as a:
            imgs=a['images'][0].transpose(0,2,3,1)+np.array([103.530,116.280,123.675])
            projections=a['lidar2img']
        with np.load(root/'supervision'/f'{i:02}.npz') as a:gt=a['boxes'];labels=a['labels']
        panels=[]
        for ci,camera in enumerate(manifest['six_camera_order']):
            image=np.ascontiguousarray(np.clip(imgs[ci],0,255).astype(np.uint8))
            for b,label in zip(gt,labels):
                center=b[:3].copy();center[2]+=b[5]/2
                box=Box(center,b[3:6],Quaternion(axis=[0,0,1],angle=-float(b[6])-np.pi/2))
                draw_box(image,box.corners().T,projections[ci],(255,220,0))
            cv2.putText(image,camera+' actual crop + actual TRAIN GT',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            cv2.imwrite(str(out/f'{i:02}_{camera}.jpg'),image)
            panels.append(cv2.resize(image,(704,256)))
        cv2.imwrite(str(out/f'{i:02}_six.jpg'),np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0))
    with (root/'review/index.html').open('a') as f:
        f.write('<h1>Actual normalized-input crops and actual saved training boxes (cyan)</h1>')
        for i in range(8):f.write(f'<p>Frame {i:02}</p><img width="1500" src="../actual_training_views/{i:02}_six.jpg">')
    print('Rendered eight actual training crops; no label or parameter change.')


if __name__=='__main__':main()
