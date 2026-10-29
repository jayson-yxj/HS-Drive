"""Private first/middle/last per-scene frozen A/Stream projections; no model."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from tools.m8j7_gpu import draw_corners
from training.m8j7_metrics import prediction_mask,image_panels

def main():
    c=json.loads(Path('configs/m8j11_protocol.json').read_text());root=Path(c['private_root']);out=root/'visuals';out.mkdir(exist_ok=False)
    base=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab');m=json.loads((base/'prepared/manifest.json').read_text())
    html='<meta charset="utf-8"><h1>Private M8-J11 fixed scene snapshots</h1><p>GT green, A orange, StreamPETR cyan; fixed score0.2. No human signoff.</p>'
    for i in (0,20,39,40,60,80):
        with np.load(base/'prepared/rgb_input'/f'{i:02}.npz',allow_pickle=False) as z:images=image_panels(z['images']);P=z['lidar2img']
        with np.load(base/'prepared/supervision'/f'{i:02}.npz',allow_pickle=False) as z:gt=z['boxes'].copy()
        arrays=[(gt,(0,180,0))]
        for path,color in [(base/'A/mini_val'/f'{i:02}.npz',(0,140,255)),(root/'full'/f'{i:02}.npz',(220,180,0))]:
            with np.load(path,allow_pickle=False) as z:b=z['boxes'];b=b[prediction_mask(b,z['scores'],.2)]
            arrays.append((b,color))
        corners=[(LiDARInstance3DBoxes(torch.from_numpy(b.copy()),box_dim=9).corners.numpy(),col) for b,col in arrays]
        panels=[];bev=np.full((1000,1000,3),245,np.uint8)
        for ci,name in enumerate(m['six_camera_order']):
            im=np.ascontiguousarray(np.clip(images[ci],0,255).astype(np.uint8))
            for boxes,col in corners:draw_corners(im,boxes,P[ci],col)
            cv2.putText(im,name,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(cv2.resize(im,(704,256)))
        for boxes,col in corners:
            for b in boxes:
                xy=b[[0,3,7,4],:2];pixel=np.stack([500-xy[:,1]*8,500-xy[:,0]*8],1).round().astype(np.int32)
                cv2.polylines(bev,[pixel],True,col,1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'{i:02}_six.jpg'),np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0));cv2.imwrite(str(out/f'{i:02}_bev.jpg'),bev)
        html+=f'<h2>Frame{i}</h2><img width="1500" src="{i:02}_six.jpg"><img width="650" src="{i:02}_bev.jpg">'
    (out/'index.html').write_text(html)
    print(out)
if __name__=='__main__':main()
