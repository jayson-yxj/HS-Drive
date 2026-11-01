"""Six prespecified private comparison groups; no model or review UI."""
from pathlib import Path
import cv2
import numpy as np
import torch
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from tools.m8j7_gpu import draw_corners
from training.m8j7_metrics import prediction_mask,image_panels
from tools.m8j11a_common import read,write,load_cache,sha,PROTOCOL


def main():
    c=read(PROTOCOL);root=Path(c['private_root']);out=root/'visuals';out.mkdir(exist_ok=False)
    base=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared')
    cameras=read(base/'manifest.json')['six_camera_order'];html='<meta charset="utf-8"><h1>Private temporal sensitivity comparisons</h1><p>Fixed predeclared frames, GT green, predictions orange. Left T0, right intervention. No human signoff.</p>'
    generated=[]
    assert len(c['visual_groups'])<=6
    for group in c['visual_groups']:
        i=group['index'];t=group['condition'];source=load_cache(base/'rgb_input'/('{:02}.npz'.format(i)))
        images=image_panels(source['images']);projections=source['lidar2img'];gt=load_cache(base/'supervision'/('{:02}.npz'.format(i)))['boxes']
        pair=[];bevs=[]
        for name,path in [('T0',c['T0_cache'][i]['path']),(t,root/t/('{:02}.npz'.format(i)))]:
            r=load_cache(path);b=r['boxes'][prediction_mask(r['boxes'],r['scores'],.2)]
            corners=[(LiDARInstance3DBoxes(torch.from_numpy(a.copy()),box_dim=9).corners.numpy(),color) for a,color in [(gt,(0,180,0)),(b,(0,140,255))]]
            panels=[];bev=np.full((800,800,3),245,np.uint8)
            for ci,cam in enumerate(cameras):
                im=np.ascontiguousarray(np.clip(images[ci],0,255).astype(np.uint8))
                for boxes,color in corners:draw_corners(im,boxes,projections[ci],color)
                cv2.putText(im,name+' '+cam,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
                panels.append(cv2.resize(im,(528,192)))
            for boxes,color in corners:
                for box in boxes:
                    xy=box[[0,3,7,4],:2];pix=np.stack([400-xy[:,1]*6,400-xy[:,0]*6],1).round().astype(np.int32)
                    cv2.polylines(bev,[pix],True,color,1,cv2.LINE_AA)
            cv2.putText(bev,name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
            pair.append(np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0));bevs.append(bev)
        prefix='{}_{:02}'.format(t,i);p=out/(prefix+'_six.jpg');b=out/(prefix+'_bev.jpg')
        cv2.imwrite(str(p),np.concatenate(pair,1));cv2.imwrite(str(b),np.concatenate(bevs,1))
        generated.append(dict(group=group,six_path=str(p),six_sha256=sha(p),bev_path=str(b),bev_sha256=sha(b)))
        html+='<h2>{}</h2><img width="1584" src="{}_six.jpg"><img width="1000" src="{}_bev.jpg">'.format(prefix,prefix,prefix)
    (out/'index.html').write_text(html)
    write(root/'visuals_manifest.json',dict(groups=generated,selection='Fixed before forwards; no prediction-driven selection',review_status='Generated_NotHumanSigned'))
    print(out)

if __name__=='__main__':main()
