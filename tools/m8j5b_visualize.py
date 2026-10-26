"""Private six-camera coverage viewer and four source-gap comparison figures."""
import json
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from tools.m8j3_validate_clip import RAW
from tools.m8j5b_build_coverage import ROOT,SOURCE,save


def draw(im,projection,color,label):
    if projection.get('status')!='Valid':return
    pts=np.rint(projection['polygon']).astype(np.int32)
    cv2.polylines(im,[pts],True,color,2)
    cv2.putText(im,label,tuple(np.maximum(pts.min(0),[0,20])),0,.6,color,2)


def main():
    web=ROOT/'web';web.mkdir(exist_ok=False);(web/'images').mkdir();data=[]
    for n in ('00017','00107','00124','00196','00213'):
        source=json.loads((SOURCE/'frames'/(n+'.json')).read_text())
        side=json.loads((ROOT/'sidecar'/(n+'.json')).read_text());qual={o['actor_id']:o for o in side['objects']}
        images={}
        for cam in CAMERAS:
            images[cam]={}
            for kind in ('rgb','semantic','instance'):
                ext='.jpg' if kind=='rgb' else '.png';dest=web/'images'/(n+'_'+cam+'_'+kind+ext)
                dest.symlink_to(RAW/'camera'/(kind+'_'+VIEW[cam])/(n+ext));images[cam][kind]='images/'+dest.name
            mask=cv2.imread(str(ROOT/'views'/(n+'_'+cam+'_ignore.png')),0)
            rgba=np.zeros((*mask.shape,4),np.uint8);rgba[:,:,:3]=[220,30,210];rgba[:,:,3]=(mask>0).astype(np.uint8)*160
            dest=web/'images'/(n+'_'+cam+'_mask.png');cv2.imwrite(str(dest),rgba);images[cam]['mask']='images/'+dest.name
        objects=[]
        for o in source['objects']:
            if o['source_class']=='ego_vehicle':continue
            objects.append(dict(actor_id=o['actor_id'],source_class=o['source_class'],views=o['views'],
                positive=qual.get(o['actor_id'],{}).get('positive',dict(eligible=False,reasons=['NotDetectionClass'])),
                geometry=o['converted']['geometry_status']))
        data.append(dict(ordinal=n,images=images,objects=objects,views=side['views'],negative_enabled=False,
            positive_count=sum(o['positive']['eligible'] for o in objects)))
    template=Path('tools/m8j5b_coverage_ui.html').read_text()
    (web/'index.html').write_text(template.replace('__DATA__',json.dumps(data).replace('<','\\u003c')))
    cases=json.loads((ROOT/'case_reproduction.json').read_text());contacts=[]
    for i,c in enumerate(cases):
        n,cam=c['ordinal'],c['camera'];src=json.loads((SOURCE/'frames'/(n+'.json')).read_text())
        im=cv2.imread(str(RAW/'camera'/('rgb_'+VIEW[cam])/(n+'.jpg')));raw=im.copy();converted=im.copy()
        s=json.loads((ROOT/'sidecar'/(n+'.json')).read_text());eligible={o['actor_id'] for o in s['objects'] if o['positive']['eligible']}
        for o in src['objects']:
            if o['source_class']=='ego_vehicle':continue
            v=o['views'].get(cam,{})
            draw(raw,v.get('raw_projection',{}),(0,180,255),o['actor_id']+' raw')
            draw(converted,v.get('converted_projection',{}),(255,170,0),o['actor_id']+' converted')
            if o['actor_id'] in eligible:draw(converted,v.get('converted_projection',{}),(0,200,0),o['actor_id']+' P-diagnostic')
        local=im.copy();mask=cv2.imread(str(ROOT/'views'/(n+'_'+cam+'_ignore.png')),0)>0
        local[mask]=(local[mask].astype(float)*.4+np.array([220,30,210])*.6).astype(np.uint8)
        b=c['region']['bbox_xyxy'];cv2.rectangle(local,tuple(b[:2]),tuple(b[2:]),(255,255,255),3)
        # Frame fallback visualization is independent of local mask coverage.
        frame=cv2.addWeighted(im,.8,np.full_like(im,(30,30,180)),.2,0)
        panels=[]
        for label,img in [('ALL RAW GT',raw),('CONVERTED + P CANDIDATES',converted),('LOCAL IGNORE (unknown depth)',local),('FRAME: ALL UNMATCHED IGNORE; N=0',frame)]:
            img=cv2.resize(img,(800,450));cv2.rectangle(img,(0,0),(800,35),(20,20,20),-1);cv2.putText(img,label,(10,24),0,.58,(255,255,255),1);panels.append(img)
        combined=np.concatenate([np.concatenate(panels[:2],1),np.concatenate(panels[2:],1)],0)
        cv2.imwrite(str(ROOT/'cases'/('coverage_case_%02d.jpg'%i)),combined);contacts.append(cv2.resize(combined,(800,450)))
    cv2.imwrite(str(ROOT/'cases/coverage_contact.jpg'),np.concatenate(contacts,0))
    save('reports/m8j5b_visualization.json',dict(private_page=str(web/'index.html'),frames=5,camera_views=30,
        missing_cases=4,layers=['AllRawGT','AllConvertedGT','DiagnosticPositive','MissingRegion','LocalIgnore','FrameNegativeFallback'],
        human_signoff='Pending_Suspended',source_image_publish=False,automatic_visual_signoff=False))


if __name__=='__main__':main()
