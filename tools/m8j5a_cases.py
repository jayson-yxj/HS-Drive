"""Reproduce omitted-visible examples with original candidate-only vs all-GT."""
import json
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import VIEW
from datasets.m8j4_gt import transform
from geometry.reference import OPTICAL_TO_CARLA_CAM as O
from tools.m8j3_validate_clip import RAW,read_annotation
from tools.m8j5a_audit import OUT,save,report


def box_draw(im,p,label,color):
    if p['status']!='Valid':return
    poly=np.round(p['polygon']).astype(np.int32);cv2.polylines(im,[poly],True,color,2)
    x,y=poly.min(0);cv2.putText(im,label,(x,max(20,y-6)),0,.6,color,2)


def main():
    directory=OUT/'cases';directory.mkdir(exist_ok=False)
    rs=json.loads((OUT/'render_regions.json').read_text())
    pack=json.loads(Path('/root/gpufree-data/hs-drive-private/m8j5-supervision/golden_candidates.json').read_text())
    frame_candidates=defaultdict(list)
    for c in pack['candidates']:frame_candidates[c['ordinal']].append(c['object']['actor_id'])
    selected=[];used=set()
    # Start with exact user reproduction: 394 page hides rear car 393.
    first=next(r for r in rs if r['ordinal']=='00017' and r['camera']=='CAM_BACK' and r['pixels']>100)
    selected.append((first,'394'));used.add('393')
    # Distinct raw actors where another selected candidate hid the visible box.
    eligible=sorted([r for r in rs if r['ordinal'] in frame_candidates and r['pixels']>=300 and r['best_raw_overlap_fraction']>=.7],key=lambda r:-r['pixels'])
    for r in eligible:
        aid=r['raw_box_overlap_candidates'][0]['actor_id']
        focus=next((a for a in frame_candidates[r['ordinal']] if a!=aid),None)
        if focus and aid not in used:
            selected.append((r,focus));used.add(aid)
    # Three unexplained vehicle render groups and the disappearance pedestrian.
    for semantic,token in [(14,45980),(14,45976),(14,48630),(12,31465)]:
        rr=[r for r in rs if r['semantic']==semantic and r['render_token']==token and r['coverage'].startswith('Unexplained')]
        r=max(rr,key=lambda x:x['pixels']);selected.append((r,None))
    details=[];contact=[]
    for i,(region,focus) in enumerate(selected):
        n=region['ordinal'];cam=region['camera'];frame=json.loads((OUT/'frames'/(n+'.json')).read_text())
        rgb=cv2.imread(str(RAW/'camera'/('rgb_'+VIEW[cam])/(n+'.jpg')))
        before=rgb.copy();after=rgb.copy();seg=rgb.copy()
        for obj in frame['objects']:
            if cam not in obj['views'] or obj['source_class']=='ego_vehicle':continue
            p=obj['views'][cam]['raw_projection']
            box_draw(after,p,obj['actor_id'],(0,210,255))
            if obj['actor_id']==focus:box_draw(before,p,obj['actor_id'],(0,210,255))
        ins=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(n+'.png')),-1)
        semantic=cv2.imread(str(RAW/'camera'/('semantic_'+VIEW[cam])/(n+'.png')),-1)
        mask=(ins[:,:,2]==region['semantic']) & ((ins[:,:,1].astype(int)+256*ins[:,:,0].astype(int))==region['render_token'])
        seg[mask]=(seg[mask]*.35+np.array([220,30,220])*.65).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(after,contours,-1,(200,0,230),1)
        cid='case_%02d'%i
        for title,im in [('before_candidate_only',before),('all_raw',after),('instance_evidence',seg)]:
            cv2.imwrite(str(directory/(cid+'_'+title+'.jpg')),im)
        x0,y0,x1,y1=region['bbox_xyxy'];w=x1-x0;h=y1-y0
        x0=max(0,x0-max(30,w//2));y0=max(0,y0-max(30,h//2));x1=min(1600,x1+max(30,w//2));y1=min(900,y1+max(30,h//2))
        tiles=[]
        for im in (before,after,seg):
            crop=im[y0:y1,x0:x1];s=min(440/crop.shape[1],250/crop.shape[0]);small=cv2.resize(crop,(int(crop.shape[1]*s),int(crop.shape[0]*s)))
            tile=np.full((300,440,3),245,np.uint8);tile[40:40+small.shape[0],:small.shape[1]]=small;tiles.append(tile)
        row=np.concatenate(tiles,1)
        cv2.putText(row,cid+' '+n+' '+cam+' token '+str(region['render_token']),(8,24),0,.55,(0,0,0),1)
        cv2.imwrite(str(directory/(cid+'_comparison.jpg')),row);contact.append(row)
        hits=region['raw_box_overlap_candidates'];best=hits[0] if hits else None
        corresponding=next((o for o in frame['objects'] if best and o['actor_id']==best['actor_id']),None)
        missing=region['coverage'].startswith('Unexplained')
        if missing:
            cause='RawWalkerAbsentWithVisibleSemanticInstance' if region['semantic']==12 else 'VisibleVehicleRenderRegionWithoutRawSpatialBoxSupport'
            assertion='ConfirmedFrameAnnotationGap_IdentityCausalityUnknown' if region['semantic']==12 else 'DatasetReferenceCoverageGap_StaticMeshOrOtherSourceUnconfirmed'
        else:cause='CandidateOnlyVisualizationHidOtherRawGT';assertion='DisplayBug_OriginalAndConvertedGTExist'
        details.append(dict(case_id=cid,ordinal=n,camera=cam,previous_focus_actor=focus,region=region,
            raw_class_count=sum(o['source_class']==('walker' if region['semantic']==12 else 'vehicle') for o in frame['objects']),
            raw_actor_id=best['actor_id'] if best else None,source_lineage=corresponding['lineage'] if corresponding else None,
            filters=corresponding['filter_reasons'] if corresponding else [],cause=cause,conclusion=assertion,
            semantic_pixels=int(np.count_nonzero(semantic==region['semantic'])),instance_pixels=int(mask.sum()),
            pixel_semantic_agreement=float(np.count_nonzero(mask&(semantic==region['semantic']))/max(1,mask.sum())),
            assistant_visual_inspection='Pending',physical_actor_identity='NotCertified',training_negative=False))
    for j in range(0,len(contact),4):cv2.imwrite(str(directory/('contact_%02d.jpg'%(j//4))),np.concatenate(contact[j:j+4]))
    save(directory/'case_records.json',details)
    save(report('m8j5a_case_summary.json'),dict(case_count=len(details),
        cases_by_cause=dict(Counter(d['cause'] for d in details)),
        distinct_raw_actor_ids=len({d['raw_actor_id'] for d in details if d['raw_actor_id']}),
        unexplained_render_groups=len({(d['region']['semantic'],d['region']['render_token']) for d in details if not d['raw_actor_id']}),
        independence='Cases are frame/view observations; same actor may recur; render groups not verified unique actors',
        human_signoff='Suspended',private_records=str(directory/'case_records.json')))
    print([(d['case_id'],d['ordinal'],d['camera'],d['raw_actor_id'],d['cause']) for d in details])


if __name__=='__main__':main()
