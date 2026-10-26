"""Build private 214-frame single-frame supervision and review views."""
import hashlib
import json
import os
import time
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from datasets.m8j4_gt import box_to_hs,project_box,regions,associate,enforce_one_to_one
from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from tools.m8j3_validate_clip import read_annotation,RAW

CFG=json.loads(Path('configs/m8j4_protocol.json').read_text())
OUT=Path(CFG['private_root'])


def save(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def render(frame,folder):
    ordinal=frame['ordinal'];panels=[];maskpanels=[]
    for cam in CAMERAS:
        img=cv2.imread(str(RAW/'camera'/('rgb_'+VIEW[cam])/(ordinal+'.jpg')))
        ins=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(ordinal+'.png')),-1)
        display=ins[:,:,:3].copy()
        for obj in frame['objects']:
            v=obj.get('views',{}).get(cam,{})
            p=v.get('projection',{})
            if p.get('status')!='Valid':continue
            x0,y0,x1,y1=np.round(p['bbox_xyxy']).astype(int)
            color=(0,210,0) if v['status']=='Valid' else (0,180,255)
            for target in (img,display):
                cv2.rectangle(target,(x0,y0),(x1,y1),color,2)
                text=f"{obj['actor_id']} {obj.get('model_class')} {v['status']}"
                cv2.putText(target,text,(x0,max(18,y0)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
        for target,output in [(img,panels),(display,maskpanels)]:
            target=cv2.resize(target,(480,270))
            cv2.putText(target,ordinal+' '+cam,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            output.append(target)
    def combine(arr):return np.concatenate([np.concatenate(arr[:3],1),np.concatenate(arr[3:],1)],0)
    cv2.imwrite(str(folder/(ordinal+'_rgb.jpg')),combine(panels))
    cv2.imwrite(str(folder/(ordinal+'_instance.jpg')),combine(maskpanels))


def main():
    os.umask(0o077);OUT.mkdir(exist_ok=False)
    (OUT/'sidecar').mkdir();(OUT/'review').mkdir()
    start=time.perf_counter()
    stats=Counter();classes=defaultdict(Counter);assoc=Counter();ranges=defaultdict(Counter)
    anomalies=[];frames=[];render_links=defaultdict(set);representatives={}
    for f in sorted((RAW/'anno').glob('*.json.gz')):
        a=read_annotation(f);ordinal=f.name.split('.')[0]
        ego=next(o for o in a['bounding_boxes'] if o['class']=='ego_vehicle')
        source_objects={str(o['id']):o for o in a['bounding_boxes'] if o['class']!='ego_vehicle'}
        objects=[box_to_hs(o,np.asarray(ego['world2ego'])) for o in source_objects.values()]
        for o in objects:
            o['views']={}
            if o['geometry_status']=='Valid' and o['center_hs_m'][2]<CFG['association']['below_ego_diagnostic_m']:
                o.update(label_status='Invalid',reason='HiddenBelowGround_NoTrainingTarget')
                anomalies.append(dict(ordinal=ordinal,actor_id=o['actor_id'],center_hs_m=o['center_hs_m'],
                    geometry_consistent=True,action='ExcludePositiveGT_KeepRecord',manual_review='Pending'))
        for cam in CAMERAS:
            s=a['sensors'][cam]
            inst=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(ordinal+'.png')),-1)
            sem=cv2.imread(str(RAW/'camera'/('semantic_'+VIEW[cam])/(ordinal+'.png')),-1)
            labels,tokens,rs=regions(inst,sem)
            views={}
            for o in objects:
                if o['geometry_status']!='Valid':continue
                original=source_objects[o['actor_id']]
                p=project_box(original['world_cord'],np.asarray(s['world2cam']),np.asarray(s['intrinsic']),
                              s['image_size_x'],s['image_size_y'],CFG['association']['near_plane_m'])
                if o.get('label_status')=='Invalid':
                    v=dict(status='Invalid',reason=o['reason'],match=None)
                else:v=associate(p,o['source_class'],labels,tokens,rs,CFG['association'])
                v['projection']=p;views[o['actor_id']]=v
            enforce_one_to_one(views)
            for o in objects:
                if o['actor_id'] in views:o['views'][cam]=views[o['actor_id']]
        for o in objects:
            statuses=[v['status'] for v in o['views'].values()]
            supported=[v for v in o['views'].values() if v['status']=='Valid']
            if 'label_status' not in o:
                o['label_status']='Valid' if supported else ('Invalid' if o['geometry_status']=='Invalid' else 'Unknown')
                o['reason']='RenderAssociationEvidenceSupported' if supported else 'UnsupportedOrVisibilityUnknown'
            o['human_review']='Required_NotCertified'
            o['positive_supervision_candidate']=o['label_status']=='Valid' and o.get('yaw_only_comparable',False) and o.get('model_class') in ('car','pedestrian')
            stats['object_'+o['label_status']]+=1;classes[o.get('model_class',o['source_class'])][o['label_status']]+=1
            if o['geometry_status']=='Valid':
                dist=o['distance_xy_m'];band='0-10' if dist<10 else '10-20' if dist<20 else '20-40' if dist<40 else '40+'
                ranges[band][o['label_status']]+=1
                for cam,v in o['views'].items():
                    assoc[v['status']]+=1
                    stats['projection_'+v['projection']['status']]+=1
                    if v['projection'].get('truncated'):stats['truncated_views']+=1
                    if v['projection']['status']=='Valid' and o.get('reason')!='HiddenBelowGround_NoTrainingTarget':
                        stats['eligible_view_'+v['status']]+=1
                    if v['status']=='Valid':
                        m=v['match'];render_links[(m['semantic'],m['render_token'])].add(o['actor_id'])
                        key=(band,v.get('occlusion_state',''),o['source_class'])
                        representatives.setdefault(key,ordinal)
            if o['positive_supervision_candidate']:stats['positive_candidates']+=1
        frame=dict(schema='m8j4.gt-sidecar.v1',ordinal=ordinal,source_frame=None,timestamp_s=None,
            source='Bench2Drive_GT_SupervisionOnly',coordinate='HS_ego_x_forward_y_left_z_up',
            box_reference='ActorBoxCenter',dimensions='length_width_height_full_extent',
            yaw='CCW_from_HS_forward_radians',objects=objects,
            background_negative_valid=False,occupancy_free_valid=False,
            instance_association='RenderTokenGeometryAssociation_NotActorNumericEquality',
            per_sensor_sync='Unknown',human_review='Pending')
        save(OUT/'sidecar'/(ordinal+'.json'),frame);frames.append(frame)
        if any(o['positive_supervision_candidate'] for o in objects):stats['frames_with_positive_candidates']+=1
        if int(ordinal)%40==0:print(ordinal,dict(stats),flush=True)
    uniform=[f'{i:05}' for i in np.linspace(0,213,24,dtype=int)]
    anomaly_frames=sorted({x['ordinal'] for x in anomalies})
    chosen=sorted(set(uniform+anomaly_frames+list(representatives.values())))
    lookup={f['ordinal']:f for f in frames}
    for ordinal in chosen:render(lookup[ordinal],OUT/'review')
    for label,ordinals in [('uniform',uniform),('anomalies',anomaly_frames)]:
        for i in range(0,len(ordinals),4):
            ims=[cv2.imread(str(OUT/'review'/(s+'_rgb.jpg'))) for s in ordinals[i:i+4]]
            cv2.imwrite(str(OUT/'review'/f'{label}_{i//4:02}.jpg'),np.concatenate(ims,0))
    collisions={f'{sem}:{tok}':sorted(ids) for (sem,tok),ids in render_links.items() if len(ids)>1}
    save(OUT/'review_manifest.json',dict(uniform=uniform,anomalies=anomaly_frames,additional=list(representatives.values()),
        all_review_frames=chosen,all_have_six_RGB_and_instance_mosaics=True,
        automatic_label_validation=False,human_review_status='Pending'))
    save(OUT/'anomaly_audit.json',anomalies)
    result=dict(status='SidecarContractGo_QualityPartial',frames=len(frames),stats=dict(stats),
        object_class_status={k:dict(v) for k,v in classes.items()},actor_view_status=dict(assoc),
        distance_status={k:dict(v) for k,v in ranges.items()},anomalies=len(anomalies),
        review_frames=len(chosen),uniform_review_frames=24,render_token_cross_actor_conflicts=collisions,
        runtime_s=time.perf_counter()-start,policy=CFG['association'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantics='Valid=evidence-supported candidate, Invalid=explicit rejection, Unknown=unresolved; no negative labels',
        source_frame=None,timestamp_s=None,training=False,human_certified=False)
    save(OUT/'sidecar_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
