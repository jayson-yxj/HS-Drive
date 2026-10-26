"""214-frame GT lineage and semantic/render-region coverage, CPU only."""
import hashlib
import json
import os
import time
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from datasets.m8j4_gt import regions
from datasets.m8j5a_annotation_audit import converted_geometry,raw_corners,source_geometry_residuals,project_both
from tools.m8j3_validate_clip import RAW,read_annotation

OUT=Path(os.environ.get('HS_M8J5A_OUT','/root/gpufree-data/hs-drive-private/m8j5a-raw-audit'))
REPORTS=Path(os.environ.get('HS_M8J5A_REPORTS','reports'))
OLD=Path('/root/gpufree-data/hs-drive-private/m8j4-single-frame')


def report(name):
    REPORTS.mkdir(parents=True,exist_ok=True)
    return REPORTS/name


def save(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def polygon_overlap(labels,tokens,region,projection):
    if projection['status']!='Valid':return 0
    x0,y0,x1,y1=region['bbox_xyxy']
    poly=np.asarray(projection['polygon'],np.float32)-[x0,y0]
    mask=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillConvexPoly(mask,np.round(poly).astype(np.int32),1)
    return int(np.count_nonzero(mask & (labels[y0:y1,x0:x1]==region['semantic']) &
                                (tokens[y0:y1,x0:x1]==region['render_token'])))


def main():
    os.umask(0o077);(OUT/'frames').mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();counts=Counter();unique=defaultdict(set);state=Counter()
    by_frame=[];by_cam=defaultdict(Counter);by_class=defaultdict(Counter);reasons=Counter();allregions=[]
    maxres=defaultdict(float);hist_loss=[];hashes=[];cases=[]
    for path in sorted((RAW/'anno').glob('*.json.gz')):
        ordinal=path.name[:5];a=read_annotation(path);ego=next(o for o in a['bounding_boxes'] if o['class']=='ego_vehicle')
        legacy=json.loads((OLD/'reviewed_sidecar'/(ordinal+'.json')).read_text())
        oldmap={o['actor_id']:o for o in legacy['objects']}
        entries=[];fc=Counter();ids=[str(o['id']) for o in a['bounding_boxes']]
        assert len(ids)==len(set(ids)), 'DuplicateActorIDsMustNotBeSilentlyDeduplicated'
        hashes.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for obj in a['bounding_boxes']:
            cls=obj['class'];aid=str(obj['id']);counts['raw_'+cls]+=1;fc['raw_'+cls]+=1;unique[cls].add(aid)
            state[cls+':'+str(obj.get('state','NotRecorded'))]+=1
            new=converted_geometry(obj,ego['world2ego']);old=oldmap.get(aid)
            lineage=dict(raw=True,legacy_sidecar=old is not None,legacy_geometry_valid=bool(old and old['geometry_status']=='Valid'),
                all_converted_geometry=new['geometry_status']=='Valid',legacy_positive_candidate=bool(old and old['positive_supervision_candidate']),
                final_training_gt=False,final_reason='TrainingDisabled_GoldenRequirementSuspended_ForAnnotationAudit')
            filters=[]
            if cls=='ego_vehicle':filters.append('EgoExcludedFromDetection')
            elif old is None:filters.append('MissingLegacyRecord');hist_loss.append(dict(ordinal=ordinal,actor_id=aid))
            else:
                if old['geometry_status']!='Valid':filters.append('LegacyGeometryUnsupported_'+old['geometry_status'])
                if old.get('model_class') not in ('car','pedestrian'):filters.append('LegacyClassUnsupported')
                if old['reason']=='HiddenBelowGround_NoTrainingTarget':filters.append('UndergroundExplicitExclusion')
                if not any(v['status']=='Valid' for v in old['views'].values()):filters.append('AssociationEvidenceNotSupported_NotGeometryFailure')
                if any('withdrawn_match' in v for v in old['views'].values()):filters.append('RenderTokenConflictQuarantine')
                if new['geometry_status']=='Valid':
                    if not old.get('yaw_only_comparable'):filters.append('YawOnlyTiltUnsupported')
                    c=np.asarray(new['center_hs_m'])
                    if not (np.all(c>[-51.2,-51.2,-5]) and np.all(c<[51.2,51.2,3])):filters.append('OutsidePETRRange')
                    if not any(v['status']=='Valid' and v['projection']['bbox_xyxy'][3]*.88>280 and v['projection']['bbox_xyxy'][1]*.88<792 for v in old['views'].values()):filters.append('NoSupportedViewInPETRCrop')
            entry=dict(actor_id=aid,source_class=cls,source_state=obj.get('state'),raw=obj,converted=new,
                legacy=old,lineage=lineage,filter_reasons=filters,views={})
            if cls!='ego_vehicle':
                reasons.update(filters);by_class[cls].update({k:int(v) for k,v in lineage.items() if isinstance(v,bool)})
            if new['geometry_status']=='Valid':
                res=source_geometry_residuals(obj);entry['source_geometry_residuals']=res
                for k,v in res.items():
                    if isinstance(v,(int,float)):maxres[k]=max(maxres[k],v)
                for cam in CAMERAS:
                    direct,converted,error=project_both(obj,new,a['sensors'][cam],ego['world2ego'])
                    entry['views'][cam]=dict(raw_projection=direct,converted_projection=converted,reprojection_bbox_error_px=error)
                    if error is not None:maxres['projection_bbox_error_px']=max(maxres['projection_bbox_error_px'],error)
                    if cls!='ego_vehicle':
                        key=cam+':'+cls;by_cam[key]['actor_views']+=1
                        by_cam[key]['raw_projectable']+=int(direct['status']=='Valid')
                        by_cam[key]['converted_projectable']+=int(converted['status']=='Valid')
                        by_cam[key]['legacy_projectable']+=int(bool(old and old.get('views',{}).get(cam,{}).get('projection',{}).get('status')=='Valid'))
                        by_cam[key]['legacy_association_supported']+=int(bool(old and old.get('views',{}).get(cam,{}).get('status')=='Valid'))
            entries.append(entry)
        frame_regions=[]
        for cam in CAMERAS:
            ins=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(ordinal+'.png')),-1)
            sem=cv2.imread(str(RAW/'camera'/('semantic_'+VIEW[cam])/(ordinal+'.png')),-1)
            labels,tokens,rs=regions(ins,sem)
            for r in rs:
                r=dict(r,ordinal=ordinal,camera=cam)
                expected='walker' if r['semantic']==12 else 'vehicle'
                hits=[];ego_pixels=0
                for entry in entries:
                    if entry['source_class'] not in (expected,'ego_vehicle') or cam not in entry['views']:continue
                    n=polygon_overlap(labels,tokens,r,entry['views'][cam]['raw_projection'])
                    if entry['source_class']=='ego_vehicle':ego_pixels=n;continue
                    if n:hits.append(dict(actor_id=entry['actor_id'],pixels=n,fraction=n/r['pixels'],
                        legacy_association_status=(entry['legacy'] or {}).get('views',{}).get(cam,{}).get('status','Unavailable')))
                hits.sort(key=lambda x:x['pixels'],reverse=True)
                best=hits[0]['fraction'] if hits else 0.
                r.update(raw_box_overlap_candidates=hits,ego_overlap_fraction=ego_pixels/r['pixels'],
                    identity='NotProven_RenderTokenNotActorID',best_raw_overlap_fraction=best,
                    coverage='RawBoxSpatialSupport' if best>=.5 else ('EgoPossible' if ego_pixels/r['pixels']>=.5 else 'UnexplainedVisibleRegion_UnknownNotBackground'))
                frame_regions.append(r)
                key=cam+':render_semantic_'+str(r['semantic']);by_cam[key]['render_region_observations']+=1
                by_cam[key][r['coverage']]+=1
                if r['pixels']>=100:by_cam[key]['region_ge100px_'+r['coverage']]+=1
                allregions.append(r)
        for e in entries:
            if e['source_class']!='ego_vehicle':
                fc['raw_non_ego']+=1;fc['legacy_retained']+=int(e['lineage']['legacy_sidecar']);fc['converted_geometry']+=int(e['lineage']['all_converted_geometry'])
                fc['legacy_positive_candidates']+=int(e['lineage']['legacy_positive_candidate'])
        save(OUT/'frames'/(ordinal+'.json'),dict(ordinal=ordinal,source_frame=None,timestamp_s=None,
            objects=entries,render_regions=frame_regions,training_enabled=False))
        by_frame.append(dict(ordinal=ordinal,**fc))
        if int(ordinal)%40==0:print(ordinal,'done',flush=True)
    save(OUT/'render_regions.json',allregions);save(OUT/'raw_annotation_hashes.json',hashes)
    save(OUT/'per_frame_counts.json',by_frame)
    counts['unexplained_regions_ge100px']=sum(r['pixels']>=100 and r['coverage'].startswith('Unexplained') for r in allregions)
    result=dict(status='RawLineageAudited_TrainingDisabled',frames=len(by_frame),source_counts=dict(counts),
        source_state_counts=dict(state),unique_actor_ids_per_class={k:len(v) for k,v in unique.items()},
        record_loss_to_legacy_sidecar=len(hist_loss),lost_records=hist_loss,
        per_class_lineage={k:dict(v) for k,v in by_class.items()},per_camera_class={k:dict(v) for k,v in by_cam.items()},
        nonexclusive_filter_reasons=dict(reasons),max_numeric_residuals=dict(maxres),runtime_s=time.perf_counter()-started,
        counting='Actor-frame records; per-view/region counts are correlated observations, not independent objects. Unique IDs valid within one clip only.',
        final_training_gt=0,training=False,golden_signoff='SuspendedByLatestUserRequest',
        caveat='RawBoxSpatialSupport is projection consistency, not proof of visibility/identity or dataset completeness.')
    save(report('m8j5a_raw_coverage.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
