"""Bounded CPU coverage audit on the existing 214-frame clip; no download."""
from collections import Counter,defaultdict
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from datasets.m8j5_supervision import hs_to_petr_center
from datasets.m8j5b_coverage import coverage_view,positive_eligibility
from datasets.m8j4_petr_input import A
from geometry.reference import CARLA_EGO_TO_HS_EGO as S,OPTICAL_TO_CARLA_CAM as O
from tools.m8j3_validate_clip import RAW,read_annotation

ROOT=Path(os.environ.get('HS_M8J5B_ROOT','/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision'))
SOURCE=Path('/root/gpufree-data/hs-drive-private/m8j5a-raw-audit')


def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-existing',action='store_true',help='Read existing sidecars after failed final assertion; never overwrite payload')
    args=parser.parse_args()
    os.umask(0o077)
    if not args.summarize_existing:
        ROOT.mkdir(exist_ok=False)
        for folder in ('sidecar','masks_v2','views','cases','inference_input'): (ROOT/folder).mkdir()
    protocol=json.loads(Path('configs/m8j5b_protocol.json').read_text())
    summary=Counter();percam=defaultdict(Counter);positive=Counter();reasons=Counter();hist=Counter()
    rows=[];case_rows=[];start=time.perf_counter()
    case_keys={('00213','CAM_FRONT',14,45980),('00196','CAM_FRONT',14,45976),('00213','CAM_FRONT',14,48630),('00124','CAM_FRONT_LEFT',12,31465)}
    if args.summarize_existing:
        for path in sorted((ROOT/'sidecar').glob('*.json')):
            f=json.loads(path.read_text());pcount=0;cropped=0
            for o in f['objects']:
                if o['positive']['eligible']:positive[o['positive']['model_class']]+=1;pcount+=1
                else:reasons.update(o['positive']['reasons'])
            hist[pcount]+=1
            for v in f['views']:
                summary.update(v['stats']);percam[v['camera']].update(v['stats'])
                summary['cropped_pixels']+=512*1408;summary['cropped_ignore_pixels']+=v['cropped_ignore_pixels'];cropped+=v['cropped_ignore_pixels']
                for r in v['regions']:
                    if (f['ordinal'],v['camera'],r['semantic'],r['render_token']) in case_keys:
                        case_rows.append(dict(ordinal=f['ordinal'],camera=v['camera'],region=r,
                            ignore_mask_pixels=v['stats']['ignore_pixels'],frame_negative_enabled=False,new_3d_boxes=0))
            rows.append(dict(ordinal=f['ordinal'],positive_source_boxes=pcount,missing_region_observations=f['missing_region_count'],
                cropped_ignore_pixels=cropped,frame_negative_enabled=False))
    for fp in sorted((SOURCE/'frames').glob('*.json')):
        if args.summarize_existing:break
        frame=json.loads(fp.read_text());n=frame['ordinal']
        anno=read_annotation(RAW/'anno'/(n+'.json.gz'))
        ego=next(o['raw'] for o in frame['objects'] if o['source_class']=='ego_vehicle')
        masks=[];projections=[];view_rows=[];observed=[]
        for cam in CAMERAS:
            sem=cv2.imread(str(RAW/'camera'/('semantic_'+VIEW[cam])/(n+'.png')),-1)
            ins=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(n+'.png')),-1)
            result=coverage_view(sem,ins,frame['objects'],cam,protocol['coverage']['ignore_dilation_original_pixels'])
            stats=result['stats'];summary.update(stats);percam[cam].update(stats)
            # Area resampling >0 preserves even a one-pixel unknown at a resized
            # pixel boundary. Exact original test crop; no random augmentation.
            resized=cv2.resize(result['ignore'].astype(np.float32),(1408,792),interpolation=cv2.INTER_AREA)>0
            crop=resized[280:792].copy();masks.append(crop)
            summary['cropped_pixels']+=crop.size;summary['cropped_ignore_pixels']+=int(crop.sum())
            sensor=anno['sensors'][cam]
            # Use exactly the same independent mount K/T as RGB-only prepare().
            # world2cam * inverse(world2ego) cancellation has stored float noise.
            optical=np.linalg.inv(S@np.asarray(sensor['cam2ego'])@O)
            projections.append(A@np.asarray(sensor['intrinsic'])@optical[:3])
            record=dict(camera=cam,stats=stats,regions=result['records'],ignore_rectangles=result['rectangles'],
                cropped_ignore_pixels=int(crop.sum()),calibration='OriginalBench2Drive_ActualPetrTestCrop',
                safe_background_status='NotCertified_SegmentationAbsenceNot3DFree')
            view_rows.append(record)
            for r in result['records']:
                if r['status']=='MissingSourceSupportCandidate':observed.append(r)
                if (n,cam,r['semantic'],r['render_token']) in case_keys:
                    key=n+'_'+cam+'_'+str(r['render_token'])
                    row=dict(ordinal=n,camera=cam,region=r,ignore_mask_pixels=stats['ignore_pixels'],
                        frame_negative_enabled=False,new_3d_boxes=0)
                    case_rows.append(row)
                    np.savez_compressed(ROOT/'cases'/(key+'_masks.npz'),ignore=result['ignore'],missing=result['missing'])
            if n in ('00124','00196','00213','00107','00017'):
                cv2.imwrite(str(ROOT/'views'/(n+'_'+cam+'_ignore.png')),result['ignore'].astype(np.uint8)*255)
        objects=[];unknown=[];pcount=0
        for o in frame['objects']:
            if o['source_class'] not in ('vehicle','walker'):continue
            eligibility=positive_eligibility(o,frame['render_regions'],CAMERAS)
            if eligibility['eligible']:
                positive[eligibility['model_class']]+=1;pcount+=1
            else:reasons.update(eligibility['reasons'])
            obj=dict(actor_id=o['actor_id'],source_class=o['source_class'],positive=eligibility,
                source_record_sha256=hashlib.sha256(json.dumps(o['raw'],sort_keys=True).encode()).hexdigest(),
                corners_hs_m=o['converted'].get('corners_hs_m'))
            if eligibility['eligible']:obj['petr_box']=hs_to_petr_center(o['legacy'])
            elif o['converted']['geometry_status']=='Valid':
                corners=np.asarray(o['converted']['corners_hs_m']);unknown.append(np.r_[corners.min(0)-.5,corners.max(0)+.5].tolist())
            objects.append(obj)
        hist[pcount]+=1
        record=dict(ordinal=n,source_frame=None,timestamp_s=None,objects=objects,unknown_aabbs=unknown,
            views=view_rows,camera_order=list(CAMERAS),frame_negative_enabled=False,safe_negative_regions=[],
            frame_fallback_reasons=['NoAllClass3DNegativeCertificate','MissingRawSensorTimestamps_OccludedSpaceUnverified'],
            human_signoff='Pending_Suspended',training_authorized=False,
            authorization='LossDiagnosticOnly_NoOptimizer',missing_region_count=len(observed))
        save(ROOT/'sidecar'/(n+'.json'),record)
        np.savez_compressed(ROOT/'masks_v2'/(n+'.npz'),ignore=np.stack(masks),image_projection=np.stack(projections).astype(np.float32))
        rows.append(dict(ordinal=n,positive_source_boxes=pcount,missing_region_observations=len(observed),
            cropped_ignore_pixels=int(np.stack(masks).sum()),frame_negative_enabled=False))
        if int(n)%40==0:print(n,'complete',flush=True)
    assert len(rows)==214 and len(case_rows)==4
    assert all(r['region']['status']=='MissingSourceSupportCandidate' for r in case_rows)
    # Low cross-sensor agreement strengthens Unknown, never suppresses its mask.
    # The original run's >.9 assertion failed on two real small distant cars;
    # retain that failure and each measured agreement instead of inventing trust.
    save(ROOT/'case_reproduction.json',case_rows)
    save(ROOT/'frame_counts.json',rows)
    report=dict(stage='M8-J5B',checked_at_utc=datetime.now(timezone.utc).isoformat(),evidence='Actual214FrameCPU_PayloadAudit',
        frames=len(rows),views=len(rows)*6,elapsed_s=time.perf_counter()-start,
        summarize_existing=args.summarize_existing,counts=dict(summary),
        per_camera={k:dict(v) for k,v in percam.items()},positive_loss_diagnostic_source_boxes=dict(positive),
        positive_boxes_per_frame_histogram=dict(hist),nonexclusive_positive_exclusions=dict(reasons),
        missing_region_observations=sum(r['missing_region_observations'] for r in rows),
        frames_with_missing_regions=sum(r['missing_region_observations']>0 for r in rows),
        original_ignore_fraction=summary['ignore_pixels']/summary['pixels'],
        cropped_ignore_fraction=summary['cropped_ignore_pixels']/summary['cropped_pixels'],
        semantic_instance_all_pixel_agreement=summary['full_image_agreement_pixels']/summary['pixels'],
        semantic_instance_foreground_agreement=summary['agreed_foreground_pixels']/summary['foreground_pixels'],
        four_missing_cases_reproduced=True,case_semantic_agreements=[r['region']['semantic_agreement'] for r in case_rows],
        real_frames_negative_disabled=214,real_safe_negative_certificates=0,
        independence='One correlated clip; region observations and frames are not independent physical objects',
        reference='Source semantics + geometric overlap; not exhaustive physical truth',
        human_review='Pending_Suspended',training='NotRun',new_3d_boxes=0)
    save('reports/m8j5b_coverage_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
