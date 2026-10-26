"""Reuse 214 frozen metadata records; render exactly eight predeclared frames."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from datasets.m8j5b_coverage import coverage_view,projection_mask
from datasets.m8j5c_detection_scope import task_eligibility,region_status,coverage_qualification
from tools.m8j3_validate_clip import RAW

ROOT=Path(os.environ.get('M8J5C_CANDIDATE_ROOT','/root/gpufree-data/hs-drive-private/m8j5c-numerical-supervision/candidates'))
REPORT=Path(os.environ.get('M8J5C_CANDIDATE_REPORT','reports'))
SOURCE=Path('/root/gpufree-data/hs-drive-private/m8j5a-raw-audit')
OLD=Path('/root/gpufree-data/hs-drive-private/m8j5b-coverage-supervision')
PROTOCOL=Path('configs/m8j5c_protocol.json')


def save(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    os.umask(0o077);ROOT.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    web=ROOT/'web';web.mkdir();(web/'images').mkdir()
    p=json.loads(PROTOCOL.read_text());selection=p['selection']['ordinals'];bounds=np.asarray(p['detection_task']['center_range_hs_m'])
    case_count=Counter({'00124':1,'00196':1,'00213':2})
    summary=Counter();rows=[];page=[];public=[];questions=[]
    sources={str(PROTOCOL):hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    for path in sorted((SOURCE/'frames').glob('*.json')):
        f=json.loads(path.read_text());n=f['ordinal'];sidepath=OLD/'sidecar'/(n+'.json')
        side=json.loads(sidepath.read_text());unexplained=[r for r in f['render_regions'] if region_status(r).startswith('Unexplained')]
        assert not side['frame_negative_enabled'] and not side['safe_negative_regions']
        summary['frames']+=1;summary['frames_with_unexplained_evidence']+=bool(unexplained)
        summary['frames_with_confirmed_source_gap']+=case_count[n]>0
        summary['unexplained_region_observations']+=len(unexplained)
        summary['frames_with_unexplained_ge100px']+=any(r['pixels']>=100 for r in unexplained)
        summary['frames_with_unexplained_lt100px']+=any(r['pixels']<100 for r in unexplained)
        # These categories are reasons/evidence statuses, not a truth label for an entire frame.
        row=dict(ordinal=n,concrete_gap_cases=case_count[n],unexplained_region_observations=len(unexplained),
            unknown_regions_may_be_real_missing_objects=True,not_equivalent_to_entire_frame_wrong=True,
            existing_evidence_not_used_for_detection_qualification=['AllSixSourceBoxesAndRenderRegions','VisibleClassSpecificCoverage','EgoProjectionProxy_NotAutomaticExclusion'],
            evidence_not_collected=['WholeSixViewTaskCoverageReview','UnknownRegionDepthAndTaskMembership','ReliableHandlingOfOutOfScopeVisibleObjects'],
            historical_rules_beyond_visible_detection=['AllTenClassesEmpty3DEnvelope','OccludedSpaceFreeCertificate','MissingTimestampAsNegativeBlockerForSingleFrame'],
            old_negative_enabled=False,new_negative_enabled=False)
        rows.append(row)
        if n not in selection:continue
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        sources[str(sidepath)]=hashlib.sha256(sidepath.read_bytes()).hexdigest()
        eligible={};objects=[];stats=Counter()
        for o in f['objects']:
            if o['source_class']=='ego_vehicle':continue
            if o['source_class'] in ('vehicle','walker'):
                e=task_eligibility(o,f['render_regions'],CAMERAS,bounds)
                eligible[o['actor_id']]=e
                stats['source_task_class_records']+=1;stats['positive_candidates']+=e['eligible']
                stats['class_'+str(e['model_class'])+'_positive']+=e['eligible']
                for reason in e['reasons']:stats['excluded_'+reason]+=1
            else:e=dict(eligible=False,reasons=['OutsideTaskClass'],human_review='Pending')
            objects.append(dict(actor_id=o['actor_id'],source_class=o['source_class'],views=o['views'],
                positive=e,geometry=o['converted']['geometry_status']))
        images={};viewrows=[];target_question_rows=[]
        for cam in CAMERAS:
            images[cam]={}
            for kind in ('rgb','semantic','instance'):
                ext='.jpg' if kind=='rgb' else '.png';link=web/'images'/(n+'_'+cam+'_'+kind+ext)
                link.symlink_to(RAW/'camera'/(kind+'_'+VIEW[cam])/(n+ext));images[cam][kind]='images/'+link.name
            sem=cv2.imread(str(RAW/'camera'/('semantic_'+VIEW[cam])/(n+'.png')),-1)
            ins=cv2.imread(str(RAW/'camera'/('instance_'+VIEW[cam])/(n+'.png')),-1)
            evidence=coverage_view(sem,ins,f['objects'],cam,4)
            ignore=evidence['ignore'].copy()
            # Excluded source boxes remain visible as Ignore, never deleted into background.
            excluded=0
            for o in f['objects']:
                if o['source_class'] in ('vehicle','walker') and not eligible[o['actor_id']]['eligible']:
                    poly=o['views'][cam]['raw_projection']
                    ignore|=projection_mask(poly,sem.shape);excluded+=poly.get('status')=='Valid'
            rgba=np.zeros((*ignore.shape,4),np.uint8);rgba[:,:,:3]=[220,30,210];rgba[:,:,3]=ignore.astype(np.uint8)*150
            maskpath=web/'images'/(n+'_'+cam+'_mask.png');cv2.imwrite(str(maskpath),rgba);images[cam]['mask']='images/'+maskpath.name
            # Human-friendly semantic colors; keep raw semantic image separately.
            colors=np.zeros((*sem.shape,3),np.uint8)
            colors[sem==12]=[30,100,240];colors[sem==14]=[40,210,80]
            colors[np.isin(sem,(13,15,16,17,18,19))]=[220,150,50]
            colpath=web/'images'/(n+'_'+cam+'_semantic_color.png');cv2.imwrite(str(colpath),colors)
            images[cam]['semantic_color']='images/'+colpath.name
            regions=[r for r in unexplained if r['camera']==cam]
            for r in regions:
                q=dict(ordinal=n,camera=cam,bbox_xyxy=r['bbox_xyxy'],semantic=r['semantic'],pixels=r['pixels'],
                    render_token=r['render_token'],evidence='UnexplainedSourceRegion_NotActorIdentity',
                    question='Is this a visible task object, ego/body artifact, out-of-range object, or unresolved? Review all source boxes and all six cameras; no 3D location may be inferred from token.',
                    status='Pending_NoHumanSignoff')
                target_question_rows.append(q);questions.append(q)
            viewrows.append(dict(camera=cam,regions=evidence['records'],unexplained_count=len(regions),
                excluded_box_projections=excluded,ignore_pixels=int(ignore.sum()),original_pixels=int(ignore.size),
                frame_detection_coverage='Unresolved',negative_enabled=False))
        qualification=coverage_qualification(case_count[n],len(unexplained),False)
        summary['selected']+=1
        public.append(dict(ordinal=n,statistics=dict(stats),unexplained_regions=len(unexplained),
            confirmed_gap_cases=case_count[n],qualification=qualification,
            six_camera_review='Pending',normal_detection_training='NotEligible',
            private_page=str(web/'index.html')+'#'+n))
        page.append(dict(ordinal=n,images=images,objects=objects,views=viewrows,negative_enabled=False,
            positive_count=stats['positive_candidates'],qualification=qualification))
        save(ROOT/(n+'_review.json'),dict(objects=objects,questions=target_question_rows,qualification=qualification))
    assert len(page)==8 and len(rows)==214
    template=Path('tools/m8j5b_coverage_ui.html').read_text()
    template=template.replace('M8-J5B','M8-J5C').replace('Coverage-Aware Supervision','固定8帧 · 可见目标检测候选')
    template=template.replace('没有安全负区域证书，N=0','检测任务全目标覆盖未闭合，N=0；不是要求Occupancy Free证书')
    template=template.replace('实际安全负区域 0','检测负监督未授权')
    template=template.replace('<option value="semantic">Semantic</option>','<option value="semantic_color">Semantic 类别配色</option><option value="semantic">原始 Semantic</option>')
    template=template.replace('正监督诊断候选（绿）','40m可见检测候选（绿）')
    template=template.replace('局部 Ignore（紫）','未知 + 范围外/不可用GT Ignore（紫）')
    template=template.replace('let index=2','let index=0')
    template=template.replace('__DATA__',json.dumps(page).replace('<','\\u003c'))
    (web/'index.html').write_text(template)
    save(ROOT/'all214_reason_categories.json',rows)
    save(ROOT/'targeted_review_questions.json',questions)
    save(ROOT/'selected_source_hashes.json',sources)
    save(REPORT/'m8j5c_candidate_manifest.json',dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sources[str(PROTOCOL)],selection_rule=p['selection'],task_definition=p['detection_task'],
        candidates=public,selected_frames=8,extra_frames=0,pixel_rescanned_frames=8,automatic_signoffs=0,
        decision='B_CurrentSetNotEligible_KeepIgnoreRegression',negative_qualified_frames=0,
        private_page=str(web/'index.html'),targeted_question_count=len(questions)))
    save(REPORT/'m8j5c_214_reason_analysis.json',dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='ReusedFrozen214Metadata_NoFullImageRescan',counts=dict(summary),
        missing_certificate_is_not_bad_annotation=True,
        detection_negative='NoUnmatchedInScopeVisibleTargetAfterTaskCoverageVerification',
        occupancy_free='PhysicalSpaceEmpty_NotRequiredForDetectionNoObject',
        rule_mismatch_confirmed=True,missing_boxes_not_repaired_by_scope_change=True,
        raw_timestamps='NotRequiredForStaticFrameNoObjectLabel_StillUnavailableForTemporalUse',
        notes='Known source gaps: four cases in three frames. All214 have unexplained region candidates, not 214 proven wrong frames. No range can be assigned to an unboxed region from render token.',
        historical_gate_unchanged=True,new_training_authorization=False))
    print(json.dumps(dict(summary=summary,selected=selection,questions=len(questions),decision='B')))


if __name__=='__main__':main()
