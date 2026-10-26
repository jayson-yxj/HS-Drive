"""CPU-only append-only trust audit and private human review packet."""
import html
import json
import os
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

from datasets.m8j5_supervision import classify_object,digest,hs_to_petr_center
from datasets.m8b_petr_adapter import old_boxes_to_hs,corners_hs
from tools.m8j4_sidecar import render


def save(path, data):
    with Path(path).open('x') as f: json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    os.umask(0o077)
    cfg=json.loads(Path('configs/m8j5_protocol.json').read_text())
    out=Path(cfg['private_root']);out.mkdir(exist_ok=False);(out/'trust_sidecar').mkdir()
    src=Path(cfg['source_sidecar_root'])
    prior_review=json.loads((src/'review_manifest.json').read_text())
    preferred=set(prior_review['uniform']+prior_review['anomalies'])
    counts=defaultdict(Counter);objects=[];frames={};difference=[];roundtrip_max=0
    for p in sorted((src/'reviewed_sidecar').glob('*.json')):
        frame=json.loads(p.read_text());frames[frame['ordinal']]=frame
        rows=[]
        for obj in frame['objects']:
            trust=classify_object(obj)
            for axis in ('geometry_validity','class_validity','identity_validity','visibility'):
                counts[axis][trust[axis]]+=1
            counts['eligibility']['provisional_positive' if trust['supervision_eligibility']['provisional_positive_candidate'] else 'not_provisional']+=1
            if obj['label_status']=='Valid' and not obj['positive_supervision_candidate']:
                difference.append(dict(ordinal=frame['ordinal'],actor_id=obj['actor_id'],tilt_deg=obj['tilt_deg'],
                    filter_reason='YawOnlyTiltAbove2Degrees',geometry_validity=trust['geometry_validity'],
                    additional_next_stage_filters=trust['supervision_eligibility']['reasons']))
            if obj['geometry_status']=='Valid':
                t=np.array(hs_to_petr_center(obj));t[2]-=t[5]/2
                hs=old_boxes_to_hs(t[None])[0]
                expected=np.array([*obj['center_hs_m'],*obj['dimensions_lwh_m'],obj['yaw_ccw_rad'],0,0])
                error=np.abs(hs[:6]-expected[:6]).max()
                yaw=abs((hs[6]-expected[6]+np.pi)%(2*np.pi)-np.pi)
                roundtrip_max=max(roundtrip_max,error,yaw)
            row=dict(candidate_id=frame['ordinal']+'_'+obj['actor_id'],ordinal=frame['ordinal'],object=obj,trust=trust)
            rows.append(row);objects.append(row)
        save(out/'trust_sidecar'/p.name,dict(schema='m8j5.trust.v1',ordinal=frame['ordinal'],source_frame=None,
            timestamp_s=None,objects=rows,positive_authorized=False,negative_authorized=False,
            safe_negative_regions=[],golden_signoff='Pending'))
    # Purposeful stratification for review only; no quality-tuned train/val split.
    pool=[o for o in objects if o['object']['geometry_status']=='Valid']
    pool.sort(key=lambda x:(x['ordinal'] not in preferred,x['ordinal'],x['candidate_id']))
    def category(o):
        b=o['object'];v=list(b['views'].values());d=b['distance_xy_m'];tags=[]
        if b.get('model_class')=='car': tags += ['near_vehicle' if d<20 else 'far_vehicle']
        if b.get('model_class')=='pedestrian': tags += ['pedestrian']
        if any('PartialOcclusion' in x.get('occlusion_state','') for x in v):tags+=['occlusion_proxy']
        if sum(x.get('projection',{}).get('status')=='Valid' for x in v)>1 or any(x.get('projection',{}).get('truncated') for x in v):tags+=['camera_boundary']
        if any(x.get('withdrawn_match') for x in v):tags+=['withdrawn_identity_conflict']
        if b['reason']=='HiddenBelowGround_NoTrainingTarget':tags+=['underground']
        if abs(max(abs(b['center_hs_m'][0]),abs(b['center_hs_m'][1]))-51.2)<10:tags+=['PETR_range_boundary']
        if b['label_status']=='Valid' and not b['positive_supervision_candidate']:tags+=['712_vs_711']
        return tags
    chosen={}; quotas={'712_vs_711':1,'withdrawn_identity_conflict':4,'underground':2,
        'PETR_range_boundary':4,'pedestrian':6,'near_vehicle':5,'far_vehicle':5,'occlusion_proxy':5,'camera_boundary':4}
    for tag,quota in quotas.items():
        eligible=[o for o in pool if tag in category(o) and o['candidate_id'] not in chosen]
        # Spread selections across the available temporal support, no repeated object in one frame.
        ids=np.linspace(0,len(eligible)-1,min(quota,len(eligible)),dtype=int) if eligible else []
        for i in ids:chosen[eligible[i]['candidate_id']]=eligible[i]
    for o in pool:
        if len(chosen)>=cfg['golden_candidate_budget']:break
        chosen.setdefault(o['candidate_id'],o)
    chosen=list(chosen.values())[:cfg['golden_candidate_budget']]
    reviewdir=out/'review';reviewdir.mkdir();reviews=[];tagcounts=Counter()
    for row in chosen:
        row=dict(row,selection_tags=category(row),golden_status='CandidateOnly_PendingHumanReview',
                 safe_negative_certification='None_AllTenClassesUnreviewed')
        row['candidate_sha256']=digest(row)
        reviews.append(row);tagcounts.update(row['selection_tags'])
    for ordinal in sorted({r['ordinal'] for r in reviews}):render(frames[ordinal],reviewdir)
    save(out/'golden_candidates.json',dict(schema='m8j5.human-review-candidates.v1',candidates=reviews,
        golden_count=0,human_signoffs=[],source_frame=None,timestamp_s=None))
    template=[]
    for r in reviews:
        template.append(dict(candidate_id=r['candidate_id'],candidate_sha256=r['candidate_sha256'],decision='NeedsReview',
            reviewer=None,reviewed_at=None,review_source=None,geometry_confirmed=False,class_confirmed=False,
            identity_confirmed=False,visible_confirmed=False,notes=''))
    save(out/'review_responses_TEMPLATE.json',template)
    blocks=['<meta charset="utf-8"><title>M8-J5 private review</title><h1>候选审核，不是 Golden GT</h1>',
        '<p>请逐对象核对中心/尺寸/Yaw、八角点、类别、身份、可见性。不得据此将其他区域认定背景。填写 JSON 副本，保留真实审核者、UTC 时间与来源；不能确认则 NeedsReview。原始时间戳未知。</p>']
    for r in reviews:
        b=r['object'];x=html.escape(json.dumps(dict(tags=r['selection_tags'],trust=r['trust'],box={k:b.get(k) for k in ('center_hs_m','dimensions_lwh_m','yaw_ccw_rad','corners_hs_m')}),ensure_ascii=False,indent=2))
        blocks+=['<h2>'+r['candidate_id']+'</h2><pre>'+x+'</pre>',
            '<img width="1440" src="review/'+r['ordinal']+'_rgb.jpg">',
            '<img width="1440" src="review/'+r['ordinal']+'_instance.jpg">']
    (out/'review_index.html').write_text('\n'.join(blocks))
    save(out/'filter_difference.json',difference)
    summary=dict(stage='M8-J5',evidence='ActualPrivatePayload_CPU',frames=len(frames),object_records=len(objects),
        trust_counts={k:dict(v) for k,v in counts.items()},old_supported=712,old_positive_candidates=711,
        difference_count=len(difference),difference_reasons=dict(Counter(x['filter_reason'] for x in difference)),
        private_difference_file=str(out/'filter_difference.json'),geometry_roundtrip_max_error=roundtrip_max,
        candidate_count=len(reviews),candidate_frames=len({r['ordinal'] for r in reviews}),review_tag_counts=dict(tagcounts),
        golden_count=0,human_signoff='Pending_NotPerformedByAgent',safe_negative_regions_certified=0,
        authorized_positive_targets=0,authorized_negative_targets=0,
        real_optimizer_smoke='NotRun_HumanSignoffMissing',private_review=str(out/'review_index.html'),
        independent_scene_count=1,no_quality_or_generalization_claim=True)
    save(Path('reports/m8j5_gt_trust.json'),summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
