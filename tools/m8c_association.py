"""Separate evaluator: masks support image identity, not 3D truth by overlap."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from tools.m8c_common import *
from diagnostics.m8b_metrics import coarse,project_bounds,iou2,upright_iou
from diagnostics.m8c_geometry import association_grade,pixel_support
from datasets.m8b_petr_adapter import corners_hs,A,ORDER


def main():
    ledger=json.loads((BASELINE/'evaluation_ledger.json').read_text())['frames'];index={x['token']:x for x in json.loads((BASELINE/'evaluation_index.json').read_text())}
    caches={s:json.loads(Path('runs/m5o_cache_'+s+'/supervision.json').read_text()) for s in ['train','val','calibration','sealed']}
    associations=[];support=[];background=[];rgb_stats=[];reference_audits=[]
    for f in ledger:
        token=f['token'];row=index[token]['row'];packet=json.loads(Path(index[token]['packet']['path']).read_text());raw=json.loads(Path(row['raw_frame']).read_text());base=Path(row['raw_frame']).parent.parent;pred=json.loads((BASELINE/'forward'/(token+'.json')).read_text())
        aa={(a['actor_id'],a['camera_index']):a for a in caches[row['split']]['actors'] if a['sample_id']==f['sample_id'] and a['seed']==101}
        depth=np.load(row['pixel_label'],allow_pickle=False)['depth'];masks={};semantic={};instance={};tight={};visible_info={}
        for ci,c in enumerate(packet['cameras']):
            with np.load(base/raw['cameras'][ORDER[ci]]['instance'],allow_pickle=False) as z:semantic[ci]=z['semantic'];instance[ci]=z['instance']
            cc=row['actors'][ci]['actors'];claims=Counter(q['render_id'] for a in cc for q in a['identity_candidates'] if q['accepted'])
            rgb=np.asarray(Image.open(c['path']).convert('RGB'));rgb_stats.append(dict(token=token,camera=c['name'],mean_rgb=rgb.mean((0,1)).tolist(),std_rgb=rgb.std((0,1)).tolist(),dark_fraction=float((rgb.mean(-1)<10).mean()),bright_fraction=float((rgb.mean(-1)>245).mean())))
            for gi,g in enumerate(f['gts']):
                key=(g['actor_id'],ci)
                if key not in aa:continue
                a=next(a for a in cc if a['actor_id']==g['actor_id']);ids=[q['render_id'] for q in a['identity_candidates'] if q['accepted'] and claims[q['render_id']]==1]
                m=np.isin(instance[ci],ids)&np.isfinite(depth[ci])&(depth[ci]>=1.5)&(depth[ci]<80)
                assert int(m.sum())==aa[key]['native_visible_pixels']
                native=m.copy();m[:151]=False # pixel center v>=ceil(332/2.2), no model mask use
                masks[(gi,ci)]=m;info=pixel_support(m,None);tight[(gi,ci)]=info['bbox'];visible_info[(gi,ci)]=aa[key]
                vv,uu=np.where(m);height=int(np.ptp(vv)+1) if len(vv) else 0
                sv=dict(token=token,frame_id=packet['frame_id'],timestamp_s=packet['timestamp_s'],split=f['split'],group=f['group'],actor_id=g['actor_id'],gt_index=gi,camera=c['name'],camera_index=ci,type=g['type'],range_m=float(np.linalg.norm(g['box'][:2])),native_pixels=int(native.sum()),retained_pixels=int(m.sum()),native_visible_height_px=height,augmented_visible_height_nominal_px=2.2*height,GT_association_purity=aa[key]['association_purity'],GT_truncated=aa[key]['truncated'],reference='RecordedActorOBB_RendererIdentityProxy',identity_ids=ids,tight_native_bbox=info['bbox'])
                support.append(sv)
                reference_audits.append(dict(token=token,actor_id=g['actor_id'],camera=c['name'],purity=aa[key]['association_purity'],corner_residual_m=aa[key]['vertex_reconstruction_residual_m'],world_closure_m=aa[key]['world_closure_residual_m'],snapshot_frame=next(a for a in raw['actors'] if a['actor_id']==g['actor_id']).get('actor_snapshot_frame_id'),rgb_frame=packet['frame_id']))
        pairs={r['gt_index']:r['prediction_index'] for r in f['curves']['0.1']['conditional']}
        proj_cache={}
        def proj(pi,ci):
            key=(pi,ci)
            if key not in proj_cache:proj_cache[key]=project_bounds(corners_hs(pred['boxes'][pi]),np.array(packet['cameras'][ci]['K']),np.array(packet['cameras'][ci]['T_cam_to_ego']),augmented=False)
            b=proj_cache[key]
            if b is None:return None
            b=b.copy();b[1]=max(b[1],332/2.2)
            return b if b[3]>b[1] else None
        for gi,g in enumerate(f['gts']):
            pi=pairs.get(gi);views=[]
            if pi is not None:
                for ci in range(6):
                    if (gi,ci) not in masks:continue
                    b=proj(pi,ci);sp=pixel_support(masks[(gi,ci)],b);competitors=[iou2(b,v) for (gj,cj),v in tight.items() if cj==ci and gj!=gi]
                    views.append(dict(camera=ORDER[ci],camera_index=ci,retained_pixels=sp['pixels'],mask_recall=sp['mask_recall'],instance_bbox_iou=sp['bbox_iou'],GT_association_purity=visible_info[(gi,ci)]['association_purity'],competing_bbox_iou=max(competitors or [0.]),projected_bbox=None if b is None else b.tolist(),instance_bbox=sp['bbox']))
            grade=association_grade(views,pi is not None)
            rec=dict(token=token,sample_id=f['sample_id'],split=f['split'],group=f['group'],frame_id=packet['frame_id'],timestamp_s=packet['timestamp_s'],gt_index=gi,actor_id=g['actor_id'],type=g['type'],range_m=float(np.linalg.norm(g['box'][:2])),grade=grade,prediction_index=pi,score=None if pi is None else pred['scores'][pi],views=views,visible_cameras=len(g['views']),metric_error=None,identity_evidence='Automated2DMultiviewSupport_NotHumanCertified',source_GT=asset(row['raw_frame']))
            if grade=='IdentitySupported':
                b=np.array(pred['boxes'][pi]);gt=np.array(g['box']);delta=b[:3]-gt[:3];chosen=max(views,key=lambda v:v['retained_pixels']);ci=chosen['camera_index'];opt=np.linalg.inv(np.array(packet['cameras'][ci]['T_cam_to_ego']))[:3,:3]@delta
                yaw=abs((b[6]-gt[6]+np.pi)%(2*np.pi)-np.pi)
                rec['metric_error']=dict(XY_m=float(np.linalg.norm(delta[:2])),XYZ_m=float(np.linalg.norm(delta)),optical_depth_signed_m=float(opt[2]),optical_lateral_signed_m=float(opt[0]),radial_signed_m=float(np.linalg.norm(b[:2])-np.linalg.norm(gt[:2])),center_Z_signed_m=float(delta[2]),dimensions_abs_lwh_m=np.abs(b[3:6]-gt[3:6]).tolist(),yaw_absolute_deg=float(np.degrees(yaw)) if g['type']!='cone' else None,yaw_axis_deg=float(np.degrees(min(yaw,np.pi-yaw))) if g['type']!='cone' else None,upright_3d_iou=upright_iou(b,gt),best_view=ORDER[ci])
            associations.append(rec)
        if token in CFG['audit_tokens']:
            matched={pi for pi,gi in f['curves']['0.1']['natural']['pairs']};candidates=[]
            for cls in ['vehicle','person','cone','unsupported_barrier']:
                candidates += [i for i in np.argsort(-np.array(pred['scores'])) if pred['scores'][i]>=.1 and i not in matched and coarse(pred['labels'][i])==cls][:3]
            for pi in candidates[:12]:
                b=np.array(pred['boxes'][pi]);cls=coarse(pred['labels'][pi]);vv=[]
                for ci in range(6):
                    bounds=proj(pi,ci)
                    if bounds is None:continue
                    x0,y0,x1,y1=np.rint(bounds).astype(int);pixels=semantic[ci][y0:y1,x0:x1]
                    if not pixels.size:continue
                    tags,counts=np.unique(pixels,return_counts=True);hist={str(int(t)):int(n) for t,n in zip(tags,counts)}
                    best=max([(pixel_support(m,bounds)['bbox_iou'],gj) for (gj,cj),m in masks.items() if cj==ci] or [(0.,None)])
                    # Semantic classes match CARLA0.9.15.CityObjectLabel contract, retained raw IDs.
                    compatible_ids={'vehicle':[14,15,16,17,18,19],'person':[12,13],'cone':[21],'unsupported_barrier':[4,5,28]}[cls]
                    frac=float(np.isin(pixels,compatible_ids).mean())
                    vv.append(dict(camera=ORDER[ci],camera_index=ci,bounds=bounds.tolist(),area=int(pixels.size),semantic_histogram=hist,class_compatible_fraction=frac,best_recorded_instance_bbox_iou=best[0],best_recorded_gt=best[1],observed_render_ids=int(len(np.unique(instance[ci][y0:y1,x0:x1])))))
                dup=next((j for j in range(pi) if np.allclose(b,np.array(pred['boxes'][j]),atol=1e-6,rtol=0)),None)
                identity=[r for r in associations if r['token']==token and r['prediction_index']==pi and r['grade']=='IdentitySupported']
                if dup is not None:status='DuplicateHypothesis'
                elif identity:status='ConfirmedPhysicalObject'
                elif cls=='unsupported_barrier':status='UnsupportedClass'
                else:status='AmbiguousOrUnlabeled'
                background.append(dict(token=token,split=f['split'],group=f['group'],frame_id=packet['frame_id'],timestamp_s=packet['timestamp_s'],prediction_index=int(pi),score=pred['scores'][pi],class_id=pred['labels'][pi],coarse_class=cls,predicted_range_m=float(np.linalg.norm(b[:2])),status=status,duplicate_prediction=dup,views=vv,interpretation='ConfirmedPhysicalObject means existing2Dmultiview support for recorded rendered object, NOT confirmed3Dlocation. UnsupportedClass=barrier taxonomy missing in recorded actor benchmark; not model unsupported. NoConfirmedFalsePositive inference.',manual_review='NotHumanReviewed'))
    write(ROOT/'association_ledger.json',dict(associations=associations,support=support,background=background,rgb_stats=rgb_stats,GT_reference_audit=reference_audits))
    meta=provenance('OfflineActualRecordedCARLA_EvaluatorOnly')
    groupcounts={s:dict(Counter(r['grade'] for r in associations if r['split']==s)) for s in ['train','val','calibration','sealed']}
    strata={}
    for name,select in [('all',lambda r:True),('far20to40',lambda r:20<=r['range_m']<40),*[(c,lambda r,c=c:r['type']==c) for c in ['vehicle','person','cone']]]:
        rr=[r for r in associations if select(r)];strata[name]=dict(known_targets=len(rr),grades=dict(Counter(r['grade'] for r in rr)),supported_groups=len({r['group'] for r in rr if r['grade']=='IdentitySupported'}),projection_proxy_localization='NotReported_NoTrustedIdentity')
    write('reports/m8c_2d_3d_association_audit.json',dict(**meta,rule=CFG['identity_rule'],strata=strata,split_grades=groupcounts,raw_topK_reference=asset('reports/m8b_candidate_burden.json'),original_natural_denominator=155,original_projection_pairs=92,rows=associations,limits=['NoGTposition used tograde identity','IdentitySupported is automated multiview mask/bbox evidence, not certifiedhuman annotation or object mesh identity','Conditional proxy preserved via immutable M8B ledger; no new GT errors assigned to ambiguous/untrusted candidates','Calibration and historicalsealed only fixed-rule descriptive audit; no selection or threshold fitting','NoSupportedAssociation means no original score0.1 projectionpair, not proofthatRGB lacks recognizability']))
    error_groups={}
    for name,select in [('all',lambda r:True),('far20to40',lambda r:20<=r['range_m']<40),*[(s,lambda r,s=s:r['split']==s) for s in ['train','val','calibration','sealed']]]:
        rr=[r for r in associations if select(r)];valid=[r for r in rr if r['metric_error'] is not None]
        fields=['XY_m','XYZ_m','optical_depth_signed_m','optical_lateral_signed_m','radial_signed_m','center_Z_signed_m','upright_3d_iou','yaw_absolute_deg','yaw_axis_deg']
        out=dict(total_known=len(rr),identity_supported=len(valid),excluded=len(rr)-len(valid),groups=len({r['group'] for r in valid}),statistics={k:stats([r['metric_error'][k] for r in valid if r['metric_error'][k] is not None]) for k in fields},over5_rate=float(np.mean([r['metric_error']['XY_m']>5 for r in valid])) if valid else None,depth_abs=stats([abs(r['metric_error']['optical_depth_signed_m']) for r in valid]),lateral_abs=stats([abs(r['metric_error']['optical_lateral_signed_m']) for r in valid]),depth_dominates_count=sum(abs(r['metric_error']['optical_depth_signed_m'])>abs(r['metric_error']['optical_lateral_signed_m']) for r in valid),dimension_abs_lwh=[stats([r['metric_error']['dimensions_abs_lwh_m'][i] for r in valid]) for i in range(3)])
        error_groups[name]=out
    write('reports/m8c_depth_lateral_error.json',dict(**meta,groups=error_groups,association_scope='OnlyIdentitySupported; inferred identity support not absoluteproof. No errors for untrusted/ambiguous hypotheses; actual denominator computed in groups.',rows=[r for r in associations if r['metric_error'] is not None],historical_untrusted_proxy_metrics=asset('reports/m8b_localization_metrics.json')))
    write('reports/m8c_annotation_completeness.json',dict(**meta,selection=CFG['annotation_sampling'],frozen_tokens=CFG['audit_tokens'],sampled_candidates=len(background),statuses=dict(Counter(r['status'] for r in background)),entire_M8B_unmatched=11045,unknown_population='No extrapolation from26diagnostic selectedframes; all11045lackjoint3Dmatch, not allconfirmedFP',GT_reference=dict(views=len(reference_audits),purity=stats([r['purity'] for r in reference_audits]),snapshot_frame_mismatches=sum(r['snapshot_frame']!=r['rgb_frame'] for r in reference_audits),corner_residual_m=stats([r['corner_residual_m'] for r in reference_audits])),records=background,ConfirmedUnmatchedBackground=0,reason='Semantic road/building overlap cannot prove absence of a small or occluded object; no automatic physicalFP confirmation. No complete staticmesh/object annotation. Explicit review sheets follow.',ledger=asset(ROOT/'association_ledger.json')))
    write('reports/m8c_crop_and_fov_analysis.json',dict(**meta,support_rows=support,classes={c:dict(views=sum(r['type']==c for r in support),native_area=stats([r['native_pixels'] for r in support if r['type']==c]),retained_area=stats([r['retained_pixels'] for r in support if r['type']==c]),retained_height=stats([r['native_visible_height_px'] for r in support if r['type']==c])) for c in ['vehicle','person','cone']},distance={name:dict(views=sum(lo<=r['range_m']<hi for r in support),native_area=stats([r['native_pixels'] for r in support if lo<=r['range_m']<hi]),height=stats([r['native_visible_height_px'] for r in support if lo<=r['range_m']<hi])) for name,lo,hi in [('near',0,10),('mid',10,20),('far',20,40)]},RGB_statistics=rgb_stats,appearance='CARLA descriptiveRGB moments only. No authenticated matchednuScenesRGB cohort; color/exposuredomain discrepancy unresolved, no causal claim.',crop=dict(native_rows_removed=151,nominal_valid_resized_ratio=512/844,source_information_not_increased_by_resize=True,full_target_loss=0,partial_view_loss=48,source_support_loss=133807/938041),causal='Far targets retainobservedpixels; crop14.26%notsolecause. Noaugmentationablation.'))
    print('grades',strata,'background',Counter(r['status'] for r in background),flush=True)
if __name__=='__main__':main()
