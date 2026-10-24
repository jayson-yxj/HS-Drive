"""Separate GT evaluator; original annotations are never inputs to PETR."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import spearmanr
from tools.m8b_common import *
from datasets.m8b_petr_adapter import corners_hs,ORDER,A
from diagnostics.m8b_metrics import *

def load_gt(index,caches):
    row=index['row'];sample=row['sample_id'];p=json.loads(Path(index['packet']['path']).read_text())
    raw=json.loads(Path(row['raw_frame']).read_text());base=Path(row['raw_frame']).parent.parent
    views=[a for a in caches[row['split']]['actors'] if a['seed']==101 and a['sample_id']==sample]
    with np.load(row['pixel_label'],allow_pickle=False) as z:depth=z['depth']
    byactor={};crop=[]
    for ci,c in enumerate(row['actors']):
        relevant=[v for v in views if v['camera_index']==ci]
        if not relevant:continue
        with np.load(base/raw['cameras'][ORDER[ci]]['instance'],allow_pickle=False) as z:idsimg=z['instance']
        claims=Counter(v['render_id'] for a in c['actors'] for v in a['identity_candidates'] if v['accepted'])
        for v in relevant:
            aa=next(a for a in c['actors'] if a['actor_id']==v['actor_id'])
            ids=[q['render_id'] for q in aa['identity_candidates'] if q['accepted'] and claims[q['render_id']]==1]
            mask=np.isin(idsimg,ids)&np.isfinite(depth[ci])&(depth[ci]>=1.5)&(depth[ci]<80)
            assert int(mask.sum())==v['native_visible_pixels'],(sample,ci,mask.sum(),v['native_visible_pixels'])
            vv,uu=np.where(mask);keep=vv*2.2>=332
            retained=int(keep.sum())
            actor=next(a for a in raw['actors'] if a['actor_id']==v['actor_id']);corners=np.array(actor['corners_hs_ego_m']);edge=corners[4]-corners[0];yaw=np.arctan2(edge[1],edge[0]);box=np.r_[v['box_center_ego'],v['bbox_dimensions_xyz'],yaw,0,0]
            approximate=corners_hs(box);corner_error=float(np.min(np.linalg.norm(corners[:,None]-approximate[None],axis=-1),axis=1).max())
            # Geometry projected reference box; retained rendered pixels only establish evaluability.
            bounds=project_bounds(corners,np.array(p['cameras'][ci]['K']),np.array(p['cameras'][ci]['T_cam_to_ego']))
            view=dict(camera=ORDER[ci],camera_index=ci,native_pixels=int(mask.sum()),retained_pixels=retained,retained_fraction=retained/int(mask.sum()),bounds_aug=None if bounds is None else bounds.tolist(),visibility_proxy=v['visibility_ratio_proxy'],truncated=v['truncated'],pitch_deg=v['pitch_deg'])
            if v['actor_id'] not in byactor:
                byactor[v['actor_id']]=dict(actor_id=v['actor_id'],type=v['type'],type_id=v['type_id'],box=box.tolist(),corners=corners.tolist(),upright_corner_error_m=corner_error,views=[])
            byactor[v['actor_id']]['views'].append(view)
            crop.append(dict(token=index['token'],sample_id=sample,group=row['group'],split=row['split'],actor_id=v['actor_id'],type=v['type'],**view))
    for g in byactor.values():g['crop_retained']=any(v['retained_pixels']>=3 for v in g['views'])
    return list(byactor.values()),crop,p

def loc_summary(rows):
    return dict(xy_m=summary([r['xy'] for r in rows]),optical_depth_abs_m=summary([r['optical_depth_abs'] for r in rows]),optical_lateral_abs_m=summary([r['optical_lateral_abs'] for r in rows]),radial_abs_m=summary([r['radial_abs'] for r in rows]),dimension_mae_lwh=np.mean([r['dimension_abs'] for r in rows],axis=0).tolist() if rows else None,yaw_abs_deg=summary([r['yaw_abs_deg'] for r in rows]),yaw_axis_abs_deg=summary([r['yaw_axis_abs_deg'] for r in rows]),upright_3d_iou=summary([r['upright_iou3d'] for r in rows]))

def main():
    index=json.loads((ROOT/'evaluation_index.json').read_text());caches={s:json.loads(Path('runs/m5o_cache_'+s+'/supervision.json').read_text()) for s in ['train','val','calibration','sealed']}
    frames=[];crops=[];crop_gt=[]
    for item in index:
        p=json.loads((ROOT/'forward'/ (item['token']+'.json')).read_text());g,cv,packet=load_gt(item,caches);crops+=cv
        matrix=conditional_matrix(p,g,packet);r=item['row']
        record=dict(token=item['token'],sample_id=r['sample_id'],group=r['group'],split=r['split'],map=r['map'],core40=item['core40'],gts=g,score_classes=dict(Counter(coarse(v) for v in p['labels'])),curves={})
        for threshold in CFG['matching']['score_grid']:
            pairs=conditional(p,g,matrix,threshold);e=errors(p,g,pairs,packet)
            record['curves'][str(threshold)]=dict(natural=natural(p,g,threshold),natural_crop=natural(p,g,threshold,crop_only=True),conditional=e)
        record['sensitivity_xy4m']=natural(p,g,.1,4.)
        record['natural_matched_localization']=errors(p,g,record['curves']['0.1']['natural']['pairs'],packet)
        record['all_scores']=p['scores'];record['all_labels']=p['labels']
        frames.append(record)
    write(ROOT/'evaluation_ledger.json',dict(frames=frames,crop_views=crops))
    meta=provenance('OfflineEvaluator_ActualRecordedCARLA_GT_SeparateFromGPUInference')
    groups={'all':frames,'core40':[f for f in frames if f['core40']],**{s:[f for f in frames if f['split']==s] for s in caches}}
    detection={};localization={}
    for key,ff in groups.items():
        detection[key]=dict(frames=len(ff),groups=len({f['group'] for f in ff}),score0_1=aggregate_natural([f['curves']['0.1']['natural'] for f in ff]),raw_topK=aggregate_natural([f['curves']['0.0']['natural'] for f in ff]),crop_retained_score0_1=aggregate_natural([f['curves']['0.1']['natural_crop'] for f in ff]),xy4m_sensitivity_score0_1=aggregate_natural([f['sensitivity_xy4m'] for f in ff]))
        localization[key]=dict(projection_association_score0_1=loc_summary([r for f in ff for r in f['curves']['0.1']['conditional']]),projection_association_raw_topK=loc_summary([r for f in ff for r in f['curves']['0.0']['conditional']]),natural_2m_truncated=loc_summary([r for f in ff for r in f['natural_matched_localization']]))
    # Independent GT denominators: multi-camera views are repeated observations, not independent targets.
    strata=defaultdict(list)
    for f in frames:
        matched={j for i,j in f['curves']['0.1']['natural']['pairs']};cond={r['gt_index']:r for r in f['curves']['0.1']['conditional']}
        for j,g in enumerate(f['gts']):
            dist=np.linalg.norm(g['box'][:2]);rng='0-10' if dist<10 else '10-20' if dist<20 else '20-40' if dist<40 else '40+'
            keys=['class/'+g['type'],'distance/'+rng,'group/'+f['group'],'map/'+f['map'],'split/'+f['split'],'crop/'+('retained' if g['crop_retained'] else 'lost'),'overlap/'+('multi' if len(g['views'])>1 else 'single')]
            keys+=['camera/'+v['camera'] for v in g['views']]
            keys+=['direction/'+d for d in sorted(set('front' if v['camera_index']<3 else 'rear' for v in g['views']))]
            vis=max(v['visibility_proxy'] for v in g['views']);keys+=['visibility/'+('low_below0.3' if vis<.3 else 'supported0.3+')]
            keys+=['truncation/'+('yes' if any(v['truncated'] for v in g['views']) else 'no')]
            for k in keys:strata[k].append(dict(group=f['group'],detected=j in matched,xy=cond[j]['xy'] if j in cond else None))
    strata={k:dict(gt=len(v),groups=len({r['group'] for r in v}),TP=sum(r['detected'] for r in v),recall=sum(r['detected'] for r in v)/len(v),conditional_xy=summary([r['xy'] for r in v if r['xy'] is not None]),support='DescriptiveOnly' if len(v)>=20 and len({r['group'] for r in v})>=3 else 'InsufficientSupport') for k,v in strata.items()}
    write('reports/m8b_detection_metrics.json',dict(**meta,protocol=CFG['matching'],groups=detection,strata=strata,ledger=asset(ROOT/'evaluation_ledger.json'),limitations=['FP means unmatched recorded annotation, not proven physical false positive','Actor identity via renderer/OBB proxy; not independent mesh ground truth','All historical sealed groups previously opened; PilotOnly','At score0 all300 topK creates substantial unmatched burden; matching these is not operational detection']))
    write('reports/m8b_localization_metrics.json',dict(**meta,groups=localization,association='Projected 2D OBB IoU>=0.1 in visible retained cameras, class compatible, one-to-one, no center cutoff. Proxy identity can be wrong under overlapping objects. Natural2m subset is explicitly truncated and cannot establish tail risk.',geometry='Upright3DIoU is yaw-only OBB overlap approximation for pitched references, not mesh IoU',upright_reference_approx_corner_error_m=summary([g['upright_corner_error_m'] for f in frames for g in f['gts']]),velocity='NotEvaluated_NoReliableVelocityReference'))
    allscores=[s for f in frames for s in f['all_scores']]
    burden=dict(raw_candidates=len(allscores),score_distribution=summary(allscores),score_by_class={str(k):summary([s for f in frames for s,l in zip(f['all_scores'],f['all_labels']) if l==k]) for k in range(10)},score0_1=detection['all']['score0_1'],empty_annotation_frames=[dict(token=f['token'],count_score0_1=f['curves']['0.1']['natural']['predictions'],max_score=max(f['all_scores'])) for f in frames if not f['gts']])
    write('reports/m8b_candidate_burden.json',dict(**meta,**burden,unmatched_status='Unknown: incomplete annotations include static background; cannot claim every unmatched candidate is physical FP',all_outputs=[asset(ROOT/'forward'/(f['token']+'.json')) for f in frames]))
    crop_summary=dict(actor_views=len(crops),partially_cropped=sum(v['retained_pixels']<v['native_pixels'] for v in crops),fully_lost_views=sum(v['retained_pixels']<3 for v in crops),total_original_pixels=sum(v['native_pixels'] for v in crops),total_retained_pixels=sum(v['retained_pixels'] for v in crops),original_gt_actors=sum(len(f['gts']) for f in frames),fully_lost_gt_actors=sum(not g['crop_retained'] for f in frames for g in f['gts']))
    write('reports/m8b_crop_visibility_audit.json',dict(**meta,summary=crop_summary,augmentation=dict(A=A.tolist(),source=[384,640],resized=[844,1408],output=[512,1408],crop=[0,332,1408,844],retained_native_v_min=332/2.2,nominal_scale_rounding='844 vs844.8; official pixel interpolation differs subpixel from idealnominal scale, preserved'),views=crops,mask_definition='Reliable unique render IDs intersect finite optical GT depth[1.5,80), exactly historical M5O>=3pixels. Native pixel-center retention v*2.2>=332; interpolation boundary uncertainty <1 native pixel. Evaluator-only.'))
    # Calibration-only risk decision; all other group curves remain descriptive regression.
    risk={}
    for split,ff in groups.items():
        curve=[]
        for th in CFG['risk']['fixed_grid']:
            nn=aggregate_natural([f['curves'][str(th)]['natural'] for f in ff]);rr=[r for f in ff for r in f['curves'][str(th)]['conditional']];ng=len({f['group'] for f in ff if f['curves'][str(th)]['conditional']});xy=summary([r['xy'] for r in rr]);coverage=len(rr)/nn['gt'] if nn['gt'] else 0
            good=coverage>=.3 and len(rr)>=20 and ng>=3 and xy['p95']<=5 and xy['over5_rate']<=.05 and (nn['precision'] or 0)>=.5
            curve.append(dict(threshold=th,conditional_coverage=coverage,matched=len(rr),matched_groups=ng,xy=xy,natural=nn,requirements_met=good))
        rr=[r for f in ff for r in f['curves']['0.0']['conditional']]
        corr=spearmanr([r['score'] for r in rr],[r['xy'] for r in rr]) if len(rr)>2 else None
        risk[split]=dict(curve=curve,score_error_spearman=float(corr.statistic) if corr is not None and np.isfinite(corr.statistic) else None)
    qualified=[c['threshold'] for c in risk['calibration']['curve'] if c['requirements_met']]
    threshold=min(qualified) if qualified else None
    risk['decision']=dict(threshold=threshold,status='CalibrationThresholdCandidate_PilotOnly' if qualified else 'NoGo_NoQualifiedRiskCoverage',selection='CalibrationOnly',inference_inputs='class scores only, not GTerrors',no_detection='Unknown_not_Free',test_used_for_selection=False,confidence_is_safety_probability=False)
    # Cluster bootstrap uncertainty, recompute descriptive precision/recall by location not pixels.
    rng=np.random.RandomState(8082);boot={}
    for split in ['val','calibration','sealed']:
        ff=groups[split];gg=sorted({f['group'] for f in ff});samples=[]
        for _ in range(1000):
            picked=rng.choice(gg,len(gg),replace=True);nn=aggregate_natural([f['curves']['0.1']['natural'] for g in picked for f in ff if f['group']==g]);samples.append([nn['precision'] or 0,nn['recall'] or 0])
        boot[split]=dict(groups=len(gg),precision_recall_percentile95=np.percentile(samples,[2.5,97.5],axis=0).tolist(),assumption='Resampling observed location clusters; 6 locations and correlated repeated layouts. Descriptive, not population safety coverage.')
    write('reports/m8b_risk_coverage.json',dict(**meta,results=risk,cluster_uncertainty=boot,high_score_severe_examples=sorted([dict(token=f['token'],group=f['group'],split=f['split'],**r) for f in frames for r in f['curves']['0.0']['conditional'] if r['xy']>5],key=lambda r:-r['score'])[:20],low_score_correct_examples=sorted([dict(token=f['token'],group=f['group'],**r) for f in frames for r in f['curves']['0.0']['conditional'] if r['xy']<2],key=lambda r:r['score'])[:20]))
    print(json.dumps(dict(detection=detection['all'],localization=localization['all'],crop=crop_summary,risk=risk['decision']),indent=2))
if __name__=='__main__':main()
