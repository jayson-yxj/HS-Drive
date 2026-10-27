"""Offline A/B diagnostics, fixed metrics, private object-level provenance."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.m8j7_metrics import match,summarize,average_precision,errors,distribution,prediction_mask
from training.m8j8_diagnostics import duplicate_groups,vertical_errors


def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stripped(r):return {k:r[k] for k in ('boxes','scores','labels','gt','gt_labels')}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    text=json.dumps(d,indent=2,allow_nan=False,ensure_ascii=False,default=scalar)
    with Path(p).open('x') as f:f.write(text+'\n')


def paired(recordsA,recordsB,classes):
    common=onlyA=onlyB=neither=0;changes=defaultdict(list);class_changes={}
    for ci,c in enumerate(classes):class_changes[c]=dict(common=0,A_only=0,B_only=0,neither=0)
    for a,b in zip(recordsA,recordsB):
        assert np.array_equal(a['gt'],b['gt']) and np.array_equal(a['gt_labels'],b['gt_labels'])
        ma=match(**stripped(a));mb=match(**stripped(b));aa={j:i for i,j in ma['pairs']};bb={j:i for i,j in mb['pairs']}
        both=set(aa)&set(bb);only_a=set(aa)-set(bb);only_b=set(bb)-set(aa);none=set(range(len(a['gt'])))-set(aa)-set(bb)
        common+=len(both);onlyA+=len(only_a);onlyB+=len(only_b);neither+=len(none)
        for ci,c in enumerate(classes):
            for name,ids in [('common',both),('A_only',only_a),('B_only',only_b),('neither',none)]:
                class_changes[c][name]+=sum(a['gt_labels'][j]==ci for j in ids)
        for j in sorted(both):
            ea=errors(a['boxes'],a['gt'],[(aa[j],j)],a['gt_labels'],classes)
            eb=errors(b['boxes'],b['gt'],[(bb[j],j)],b['gt_labels'],classes)
            ea['height_abs_m']=[abs(float(a['boxes'][aa[j],5]-a['gt'][j,5]))]
            eb['height_abs_m']=[abs(float(b['boxes'][bb[j],5]-b['gt'][j,5]))]
            for k in ea:
                if ea[k]:
                    changes[k+'_A'].extend(ea[k]);changes[k+'_B'].extend(eb[k])
                    changes[k+'_B_minus_A'].append(eb[k][0]-ea[k][0])
    return dict(common=common,A_only=onlyA,B_only=onlyB,neither=neither,per_class_counts=class_changes,
        paired_errors={k:distribution(v) for k,v in changes.items()},
        direction_counts={k:dict(improved=sum(x<0 for x in v),worsened=sum(x>0 for x in v),equal=sum(x==0 for x in v))
                          for k,v in changes.items() if k.endswith('_B_minus_A')})


def vertical_summary(records,classes):
    values=defaultdict(list);per_class={c:defaultdict(list) for c in classes}
    for r in records:
        m=match(**stripped(r))
        for i,j in m['pairs']:
            p,g=r['boxes'][i],r['gt'][j];c=classes[int(r['gt_labels'][j])]
            row=dict(signed_dz_m=float(p[2]+p[5]/2-g[2]-g[5]/2),signed_dh_m=float(p[5]-g[5]),
                abs_width_m=abs(float(p[3]-g[3])),abs_length_m=abs(float(p[4]-g[4])))
            row['abs_dz_m']=abs(row['signed_dz_m']);row['abs_dh_m']=abs(row['signed_dh_m'])
            for k,v in row.items():values[k].append(v);per_class[c][k].append(v)
    return dict(all={k:distribution(v) for k,v in values.items()},
        per_class={c:{k:distribution(v[k]) for k in values} for c,v in per_class.items()})


def metrics(records,classes,protocol):
    rr=[stripped(r) for r in records]
    return dict(primary=summarize(rr,classes,.2,2.),strict_05m=summarize(rr,classes,.2,.5),
        strict_1m=summarize(rr,classes,.2,1.),PR=[summarize(rr,classes,t,2.) for t in protocol['evaluation']['PR_score_thresholds']],
        AP=average_precision(rr,classes),vertical=vertical_summary(records,classes))


def residuals(root,classes):
    old=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    traces={i:[t for t in read(old/'supervision'/f'{i:02}_trace.json') if t['retained']] for i in range(8)}
    rs={model:[load(root/model/'residual'/f'{i:02}.npz') for i in range(8)] for model in ('A','B')}
    heights=[];misses=[];duplicates=[];multi=[];case_keys=set();case_instances=set()
    for index,pi,gi in [(2,40,23),(1,64,18),(3,38,21)]:
        r=rs['B'][index];g=r['gt'][gi];p=r['boxes'][pi];q=int(r['query_indices'][pi]);raw=r['raw_bbox'][q];trace=traces[index][gi]
        assert (pi,gi) in match(**stripped(r))['pairs']
        center=float(g[2]+g[5]/2);height=float(g[5]);pred_center=float(p[2]+p[5]/2)
        assert abs(center-trace['converted_reference_center'][2])<1e-5
        assert abs(height-trace['raw_full_wlh'][2])<1e-6
        assert abs(float(raw[4])-pred_center)<1e-6 and abs(float(np.exp(raw[5]))-p[5])<1e-6
        d=dict(frame=index,annotation_token=trace['annotation_token'],instance_token=trace['instance_token'],
            query_index=q,prediction_index=pi,GT_index=gi,raw_GT=trace,
            training_box_bottom_z=float(g[2]),training_center_z=center,training_full_height=height,
            training_encoded_cz=center,training_encoded_logh=float(np.log(height)),
            network_cz=float(raw[4]),network_logh=float(raw[5]),decoded_center_z=pred_center,decoded_h=float(p[5]),
            box_object_bottom_z=float(p[2]),display_bottom_z=float(p[2]),display_top_z=float(p[2]+p[5]),
            delta_z=pred_center-center,delta_h=float(p[5]-height),xy_error=float(np.linalg.norm(p[:2]-g[:2])),
            source_to_display_contract='Consistent_NoHalfHeightOrUnitCorrectionApplied',
            preliminary_cause='PredictedCenterZError_NotConversionShift')
        heights.append(d);case_keys.add(trace['annotation_token']);case_instances.add(trace['instance_token'])
    for index,r in enumerate(rs['B']):
        m=match(**stripped(r));d=duplicate_groups(r,r['query_indices'])
        for row in d['details']:
            gt=traces[index][row['nearest_GT_index']]
            duplicates.append(dict(frame=index,annotation_token=gt['annotation_token'],instance_token=gt['instance_token'],**row))
        multi.extend(dict(frame=index,**x) for x in d['cross_class_same_query_groups'])
        for gi in m['fn']:
            g=r['gt'][gi];ci=int(r['gt_labels'][gi]);trace=traces[index][gi]
            full=r['raw_bbox'];score=1/(1+np.exp(-r['raw_cls'][:,ci]));xy=full[:,:2]
            ds=np.linalg.norm(xy-g[:2],axis=1);nearest=int(np.argmin(ds))
            decoded_ids=np.where(r['labels']==ci)[0]
            close=[int(i) for i in decoded_ids if np.linalg.norm(r['boxes'][i,:2]-g[:2])<2.]
            confident=[i for i in close if r['scores'][i]>=.2 and prediction_mask(r['boxes'],r['scores'],.2)[i]]
            if confident:reason='OneToOneAssociationCompetition'
            elif close:reason='DecodedWithin2mBelowFixedScore'
            elif ds.min()<2:
                best=int(np.argmax(np.where(ds<2,score,-1)))
                flat=best*10+ci
                reason='NearbyRawQueryClassNotRetainedByTop300OrDecoderRange' if flat not in r['flat_indices'] else 'InsufficientDecodedEvidence'
            else:reason='NoRawQueryCenterWithin2m_LocalizationLimitation'
            row=dict(frame=index,annotation_token=trace['annotation_token'],instance_token=trace['instance_token'],
                GT_index=gi,class_name=classes[ci],reason=reason,nearest_raw_query=nearest,
                nearest_raw_xy_distance=float(ds[nearest]),nearest_raw_GT_class_score=float(score[nearest]),
                nearest_raw_argmax_class=classes[int(r['raw_cls'][nearest].argmax())],
                decoded_same_class_within2m=close,decoded_confident_within2m=confident,
                GT_filtering_reasons=trace['filtering_reasons'])
            misses.append(row);case_keys.add(trace['annotation_token']);case_instances.add(trace['instance_token'])
    assert len(heights)==3 and len(misses)==4 and len(duplicates)==56
    # Every duplicated prediction has its own decoder query within the same class.
    shared_sameclass=sum(x['query_index'] in x['matched_query_indices'] for x in duplicates)
    counts=Counter(classes[int(rs['B'][x['frame']]['labels'][x['prediction_index']])] for x in duplicates)
    groups={(x['frame'],x['annotation_token']) for x in duplicates}
    private=dict(height_cases=heights,misses=misses,duplicate_predictions=duplicates,same_query_multiple_classes=multi,
        case_unique_annotation_records=len(case_keys),case_unique_physical_instances=len(case_instances))
    write(root/'residual_object_audit.json',private)
    return dict(height_observations=3,height_unique_physical_instances=len({h['instance_token'] for h in heights}),
        height_values=[{k:h[k] for k in ['frame','delta_z','delta_h','xy_error']} for h in heights],
        full_numeric_chain_private_path=str(root/'residual_object_audit.json'),
        height_source='ThreeConsecutiveObservationsOfSameAnnotatedPerson',
        unique_case_annotation_records=len(case_keys),unique_case_instances=len(case_instances),
        missed=4,miss_reasons=dict(Counter(m['reason'] for m in misses)),
        miss_summary=[{k:m[k] for k in ['frame','class_name','reason','nearest_raw_xy_distance','nearest_raw_GT_class_score','nearest_raw_argmax_class']} for m in misses],
        duplicate_predictions=56,duplicate_groups=len(groups),duplicate_target_instances=len({x['instance_token'] for x in duplicates}),
        duplicate_by_class=dict(counts),same_class_duplicate_same_query=shared_sameclass,
        ambiguous_duplicate_predictions=sum(x['ambiguous_GT_count']>1 for x in duplicates),
        multi_class_same_query_groups=len(multi),query_identity='OnlyWithinForward_NotStableObjectIdentity',
        private_audit_sha256=sha(root/'residual_object_audit.json'),
        cause='ZOutputErrorsWithConsistentConversion;DetectionAndAssignmentResiduals',
        historical_J7_predictions_match=True,GT_or_metric_change_applied=False)


def main():
    cfg=read('configs/m8j8_protocol.json');root=Path(cfg['prepared_root']).parent
    reports=Path('reports/m8j8_sources')/cfg['run_id'];sup=read(reports/'supervisor.json')
    assert sup['status']=='ReadOnlyABCompleted'
    prep=read(root/'prepared/manifest.json');frames=prep['frames'];classes=prep['training_classes']
    runs={m:read(reports/m/'result.json') for m in ('A','B')}
    for m,r in runs.items():
        assert r['status']=='ReadOnlyEvaluationPassed' and r['optimizer_steps']==r['backward_calls']==0
        assert r['initial_state_fingerprint']==r['final_state_fingerprint']
        for f in r['frames']:assert sha(f['cache_path'])==f['cache_sha256']
    records={m:[load(root/m/'mini_val'/f'{f["index"]:02}.npz') for f in frames] for m in ('A','B')}
    aggregate={m:metrics(rs,classes,cfg) for m,rs in records.items()}
    scene={}
    for name in sorted({f['scene_name'] for f in frames}):
        ids=[f['index'] for f in frames if f['scene_name']==name]
        subsets={m:[rs[i] for i in ids] for m,rs in records.items()}
        scene[name]=dict(frames=len(ids),**{m:metrics(rs,classes,cfg) for m,rs in subsets.items()},
            paired=paired(subsets['A'],subsets['B'],classes))
    both=paired(records['A'],records['B'],classes)
    write('reports/m8j8_ab_statistics.json',dict(scope='All81MiniValKeyframes_ProjectDiagnostic_DevelopmentOnly',
        protocol_sha256=sha('configs/m8j8_protocol.json'),aggregate=aggregate,scenes=scene,paired=both))
    write('reports/m8j8_residual_attribution.json',residuals(root,classes))
    # Per-frame aggregate metrics only; no object labels published.
    write('reports/m8j8_per_frame.json',dict(frames=[dict(index=f['index'],sample_token=f['sample_token'],scene=f['scene_name'],
        A=summarize([stripped(records['A'][f['index']])],classes),
        B=summarize([stripped(records['B'][f['index']])],classes)) for f in frames]))
    # All per-object pairs and loss/gain identities stay private.
    object_comparison=[];tails=[]
    for f,a,b in zip(frames,records['A'],records['B']):
        ma=match(**stripped(a));mb=match(**stripped(b));ia={j:i for i,j in ma['pairs']};ib={j:i for i,j in mb['pairs']}
        trace=[x for x in read(root/'prepared/supervision'/f'{f["index"]:02}_trace.json') if x['retained']]
        for j,t in enumerate(trace):
            row=dict(frame=f['index'],annotation_token=t['annotation_token'],instance_token=t['instance_token'],GT_index=j,
                A_prediction=ia.get(j),B_prediction=ib.get(j))
            object_comparison.append(row)
            for m,indices,r in [('A',ia,a),('B',ib,b)]:
                if j in indices:
                    i=indices[j];xy=float(np.linalg.norm(r['boxes'][i,:2]-r['gt'][j,:2]))
                    tails.append(dict(model=m,frame=f['index'],class_name=classes[int(r['gt_labels'][j])],xy_m=xy,
                        annotation_token=t['annotation_token'],prediction_index=i,GT_index=j))
    write(root/'mini_val_object_comparison.json',object_comparison)
    write(root/'mini_val_xy_tails.json',sorted(tails,key=lambda x:-x['xy_m']))
    out=Path('reports/m8j8_figures');out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for m in ('A','B'):
        axes[0].plot([p['recall'] for p in aggregate[m]['PR']],[p['precision'] for p in aggregate[m]['PR']],marker='o',label=m)
    axes[0].set_xlabel('Recall');axes[0].set_ylabel('Precision');axes[0].set_title('Fixed threshold PR, XY <2m');axes[0].legend()
    x=np.arange(len(classes));w=.35
    for m,shift in [('A',-w/2),('B',w/2)]:
        vals=[aggregate[m]['primary']['per_class'][c]['recall'] for c in classes]
        axes[1].bar(x+shift,[np.nan if v is None else v for v in vals],w,label=m)
    axes[1].set_xticks(x);axes[1].set_xticklabels(classes,rotation=75,ha='right');axes[1].set_title('Class recall; absent GT omitted');axes[1].legend()
    for m,shift in [('A',-w/2),('B',w/2)]:
        axes[2].bar(np.arange(2)+shift,[scene[s][m]['primary']['precision'] for s in scene],w,label=m)
    axes[2].set_xticks(np.arange(2));axes[2].set_xticklabels(list(scene));axes[2].set_title('Scene precision @score 0.2');axes[2].legend()
    fig.suptitle('M8-J8 mini_val 81 frames: project diagnostics, not official scores')
    fig.tight_layout();fig.savefig(out/'ab_comparison.png',dpi=150);plt.close(fig)
    html='<meta charset="utf-8"><h1>M8-J8 private A/B comparison</h1><p>A original; B step400. Fixed first/middle/last frames. GT green, predictions orange.</p>'
    for i in (0,20,39,40,60,80):
        html+=f'<h2>mini_val frame {i}</h2>'
        for m in ('A','B'):html+=f'<h3>{m}</h3><img width="1500" src="{m}/mini_val/{i:02}_six.jpg"><img width="700" src="{m}/mini_val/{i:02}_bev.jpg">'
    (root/'comparison.html').write_text(html)
    print(json.dumps({m:{k:aggregate[m]['primary'][k] for k in ['GT','TP','FP','FN','precision','recall','duplicates']} for m in ('A','B')},indent=2))


if __name__=='__main__':main()
