"""Offline intervention evaluation; reuse official J9 export and fixed J8 metrics."""
import os
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8j11a_common import read,write,sha,load_cache,compare_arrays,PROTOCOL,clean
from datasets.m8j9_official_export import read_metadata_only,pose_info,prediction_to_export_input,exporter,compare_export
from tools.m8j8_report import paired,vertical_summary,stripped
from tools.m8j10_report import anomalies,paired_classes
from training.m8j7_metrics import summarize,distribution,match


def quality(records,classes):
    return dict(primary=summarize(records,classes,.2,2.),strict_05m=summarize(records,classes,.2,.5),
                strict_1m=summarize(records,classes,.2,1.),vertical=vertical_summary(records,classes),
                vertical_anomalies=anomalies(records,classes))


def comparison(A,B,classes):
    return dict(pair_field_convention='A means T0; B means intervention; same GT frame subset',
                paired=paired(A,B,classes),per_class_paired=paired_classes(A,B,classes))


def main():
    os.umask(0o077);cfg=read(PROTOCOL);base=read('configs/m8j9_retry1_protocol.json');root=Path(cfg['private_root'])
    out=root/'evaluation';out.mkdir(exist_ok=False);classes=base['classes']
    frames=read(cfg['prepared_manifest'])['frames'];runs={t:read(root/t/'result.json') for t in ('T1','T2','T3')}
    assert all(r['status']=='InterventionSequencePassed' for r in runs.values())
    ledger=[__import__('json').loads(l) for l in (root/'forward_budget.jsonl').read_text().splitlines()]
    assert len(ledger)==sum(r['actual_forwards'] for r in runs.values())==203
    assert len(ledger)<=cfg['budget']['total_max']
    checks=[];memories={}
    for condition,r in runs.items():
        plan=cfg['schedule'][condition]
        assert [x['index'] for x in r['frames']]==[x['index'] for x in plan]
        assert [x['index'] for x in ledger if x['condition']==condition]==[x['index'] for x in plan]
        assert r['initial_state_sha256']==r['final_state_sha256']==read('reports/m8j11_sources/full/result.json')['initial_state_sha256']
        memories[condition]=[__import__('json').loads(l) for l in (root/condition/'memory.jsonl').read_text().splitlines()]
        assert len(memories[condition])==len(plan)
        for row,actual in zip(plan,memories[condition]):
            for k in row: assert actual[k]==row[k]
            assert actual['actual_head_prev_exists']==(0. if row['reset_reason'] else 1.)
            for source in actual['history_before']:
                assert source['scene_token']==row['scene_token'] and source['timestamp_us']<row['timestamp_us']
                assert source['index'] in [p['index'] for p in plan]
            if condition=='T3':assert row['scene_index']%2==0
            expected=[]
            if row['scene_index']==0 or condition=='T2' and row['scene_index']<row['reset_scene_index']:
                expected.append(('T0',cfg['T0_cache'][row['index']]['path']))
            if condition=='T2' and row['scene_index']==row['reset_scene_index']:
                expected.append(('T1',str(root/'T1'/('{:02}.npz'.format(row['index'])))))
            for reference,path in expected:
                delta=compare_arrays(load_cache(root/condition/('{:02}.npz'.format(row['index']))),
                                     load_cache(path),cfg['atol'],cfg['rtol'])
                checks.append(dict(condition=condition,index=row['index'],reference=reference,max_abs_error=delta))
    for p,h in cfg['fixed_assets_sha256'].items():assert sha(p)==h,p
    for f in cfg['T0_cache']:assert sha(f['path'])==f['sha256']
    records={'T0':{}};traces={};sup=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared/supervision')
    for f in frames:
        i=f['index'];r=load_cache(cfg['T0_cache'][i]['path']);g=load_cache(sup/('{:02}.npz'.format(i)))
        r.update(gt=g['boxes'],gt_labels=g['labels']);records['T0'][i]=r
        traces[i]=[x for x in read(sup/('{:02}_trace.json'.format(i))) if x['retained']]
        assert len(traces[i])==len(r['gt'])
    for t in runs:
        records[t]={}
        for row in cfg['schedule'][t]:
            i=row['index'];r=load_cache(root/t/('{:02}.npz'.format(i)))
            r.update(gt=records['T0'][i]['gt'],gt_labels=records['T0'][i]['gt_labels']);records[t][i]=r
    assert sum(len(r['gt']) for r in records['T0'].values())==3559
    t0=quality(list(records['T0'].values()),classes)
    historical=read('reports/m8j11_project_comparison.json')['StreamPETR']
    for key in ('primary','strict_05m','strict_1m','vertical'):assert t0[key]==historical[key],'T0MetricChanged:'+key
    project={};per_frame=[];objects=[]
    for t in runs:
        ids=[x['index'] for x in cfg['schedule'][t]]
        A=[records['T0'][i] for i in ids];B=[records[t][i] for i in ids]
        unique={c:set() for c in classes}
        for i in ids:
            for j,trace in enumerate(traces[i]):unique[classes[int(records['T0'][i]['gt_labels'][j])]].add(trace['instance_token'])
        project[t]=dict(frames=len(ids),T0=quality(A,classes),intervention=quality(B,classes),
            **comparison(A,B,classes),unique_instances_by_class={k:len(v) for k,v in unique.items()},scenes={})
        for scene in cfg['scenes']:
            ii=[i for i in ids if frames[i]['scene_name']==scene['scene_name']]
            aa=[records['T0'][i] for i in ii];bb=[records[t][i] for i in ii]
            project[t]['scenes'][scene['scene_name']]=dict(frames=len(ii),T0=quality(aa,classes),
                intervention=quality(bb,classes),**comparison(aa,bb,classes))
        for i in ids:
            aa=records['T0'][i];bb=records[t][i]
            per_frame.append(dict(condition=t,index=i,scene=frames[i]['scene_name'],
                T0=summarize([aa],classes),intervention=summarize([bb],classes),paired=paired([aa],[bb],classes)))
            maps=[]
            for r in (aa,bb):maps.append({j:k for k,j in match(**stripped(r))['pairs']})
            for j,trace in enumerate(traces[i]):
                objects.append(dict(condition=t,index=i,annotation_token=trace['annotation_token'],instance_token=trace['instance_token'],
                    T0_prediction_index=maps[0].get(j),intervention_prediction_index=maps[1].get(j)))
    write(out/'per_frame.json',per_frame);write(out/'object_pairs.json',objects)
    recovery={}
    for scene in cfg['scenes']:
        rows=[x for x in cfg['schedule']['T2'] if x['scene_name']==scene['scene_name']]
        r=scene['reset_scene_index'];treset=rows[r]['timestamp_us'];seq=[]
        for row in rows[r:]:
            i=row['index'];aa=[records['T0'][i]];bb=[records['T2'][i]]
            seq.append(dict(offset=row['scene_index']-r,elapsed_s=(row['timestamp_us']-treset)/1e6,
                T0=summarize(aa,classes),T2=summarize(bb,classes),paired=paired(aa,bb,classes)))
        recovery[scene['scene_name']]=dict(reset_scene_index=r,selected_offsets=[s for s in seq if s['offset'] in (0,1,2,4)],
            remaining_frames=seq,caution='Two correlated scene resets; individual improvements do not certify stable recovery')
    official={'T0':read('reports/m8j11_sources/metrics_summary.json')};exports={};filters={}
    tables=read_metadata_only(base['dataset_root'],'v1.0-mini')
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.config import config_factory
    from tools.m8j9_stage import audit_official_filters
    nusc=NuScenes(version='v1.0-mini',dataroot=base['dataset_root'],verbose=False)
    for t in ('T1','T2'):
        infos=[];dets=[]
        for f in frames:
            r=records[t][f['index']]
            # Only this explicit prediction dictionary reaches the original exporter.
            det,_=prediction_to_export_input({k:r[k] for k in ('boxes','scores','labels')},classes)
            infos.append(pose_info(f['sample_token'],tables));dets.append(det)
        target=out/t;target.mkdir()
        ds=exporter(infos,classes,base['output_policy']['meta']);path=ds._format_bbox(dets,str(target/'export'))
        payload=read(path);assert set(payload['results'])=={f['sample_token'] for f in frames}
        errors=[]
        for det,info in zip(dets,infos):errors.append(compare_export(det,info,payload['results'][info['token']],classes,ds.eval_detection_configs)[0])
        engine=NuScenesEval(nusc,config_factory('detection_cvpr_2019'),path,'mini_val',str(target/'official'),verbose=False)
        engine.main(plot_examples=0,render_curves=False)
        support,private=audit_official_filters(nusc,base,engine)
        filters[t]=support;write(target/'GT_membership.json',private)
        official[t]=read(target/'official/metrics_summary.json')
        exports[t]=dict(result_path=path,result_sha256=sha(path),
            metrics_summary_sha256=sha(target/'official/metrics_summary.json'),metrics_details_sha256=sha(target/'official/metrics_details.json'),
            max_transform_errors={k:max(e[k] for e in errors) for k in errors[0]},GT_fields_used_in_export=[])
        assert support['official_GT_filtered']==read('reports/m8j11_sources/official_filters.json')['official_GT_filtered']
    output=dict(scope='FrozenSameStreamPETRWeights_ArtificialHistoryInterventions_DevelopmentOnly',
                T0_exact_cache_metrics=t0,project=project,official=official,official_filters=filters,exports=exports,
                recovery=recovery,official_T3='NotRun_ProjectOnly41MatchedSubset',
                GT_rule_note='Official3393 vs project3559; T3 project subset denominator explicitly computed')
    write('reports/m8j11a_comparison.json',output)
    resources={t:dict(actual_forwards=r['actual_forwards'],process_elapsed_s=r['elapsed_s'],
        summary={k:distribution([f[k] for f in r['frames']]) for k in ('gpu_instrumented_ms','cache_load_h2d_ms',
            'instrumented_frame_wall_ms','peak_allocated_bytes','peak_reserved_bytes')}) for t,r in runs.items()}
    evidence=dict(status='CrossChecksPassed',new_forwards=len(ledger),budget=cfg['budget'],
        all_pairwise_checks=len(checks),max_pairwise_abs=max(v for ch in checks for v in ch['max_abs_error'].values()),
        checks=checks,parameters_buffers_unchanged=True,T3_skipped_frames_not_forwarded=True,
        cross_scene_or_future_or_other_experiment_observations=False,actual_prev_exists_verified=True,
        resources=resources,resource_limit='Instrumented cached-input path includes synchronization/state validation; not comparable to J11 uninstrumented raw-image latency',
        T0_cached_metrics_exact=True,original_assets_unchanged=True,GT_inference=False,optimizer_steps=0,
        states_inventory=dict(detector=['prev_scene_token','test_flag','single_test'],head=['memory_embedding','memory_reference_point',
            'memory_timestamp','memory_egopose','memory_velo'],validity='prev_exists scalar; no independent per-slot visibility flag',
            capacity='1024 read, 256 written per step; 1280 stored after update',
            time='Stored negative source seconds, pre-update adds current real time to produce age. Padding/pseudo references not historical observations',
            automatic_reset='Only scene_token mismatch in frozen simple_test_pts; no time-gap automatic reset found or observed'))
    write(root/'cross_checks.json',evidence)
    public=dict(evidence);public.pop('checks')
    write('reports/m8j11a_validation.json',public)
    print({t:(v['mean_ap'],v['nd_score']) for t,v in official.items()},flush=True)
    print({t:{k:d['intervention']['primary'][k] for k in ['GT','TP','FP','FN','precision','recall','duplicates']} for t,d in project.items()},flush=True)

if __name__=='__main__':main()
