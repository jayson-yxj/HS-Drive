"""M8-J8 cache-only closeout. NumPy only; no model, image decoding or update."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from training.m8j8a_cache_diagnostics import (exact_curve, match_record, operating_points,
    select_cases, summarize_transitions, evidence, query_observation, sigmoid)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)
    with Path(path).open('x') as f:f.write(text+'\n')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def deny_noncache(event,args):
    if event!='open':return
    path=str(args[0])
    # This phase has no business opening weight archives, raw sensors or model inputs.
    if path.endswith(('.pth','.pt','.jpg','.jpeg','.png')) or '/rgb_input/' in path or '/nuscenes-mini/data/' in path:
        raise PermissionError('M8J8A_CacheOnly_NoRawImageWeightOrSensorRead')


def main():
    start=time.monotonic();cfg=read('configs/m8j8a_protocol.json')
    root=Path(cfg['private_cache_root']);private=Path(cfg['private_output'])
    private.mkdir(mode=0o700,exist_ok=False)
    sys.addaudithook(deny_noncache)
    inputs={}
    def checked(path,expected=None):
        value=sha(path)
        if expected is not None:assert value==expected,str(path)
        inputs[str(path)]=value
    for path,expected in cfg['source_sha256'].items():checked(path,expected)
    release=read('reports/m8j8_release_manifest.json')
    for f in release['files']:checked(f['path'],f['sha256']) if not f['path'].endswith('.png') else None
    # Existing aggregate figure is not needed and is deliberately not opened.
    prepared=read(root/'prepared/manifest.json');checked(root/'prepared/manifest.json')
    comparison=read(root/'mini_val_object_comparison.json');checked(root/'mini_val_object_comparison.json')
    baseline=read('reports/m8j8_ab_statistics.json')
    classes=prepared['training_classes'];frames=prepared['frames'];assert len(frames)==81
    records={};matches={};inventory={};curves={};identical_slots={}
    for model in ('A','B'):
        source=read(f'reports/m8j8_sources/m8j8_20261027_ab/{model}/result.json')
        expected={f['index']:f for f in source['frames'] if f['dataset']=='mini_val'}
        records[model]=[];matches[model]=[];shape_counts=Counter();decoded=[];mins=[];outside=0;ties=0
        for frame in frames:
            i=frame['index'];path=root/model/'mini_val'/f'{i:02}.npz';checked(path,expected[i]['cache_sha256'])
            r=load(path);assert r['raw_cls'].shape==r['raw_bbox'].shape==(900,10)
            assert all(np.isfinite(r[k]).all() for k in r)
            assert r['raw_cls'].dtype==r['raw_bbox'].dtype==np.float32
            q=r['query_indices'];flat=r['flat_indices'];ci=r['labels']
            assert np.array_equal(flat,q*10+ci)
            np.testing.assert_allclose(sigmoid(r['raw_cls'])[q,ci],r['scores'],atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(r['raw_bbox'][q,:2],r['boxes'][:,:2],atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(r['raw_bbox'][q,4],r['boxes'][:,2]+r['boxes'][:,5]/2,atol=1e-6,rtol=1e-5)
            logits=r['raw_cls'].reshape(-1);cutoff=np.partition(logits,-300)[-300]
            assert (logits[flat]>=cutoff-1e-6).all()
            ties+=int((logits==cutoff).sum()>1)
            decoded.append(len(q));mins.append(float(r['scores'].min()))
            outside+=int(((np.abs(r['boxes'][:,0])>=51.2)|(np.abs(r['boxes'][:,1])>=51.2)).sum())
            shape_counts.update({k+':'+str(r[k].shape)+':'+str(r[k].dtype):1 for k in r})
            records[model].append(r);matches[model].append(match_record(r,.2))
        curve=exact_curve(records[model]);curves[model]=curve
        for point in baseline['aggregate'][model]['PR']:
            supported=[p for p in curve if p['score_threshold']>=point['score_threshold']]
            current=supported[-1] if supported else dict(TP=0,FP=0,duplicates=0)
            assert all(current[k]==point[k] for k in ('TP','FP','duplicates')),(model,point)
        with Path(f'reports/m8j8a_{model}_cached_PR.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
        inventory[model]=dict(frames=81,keys_shapes_dtypes=dict(shape_counts),
            final_decoder_raw_logits=[900,10],final_decoder_raw_boxes=[900,10],
            decoded_min=min(decoded),decoded_max=max(decoded),decoded_total=sum(decoded),
            decoded_minimum_score=min(mins),decoded_minimum_score_max=max(mins),outside_task_range=outside,
            raw_top300_boundary_tied_frames=ties,formal_score_not_applied_to_cache=True,
            cached_logits_are_independent_sigmoids_not_softmax=True,
            auxiliary_decoder_outputs_available=False,
            unavailable=['Backbone/Context activations','attention maps','intermediate decoder layers','training-time cause interventions'])
    # Stored object matching is checked against unchanged greedy code, not trusted blindly.
    by_frame=defaultdict(list)
    for r in comparison:by_frame[r['frame']].append(r)
    rows=[]
    for frame in frames:
        i=frame['index'];a,b=records['A'][i],records['B'][i]
        assert np.array_equal(a['gt'],b['gt']) and np.array_equal(a['gt_labels'],b['gt_labels'])
        traces_path=root/'prepared/supervision'/f'{i:02}_trace.json';checked(traces_path)
        traces=[t for t in read(traces_path) if t['retained']]
        maps={m:{j:p for p,j in matches[m][i]['pairs']} for m in ('A','B')}
        assert len(by_frame[i])==len(traces)==len(a['gt'])
        for original in by_frame[i]:
            j=original['GT_index'];t=traces[j];pa,pb=maps['A'].get(j),maps['B'].get(j)
            assert original['A_prediction']==pa and original['B_prediction']==pb
            assert original['annotation_token']==t['annotation_token'] and original['instance_token']==t['instance_token']
            state=('common' if pb is not None else 'A_only') if pa is not None else ('B_only' if pb is not None else 'neither')
            xa=None if pa is None else float(np.linalg.norm(a['boxes'][pa,:2]-a['gt'][j,:2]))
            xb=None if pb is None else float(np.linalg.norm(b['boxes'][pb,:2]-b['gt'][j,:2]))
            rows.append(dict(original,scene=frame['scene_name'],sample_token=frame['sample_token'],timestamp_us=frame['timestamp_us'],
                class_name=classes[int(a['gt_labels'][j])],state=state,A_xy_m=xa,B_xy_m=xb,
                xy_delta_m=None if xa is None or xb is None else xb-xa))
    assert len(rows)==3559
    transitions=summarize_transitions(rows);states=transitions['total']['states']
    for k,v in cfg['expected_transitions'].items():assert states[k]['records']==v
    assert states['common']['records']+states['A_only']['records']==2947
    assert states['common']['records']+states['B_only']['records']==2564
    assert states['A_only']['records']-states['B_only']['records']==383
    write(private/'all_GT_transitions.json',rows)
    write('reports/m8j8a_GT_transitions.json',transitions)
    reference={k:baseline['aggregate']['A']['primary'][k] for k in ('GT','TP','FP','FN','precision','recall','score_threshold')}
    pr=dict(scope='PostHocCacheMechanismDiagnostic_NotNewTestSetScore',original_grid={m:baseline['aggregate'][m]['PR'] for m in ('A','B')},
        fixed_formal_threshold=.2,decoded_top300_only=True,all_original_grid_counts_reproduced=True,
        B_relative_to_original_A=operating_points(curves['B'],reference),
        maximum_A=curves['A'][-1],maximum_B=curves['B'][-1],
        newly_recovered_deleted_predictions=0,original_metrics_preserved=True)
    write('reports/m8j8a_PR_diagnostic.json',pr)
    write('reports/m8j8a_cache_inventory.json',dict(models=inventory,
        stages=['All900FinalDecoderLogitsAndEncodedBoxes','Top300QueryClassAfterDecoderRange','FormalScore0.2AppliedOnlyAtMetricTime'],
        decoder_post_center_range=[-61.2,-61.2,-10,61.2,61.2,10],task_xy_range='strict abs(x),abs(y)<51.2',
        matching='Unchanged same-class descending-score one-to-one XY<2m',
        no_NMS=True,query_slot_identity='WithinForwardIndex_NotStableObjectIdentityAcrossWeightsOrFrames'))
    selected=select_cases(rows,cfg['cases']);write(private/'case_manifest.json',selected)
    details=[];public=[]
    for row in selected:
        i,j=row['frame'],row['GT_index'];item=dict(row,models={})
        for m in ('A','B'):item['models'][m]=evidence(records[m][i],j,matches[m][i])
        # Same slot numbers across weights can show numerical change, never object tracking.
        item['fixed_query_slot_comparisons']=[]
        qs={x['matched']['query'] for x in item['models'].values() if x['matched'] is not None}
        for q in sorted(qs):
            item['fixed_query_slot_comparisons'].append(dict(query=q,A=query_observation(records['A'][i],j,q),
                B=query_observation(records['B'][i],j,q),warning='SameParameterSlot_NotCertifiedSameObject'))
        images=[]
        for m in ('A','B'):
            for suffix in ('six','bev'):
                path=root/m/'mini_val'/f'{i:02}_{suffix}.jpg'
                if path.exists():images.append(str(path))
        item['existing_visualizations']=images;item['human_confirmed']=False
        details.append(item)
        public_models={}
        for m,d in item['models'].items():
            public_models[m]=dict(matched=d['matched'] is not None,flags=d['flags'],
                match_metrics=None if d['matched'] is None else {k:v for k,v in d['matched'].items() if k not in ('prediction_index','query')},
                nearest_raw_distance_m=d['nearest_raw']['xy_distance_m'],
                nearest_raw_correct_class_score=d['nearest_raw']['correct_class_score'],
                best_nearby_correct_class_score=None if d['best_correct_class_nearby'] is None else d['best_correct_class_nearby']['correct_class_score'],
                best_nearby_argmax_class=None if d['best_correct_class_nearby'] is None else classes[d['best_correct_class_nearby']['argmax_class']],
                strong_nearby_top300_exclusion_count=d['strong_nearby_top300_exclusion_count'],
                nearby_decoded_count=len(d['near_decoded']),nearby_above_threshold_count=sum(x['score']>=.2 for x in d['near_decoded']))
        public.append(dict(case_id=row['case_id'],bucket=row['bucket'],frame=i,scene=row['scene'],class_name=row['class_name'],
            models=public_models,existing_visualization_available=bool(images),human_confirmed=False,
            private_evidence=str(private/'case_details.json')))
    write(private/'case_details.json',details)
    write('reports/m8j8a_case_summary.json',dict(selection_protocol=cfg['cases'],count=len(selected),
        unique_instances=len({r['instance_token'] for r in selected}),population_attribution_allowed=False,cases=public))
    page='<meta charset="utf-8"><h1>M8-J8A cached object comparison</h1><p>Post-hoc selected cases, not a random sample. No human sign-off. Links reuse existing M8-J8 images; green GT, orange normal predictions. No new rendering.</p>'
    page+='<p><a href="case_manifest.json">Private selection / identities</a> | <a href="case_details.json">Full cache evidence</a> | <a href="all_GT_transitions.json">All GT transitions</a></p>'
    for d in details:
        page+=f'<h2>{d["case_id"]}: {d["bucket"]}, {d["scene"]}, frame {d["frame"]}, {d["class_name"]}</h2>'
        page+='<pre>'+html.escape(json.dumps(d,indent=2))+'</pre>'
        if not d['existing_visualizations']:page+='<p>No previously generated visualization for this frame; numeric-only. Not visually confirmed.</p>'
        for path in d['existing_visualizations']:
            page+=f'<p>{html.escape(path)}</p><img width="1400" src="{os.path.relpath(path,private)}">'
    with (private/'case_comparison.html').open('x') as f:f.write(page)
    # Rehash every source/cache opened; all are read-only. Preserve old figure without opening it.
    for path,expected in inputs.items():assert sha(path)==expected,path
    assert 'torch' not in sys.modules and 'cv2' not in sys.modules
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write('reports/m8j8a_execution.json',dict(code_commit=code_commit,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=time.monotonic()-start,python=sys.version,numpy=np.__version__,model_forward_calls=0,optimizer_steps=0,
        backward_calls=0,torch_imported=False,GPU_used=False,image_payloads_opened=0,
        historical_inputs_and_caches_unchanged=True,inputs_sha256=inputs,
        private_outputs_sha256={p.name:sha(p) for p in private.iterdir() if p.is_file()},
        new_output_scope='AggregatesAndUpTo12PrivateCases_NoNewEvaluationDataset',status='CacheOnlyCompleted'))
    print(json.dumps(dict(status='CacheOnlyCompleted',transition_counts={k:v['records'] for k,v in states.items()},
        B_workpoints=pr['B_relative_to_original_A'],cases=public),indent=2))


if __name__=='__main__':main()
