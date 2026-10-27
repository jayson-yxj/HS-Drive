"""Frozen A export and unchanged official nuScenes eval, in separate CPU processes."""
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime,timezone
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import hashlib

import numpy as np
from datasets.m8j9_official_export import (read_metadata_only,check_manifest,pose_info,
    prediction_to_export_input,exporter,compare_export,ALLOWED_TABLES)
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_prediction,load_gt,add_center_dist,filter_eval_boxes
from nuscenes.eval.detection.utils import category_to_detection_name


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)


def counts(boxes,names):
    raw=Counter(b.detection_name for b in boxes.all)
    return dict(total=len(boxes.all),classes={n:raw[n] for n in names},samples=len(boxes.sample_tokens))


def export_stage(cfg,rep):
    private=Path(cfg['private_output']);manifest=read(cfg['manifest'])
    sources={}
    for p,h in cfg['sources_sha256'].items():assert sha(p)==h,p
    root=Path(cfg['dataset_root'])/cfg['version']
    def guard(event,args):
        if event!='open':return
        p=str(args[0]);base=Path(p).name
        if p.endswith(('.pth','.pt','.jpg','.jpeg','.png','.bin')) or '/B/' in p or '/supervision/' in p:
            raise PermissionError('M8J9_Export_NoGT_Weights_Images_OrB')
        if str(root) in p and base not in {n+'.json' for n in ALLOWED_TABLES}:
            raise PermissionError('M8J9_Export_DisallowedMetadataTable')
    sys.addaudithook(guard)
    canary_rejected=0
    for path in (root/'sample_annotation.json',private/'forbidden.pth'):
        try:path.read_bytes()
        except PermissionError:canary_rejected+=1
    assert canary_rejected==2
    tables=read_metadata_only(cfg['dataset_root'],cfg['version'])
    for name in ALLOWED_TABLES:sources[str(root/(name+'.json'))]=sha(root/(name+'.json'))
    tokens=check_manifest(manifest,tables,create_splits_scenes()['mini_val']);assert len(tokens)==81
    A_run=read('reports/m8j8_sources/m8j8_20261027_ab/A/result.json')
    expected={f['index']:f for f in A_run['frames'] if f['dataset']=='mini_val'}
    infos=[];detections=[];rows=[];indices=[]
    for f in manifest['frames']:
        i=f['index'];p=Path(cfg['cache_root'])/f'{i:02}.npz';h=sha(p);assert h==expected[i]['cache_sha256']
        sources[str(p)]=h;assert expected[i]['sample_token']==f['sample_token']
        with np.load(p,allow_pickle=False) as z:
            # Explicit key whitelist: GT entries in the same archive are not read.
            keys=('boxes','scores','labels','query_indices','raw_bbox')
            pred={k:z[k].copy() for k in keys}
        assert pred['boxes'].shape==(300,9) and pred['raw_bbox'].shape==(900,10)
        np.testing.assert_array_equal(pred['boxes'][:,7:9],pred['raw_bbox'][pred['query_indices'],8:10])
        det,keep=prediction_to_export_input(pred,cfg['classes']);info=pose_info(f['sample_token'],tables)
        infos.append(info);detections.append(det);indices.append(keep.tolist())
        sample=tables['sample'][f['sample_token']]
        rows.append(dict(index=i,sample_token=f['sample_token'],scene=f['scene_name'],cached=300,
            after_existing_score_and_XY=len(keep),reference_timestamp_us=info['reference_timestamp_us'],
            sample_timestamp_us=sample['timestamp'],source_coordinate='LIDAR_TOP at its own sample_data ego pose',
            cache_sha256=h,prediction_keys_read=list(keys)))
    ds=exporter(infos,cfg['classes'],cfg['output_policy']['meta'])
    result_path=ds._format_bbox(detections,str(private/'export'))
    payload=read(result_path);assert set(payload['results'])==set(tokens)
    maxima={};attributes=Counter();retained_indices=[]
    for info,det,row,idx in zip(infos,detections,rows,indices):
        exported=payload['results'][info['token']]
        err,kept=compare_export(det,info,exported,cfg['classes'],ds.eval_detection_configs)
        for key,val in err.items():maxima[key]=max(maxima.get(key,0.),val)
        row['after_original_export_class_range']=len(exported)
        row['export_class_range_removed']=len(idx)-len(exported)
        assert len(exported)<=ds.eval_detection_configs.max_boxes_per_sample
        assert all(x['detection_score']>=.2 for x in exported)
        attributes.update(x['attribute_name'] for x in exported)
        retained_indices.append(dict(sample_token=info['token'],cache_prediction_indices=[idx[j] for j in kept]))
    boxes,meta=load_prediction(result_path,500,DetectionBox,verbose=True)
    assert set(boxes.sample_tokens)==set(tokens) and meta==cfg['output_policy']['meta']
    for token in tokens:
        for original,box in zip(payload['results'][token],boxes[token]):
            back=DetectionBox.deserialize(json.loads(json.dumps(box.serialize())))
            assert back==box
            for key in ('translation','size','rotation','velocity'):
                np.testing.assert_array_equal(original[key],box.serialize()[key])
    write(private/'export_provenance.json',dict(poses=infos,prediction_indices=retained_indices,
        GT_fields_accessed=[],cache_and_metadata_sha256=sources))
    for p,h in sources.items():assert sha(p)==h
    return dict(status='ExportInterfacePassed',sample_count=81,scene_names=sorted({f['scene_name'] for f in manifest['frames']}),
        A_identity=cfg['A'],result_path=result_path,result_sha256=sha(result_path),
        cached_total=81*300,after_existing_score_and_XY=sum(r['after_existing_score_and_XY'] for r in rows),
        exported_total=len(boxes.all),frames=rows,source_hashes=sources,transform_max_errors=maxima,
        serialization_roundtrip=True,missing_samples=0,empty_result_samples=sum(not payload['results'][t] for t in tokens),
        additional_cap_truncation=0,exporter_class_range_removed=sum(r['export_class_range_removed'] for r in rows),
        attribute_counts=dict(attributes),attribute_provenance='OriginalDeterministicExporterRule_NotGT_NotLearnedHead',
        velocity_provenance='CachedPrediction7:9_EqualsRawDecoder8:10',GT_input_access=False,canary_rejections=canary_rejected,
        official_config=ds.eval_detection_configs.serialize(),raw_images_or_pointcloud_reads=0)


def audit_official_filters(nusc,cfg,engine):
    """Read-only provenance around official functions; no evaluator changes."""
    config=engine.cfg;names=list(config.class_names)
    raw_gt=load_gt(nusc,cfg['eval_set'],DetectionBox,verbose=False)
    raw_counts=counts(raw_gt,names)
    object_to_annotation={};total_raw=0;unsupported=0
    for token in raw_gt.sample_tokens:
        annotations=[nusc.get('sample_annotation',a) for a in nusc.get('sample',token)['anns']]
        total_raw+=len(annotations)
        supported=[a for a in annotations if category_to_detection_name(a['category_name']) is not None]
        unsupported+=len(annotations)-len(supported)
        assert len(supported)==len(raw_gt[token])
        for box,ann in zip(raw_gt[token],supported):
            assert tuple(box.translation)==tuple(ann['translation'])
            object_to_annotation[id(box)]=ann['token']
    add_center_dist(nusc,raw_gt)
    per_class_stage={name:dict(loaded=0,after_range=0,after_points=0,after_official_filter=0) for name in names}
    for box in raw_gt.all:
        row=per_class_stage[box.detection_name];row['loaded']+=1
        if box.ego_dist<config.class_range[box.detection_name]:
            row['after_range']+=1
            if box.num_pts!=0:row['after_points']+=1
    captured=io.StringIO()
    with redirect_stdout(captured):final_gt=filter_eval_boxes(nusc,raw_gt,config.class_range,verbose=True)
    assert json.dumps(final_gt.serialize(),sort_keys=True)==json.dumps(engine.gt_boxes.serialize(),sort_keys=True)
    kept={object_to_annotation[id(box)] for box in final_gt.all}
    for box in final_gt.all:per_class_stage[box.detection_name]['after_official_filter']+=1
    retained_instances={name:set() for name in names}
    for box in final_gt.all:
        retained_instances[box.detection_name].add(nusc.get('sample_annotation',object_to_annotation[id(box)])['instance_token'])
    prepared=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared')
    manifest=read(cfg['manifest']);project=set();private_reasons=[]
    for f in manifest['frames']:
        for trace in read(prepared/'supervision'/f'{f["index"]:02}_trace.json'):
            if trace['retained']:project.add(trace['annotation_token'])
    # Membership diagnostic only, never passed to exporter or evaluator.
    for token in sorted(kept|project):
        ann=nusc.get('sample_annotation',token)
        private_reasons.append(dict(annotation_token=token,instance_token=ann['instance_token'],
            sample_token=ann['sample_token'],official_retained=token in kept,project_retained=token in project))
    raw_pred,_=load_prediction(engine.result_path,500,DetectionBox,verbose=False)
    pred_counts=counts(raw_pred,names);add_center_dist(nusc,raw_pred)
    pred_log=io.StringIO()
    with redirect_stdout(pred_log):final_pred=filter_eval_boxes(nusc,raw_pred,config.class_range,verbose=True)
    assert json.dumps(final_pred.serialize(),sort_keys=True)==json.dumps(engine.pred_boxes.serialize(),sort_keys=True)
    return dict(raw_all_annotations=total_raw,unsupported_taxonomy=unsupported,official_GT_loaded=raw_counts,
        official_GT_filtered=counts(final_gt,names),GT_stages_by_class=per_class_stage,
        GT_instances_by_class={n:len(v) for n,v in retained_instances.items()},
        official_GT_filter_log=captured.getvalue(),official_prediction_filter_log=pred_log.getvalue(),
        predictions_loaded=pred_counts,predictions_filtered=counts(final_pred,names),
        project_GT_count=len(project),official_project_common=len(project&kept),
        project_only=len(project-kept),official_only=len(kept-project),
        source='UnmodifiedOfficialLoadGT_AddCenterDist_FilterEvalBoxes; diagnostic compares object identity memberships',
        evaluator_outputs_equal_audit=True),private_reasons


def eval_stage(cfg,rep,phase):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    def guard(event,args):
        if event!='open':return
        p=str(args[0])
        if p.endswith(('.pth','.pt','.jpg','.jpeg','.png','.bin')) or '/B/' in p or '/rgb_input/' in p:
            raise PermissionError('M8J9_Evaluation_MetadataOnly_NoSensorPayloadsWeightsOrB')
    sys.addaudithook(guard)
    private=Path(cfg['private_output']);export=read('reports/m8j9_sources/export/result.json')
    result_path=export['result_path'];assert sha(result_path)==export['result_sha256']
    for p,h in cfg['sources_sha256'].items():assert sha(p)==h,p
    # Actual NuScenes loads original GT metadata; does not read map/sensor raster payloads.
    nusc=NuScenes(version=cfg['version'],dataroot=cfg['dataset_root'],verbose=True)
    config=config_factory(cfg['official_detection_config'])
    out=private/phase;assert not out.exists()
    engine=DetectionEval(nusc,config,result_path,cfg['eval_set'],str(out),verbose=True)
    metrics=engine.main(plot_examples=0,render_curves=False)
    # Preserve verbatim official summaries/details; these contain aggregate curves only.
    for name in ('metrics_summary.json','metrics_details.json'):
        with (rep/name).open('xb') as f:f.write((out/name).read_bytes())
    report=dict(status='OfficialEvaluationCompleted',phase=phase,result_sha256=sha(result_path),
        evaluation_name='冻结A在mini_val上的本地官方devkit评测，使用指定缓存与输出策略',
        config=config.serialize(),GT_filtered=counts(engine.gt_boxes,list(config.class_names)),
        predictions_filtered=counts(engine.pred_boxes,list(config.class_names)),
        summary_sha256=sha(out/'metrics_summary.json'),details_sha256=sha(out/'metrics_details.json'),
        prediction_result_unchanged=sha(result_path)==export['result_sha256'],
        output_dir=str(out),model_forward=0,optimizer_step=0)
    if phase=='eval1':
        audit,private_reasons=audit_official_filters(nusc,cfg,engine)
        write('reports/m8j9_GT_filter_audit.json',audit)
        write(private/'GT_membership_comparison.json',private_reasons)
    for p,h in cfg['sources_sha256'].items():assert sha(p)==h,p
    return report


def main():
    cfg=read('configs/m8j9_protocol.json');phase=os.environ['M8J9_PHASE'];rep=Path(os.environ['M8J9_REPORT'])
    start=time.monotonic();os.umask(0o077)
    result=dict(phase=phase,started_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=os.environ['M8J9_COMMIT'],
        software={p:version(p) for p in ('nuscenes-devkit','numpy','torch','mmcv-full','mmdet','mmdet3d')},
        model_forward_calls=0,optimizer_steps=0,backward_calls=0,GPU_used=False)
    try:
        assert version('nuscenes-devkit')==cfg['nuscenes_devkit']
        result.update(export_stage(cfg,rep) if phase=='export' else eval_stage(cfg,rep,phase))
    except Exception as exc:
        result.update(status='FailedStopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(elapsed_s=time.monotonic()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(rep/'result.json',result)


if __name__=='__main__':main()
