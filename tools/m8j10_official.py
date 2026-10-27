"""B/C cached output through unchanged J9 exporter and official devkit."""
from datetime import datetime,timezone
import json,os,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
from datasets.m8j6_selection import sha256
from datasets.m8j9_official_export import (read_metadata_only,check_manifest,pose_info,prediction_to_export_input,exporter,compare_export,ALLOWED_TABLES)
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from tools.m8j6_prepare_actual import save


def main():
    cfg=json.loads(Path('configs/m8j10_protocol.json').read_text());rep=Path(os.environ['M8J10_REPORT']);private=Path(cfg['private_root'])
    assert version('nuscenes-devkit')==cfg['devkit']
    for p,h in cfg['official_sources_sha256'].items():assert sha256(p)==h,p
    fixed=json.loads(Path(cfg['evaluation_manifest']).read_text());assert sha256(cfg['evaluation_manifest'])==cfg['evaluation_manifest_sha256']
    tables=read_metadata_only(cfg['dataset_root'],'v1.0-mini');tokens=check_manifest(fixed,tables,create_splits_scenes()['mini_val']);assert len(tokens)==81
    j8=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab')
    records={}
    b=json.loads(Path('reports/m8j8_sources/m8j8_20261027_ab/B/result.json').read_text())
    records['B']={f['index']:f for f in b['frames'] if f['dataset']=='mini_val'}
    for step in (200,400):
        e=json.loads((rep.parent/'train'/f'eval_{step:03}.json').read_text())
        records['C'+str(step)]={f['index']:f for f in e['frames']}
    def guard(event,args):
        if event!='open':return
        p=str(args[0])
        if p.endswith(('.pth','.pt','.jpg','.png','.jpeg','.bin')):raise PermissionError('CachedExportAndEval_NoWeightsOrSensorPayloads')
        if os.environ.get('M8J10_EXPORT_ACTIVE')=='1' and str(Path(cfg['dataset_root'])/'v1.0-mini') in p:
            if Path(p).stem not in ALLOWED_TABLES:raise PermissionError('NoGTInPredictionExport')
    sys.addaudithook(guard)
    results={}
    for name in ('B','C200','C400'):
        start=time.perf_counter();os.environ['M8J10_EXPORT_ACTIVE']='1'
        output=private/'official'/name;output.mkdir(parents=True,exist_ok=False)
        infos=[];detections=[];rows=[];hashes={}
        for frame in fixed['frames']:
            i=frame['index'];path=j8/'B/mini_val'/f'{i:02}.npz' if name=='B' else private/name/'mini_val'/f'{i:02}.npz'
            expected=records[name][i];assert expected['sample_token']==frame['sample_token']
            assert sha256(path)==expected['cache_sha256'];hashes[str(path)]=sha256(path)
            with np.load(path,allow_pickle=False) as z:
                pred={k:z[k].copy() for k in ('boxes','scores','labels','raw_bbox','query_indices')}
            assert pred['boxes'].shape==(300,9) and pred['raw_bbox'].shape==(900,10)
            np.testing.assert_array_equal(pred['boxes'][:,7:9],pred['raw_bbox'][pred['query_indices'],8:10])
            det,indices=prediction_to_export_input(pred,cfg['official_classes'])
            detections.append(det);infos.append(pose_info(frame['sample_token'],tables))
            rows.append(dict(index=i,sample_token=frame['sample_token'],cached=300,after_score_XY=len(indices)))
        ds=exporter(infos,cfg['official_classes'],cfg['official_output']['meta'])
        path=ds._format_bbox(detections,str(output/'export'))
        payload=json.loads(Path(path).read_text());assert set(payload['results'])==set(tokens)
        transform_errors={}
        for info,det,row in zip(infos,detections,rows):
            boxes=payload['results'][info['token']];assert len(boxes)<=500
            errors,_=compare_export(det,info,boxes,cfg['official_classes'],ds.eval_detection_configs)
            for k,v in errors.items():transform_errors[k]=max(transform_errors.get(k,0.),v)
            row['exported']=len(boxes)
        loaded,meta=load_prediction(path,500,DetectionBox,verbose=False)
        assert set(loaded.sample_tokens)==set(tokens) and meta==cfg['official_output']['meta']
        result_sha=sha256(path)
        os.environ['M8J10_EXPORT_ACTIVE']='0'
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        nusc=NuScenes(version='v1.0-mini',dataroot=cfg['dataset_root'],verbose=False)
        engine=DetectionEval(nusc,ds.eval_detection_configs,path,'mini_val',str(output/'metrics'),verbose=True)
        assert len(engine.gt_boxes.all)==3393
        engine.main(plot_examples=0,render_curves=False)
        target=rep/name;target.mkdir(exist_ok=False)
        for f in ('metrics_summary.json','metrics_details.json'):
            with (target/f).open('xb') as handle:handle.write((output/'metrics'/f).read_bytes())
        assert sha256(path)==result_sha
        result=dict(status='OfficialEvaluationCompleted',name=name,frames=rows,result_path=path,result_sha256=result_sha,
            exported_predictions=len(loaded.all),evaluated_predictions=len(engine.pred_boxes.all),GT_official=len(engine.gt_boxes.all),
            source_cache_sha256=hashes,transform_max_errors=transform_errors,model_forwards=0,optimizer_steps=0,
            metadata_only_GT_in_evaluator=True,export_GT_fields_read=[],elapsed_s=time.perf_counter()-start,
            metrics_summary_sha256=sha256(target/'metrics_summary.json'),metrics_details_sha256=sha256(target/'metrics_details.json'))
        save(target/'result.json',result);results[name]=result
    for p,h in cfg['official_sources_sha256'].items():assert sha256(p)==h,p
    save(rep/'result.json',dict(status='SameProtocolB_C200_C400_Completed',evaluations=list(results),
        A_reused='reports/m8j9_sources/resume/eval2/metrics_summary.json',model_forwards=0,optimizer_steps=0,
        devkit=version('nuscenes-devkit'),code_commit=os.environ['M8J10_COMMIT'],timestamp_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
