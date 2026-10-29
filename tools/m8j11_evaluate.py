"""Reuse frozen J9 exporter/devkit and J8 diagnostics; predictions only in export."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from datasets.m8j9_official_export import read_metadata_only,pose_info,prediction_to_export_input,exporter,compare_export
from tools.m8j8_report import metrics,paired,vertical_summary,load
from training.m8j7_metrics import distribution

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    with Path(p).open('x') as f:json.dump(x,f,indent=2,default=convert)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['probe','full'],required=True);a=ap.parse_args();os.umask(0o077)
    c=read('configs/m8j11_protocol.json');old=read('configs/m8j9_retry1_protocol.json');out=Path(c['private_root'])/a.phase
    run=read(f'reports/m8j11_sources/{a.phase}/result.json');assert run['status']=='ReadOnlySequencePassed'
    mf=read(c['eval_manifest']);ids=[0,1,2,40,41] if a.phase=='probe' else list(range(81))
    t=read_metadata_only(old['dataset_root'],'v1.0-mini');infos=[];dets=[];records=[];sources={};maxima={}
    for i in ids:
        p=out/f'{i:02}.npz';sources[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:r={k:z[k].copy() for k in ('boxes','native_boxes','scores','labels')}
        det,keep=prediction_to_export_input(r,old['classes']);info=pose_info(mf['frames'][i]['sample_token'],t)
        # Native rc6 LWH/CCW versus bridge WLH/SECOND: identical physical box.
        np.testing.assert_array_equal(r['boxes'][:,3:5],r['native_boxes'][:,[4,3]])
        for native,legacy in zip(r['native_boxes'],r['boxes']):
            np.testing.assert_allclose(Quaternion(axis=[0,0,1],angle=float(native[6])).rotation_matrix,
                Quaternion(axis=[0,0,1],angle=-float(legacy[6])-np.pi/2).rotation_matrix,atol=1e-6)
        infos.append(info);dets.append(det);records.append(r)
    ds=exporter(infos,old['classes'],old['output_policy']['meta']);path=ds._format_bbox(dets,str(out/'official_export'))
    payload=read(path);assert set(payload['results'])=={x['token'] for x in infos}
    for det,info in zip(dets,infos):
        err,_=compare_export(det,info,payload['results'][info['token']],old['classes'],ds.eval_detection_configs)
        for k,v in err.items():maxima[k]=max(maxima.get(k,0),v)
    export_record=dict(status='ShortSequenceExportPassed' if a.phase=='probe' else 'FullExportPassed',samples=len(ids),
        result_path=path,result_sha256=sha(path),transform_max_errors=maxima,source_cache_hashes=sources,GT_fields_used_for_export=[],
        candidate_native_box='MMDet3Drc6 bottomXYZ fullLWH CCW yaw',bridge='swap L/W and second_yaw=-native_yaw-pi/2; velocity unchanged',
        exporter='Unmodified frozen J9 MMDet3D0.17.1',score_threshold=.2,TopK=300,NMS=False,
        metadata_flag_note='J9 frozen use_external=False retained; ImageNet pretraining disclosed in comparison report')
    write(f'reports/m8j11_sources/{a.phase}_export.json',export_record)
    if a.phase=='probe':return
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.config import config_factory
    from tools.m8j9_stage import audit_official_filters
    nusc=NuScenes(version='v1.0-mini',dataroot=old['dataset_root'],verbose=False)
    engine=NuScenesEval(nusc,config=config_factory('detection_cvpr_2019'),result_path=path,eval_set='mini_val',output_dir=str(out/'official_metrics'),verbose=True)
    engine.main(plot_examples=0,render_curves=True)
    filters,private_reasons=audit_official_filters(nusc,old,engine);write('reports/m8j11_sources/official_filters.json',filters);write(out/'GT_membership.json',private_reasons)
    for name in ('metrics_summary.json','metrics_details.json'):
        Path('reports/m8j11_sources',name).write_bytes((out/'official_metrics'/name).read_bytes())
    A=[];S=[];pre=Path('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared')
    for i,r in zip(ids,records):
        with np.load(pre/'supervision'/f'{i:02}.npz',allow_pickle=False) as z:r.update(gt=z['boxes'].copy(),gt_labels=z['labels'].copy())
        S.append(r);aa=load(Path(old['cache_root'])/f'{i:02}.npz');A.append(aa)
    pp=read('configs/m8j7_protocol.json');classes=old['classes']
    from tools.m8j10_report import anomalies
    quality=dict(A=metrics(A,classes,pp),StreamPETR=metrics(S,classes,pp),paired=paired(A,S,classes),scenes={},vertical_anomalies=dict(A=anomalies(A,classes),StreamPETR=anomalies(S,classes)))
    for scene in sorted({f['scene_name'] for f in mf['frames']}):
        idx=[f['index'] for f in mf['frames'] if f['scene_name']==scene]
        quality['scenes'][scene]=dict(A=metrics([A[i] for i in idx],classes,pp),StreamPETR=metrics([S[i] for i in idx],classes,pp))
    write('reports/m8j11_project_comparison.json',quality)
    timing={}
    for label,rows in [('all',run['frames']),('cold_scene',[r for r in run['frames'] if r['cold_scene']]),('subsequent',[r for r in run['frames'] if not r['cold_scene']])]:
        timing[label]={k:distribution([r[k] for r in rows]) for k in ('gpu_ms','cached_input_to_cpu_prediction_ms','cache_load_transfer_ms','peak_allocated_bytes','peak_reserved_bytes')}
    write('reports/m8j11_resource_comparison.json',dict(StreamPETR=timing,timing_boundary='Prepared numeric input disk read + H2D + inference + CPU decoded boxes; excludes raw JPEG transforms, diagnostics and output serialization',
        A='Same-boundary timing not yet measured; historical timings not directly comparable'))
    print('Official and project evaluation complete')
if __name__=='__main__':main()
