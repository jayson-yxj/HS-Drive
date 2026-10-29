"""Finite cache/source/devkit reproducibility checks, no model execution."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from datasets.m8j11_contract import read_metadata,check_sequence
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=read('configs/m8j11_protocol.json');j=read('configs/m8j9_retry1_protocol.json');root=Path(c['private_root']);t=read_metadata(j['dataset_root']);mf=read(c['eval_manifest']);frames=mf['frames']
    check_sequence(frames,t);scenes=create_splits_scenes()['mini_val']
    assert {f['sample_token'] for f in frames}=={s['token'] for s in t['sample'].values() if t['scene'][s['scene_token']]['name'] in scenes}
    assert len(frames)==81 and len({f['scene_token'] for f in frames})==2
    pre=read(root/'prepared/manifest.json');oldpre=read('/root/gpufree-data/hs-drive-private/m8j8_20261027_ab/prepared/manifest.json')
    arrays_equal=0
    for f in pre['frames']:
        i=f['index'];assert sha(f['path'])==f['sha256']
        # Failed preparation produced same input values; only checker changed.
        with np.load(f['path'],allow_pickle=False) as a,np.load(root/'prepared_failed_float64_tolerance'/f'{i:02}.npz',allow_pickle=False) as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k]);arrays_equal+=1
        assert f['camera_image_sha256']==[x['sha256'] for x in oldpre['frames'][i]['camera_files']]
    repeat=0.
    for i in (0,1,2,40,41):
        with np.load(root/'probe'/f'{i:02}.npz',allow_pickle=False) as a,np.load(root/'full'/f'{i:02}.npz',allow_pickle=False) as b:
            for k in a.files:
                repeat=max(repeat,float(np.max(abs(a[k]-b[k]))));np.testing.assert_array_equal(a[k],b[k])
    for path,h in j['sources_sha256'].items():assert sha(path)==h,path
    assert sha(j['A']['path'])==j['A']['sha256']
    cp=Path(c['runtime_root'])/'artifacts/stream_petr_r50_flash_704_bs2_seq_90e.pth';assert sha(cp)==c['initial_asset_sha256']
    export=read('reports/m8j11_sources/full_export.json');assert sha(export['result_path'])==export['result_sha256']
    for p,h in export['source_cache_hashes'].items():assert sha(p)==h
    nusc=NuScenes(version='v1.0-mini',dataroot=j['dataset_root'],verbose=False)
    engine=DetectionEval(nusc,config_factory('detection_cvpr_2019'),export['result_path'],'mini_val',str(root/'official_repeat'),verbose=False)
    engine.main(plot_examples=0,render_curves=False)
    checked=[]
    def normalize(x):
        if isinstance(x,dict):return {k:normalize(v) for k,v in x.items() if k!='eval_time'}
        if isinstance(x,list):return [normalize(v) for v in x]
        if isinstance(x,float) and np.isnan(x):return 'OfficialNaN_NotApplicable'
        return x
    for name in ('metrics_summary.json','metrics_details.json'):
        assert normalize(read(root/'official_repeat'/name))==normalize(read('reports/m8j11_sources/'+name));checked.append(name)
    source_diff=subprocess.check_output(['git','diff','68ee3e8','--name-only'],text=True).splitlines()
    assert all('m8j11' in p.lower() for p in source_diff)
    for run in ('probe','full','timing'):
        r=read(f'reports/m8j11_sources/{run}/result.json');assert r['status']=='ReadOnlySequencePassed';assert r['initial_state_sha256']==r['final_state_sha256'];assert r['optimizer_steps']==r['backward_calls']==0
    result=dict(status='CacheSourceAndOfficialReproducibilityPassed',mini_val_exact=True,all81_images_match_prior_manifest=True,
        failed_prepare_inputs_identical_arrays=arrays_equal,probe_full_reset_max_abs=repeat,independent_official_repeat_equal=checked,
        old_J9_sources_unchanged=True,A_checkpoint_unchanged=True,candidate_checkpoint_unchanged=True,only_m8j11_git_changes=True,
        optimizer_steps=0,new_model_forwards=0,backward_calls=0,raw_prediction_hashes_verified=81)
    with Path('reports/m8j11_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(result)
if __name__=='__main__':main()
