"""Freeze T0 identity and input-only schedules before any new forward."""
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from tools.m8j11a_common import read, write, sha, schedules, PROTOCOL


def main():
    os.umask(0o077)
    c=read('configs/m8j11_protocol.json'); rt=Path(c['runtime_root']); old=Path(c['private_root'])
    prepared=read(old/'prepared/manifest.json'); mf=read(c['eval_manifest'])
    run=read('reports/m8j11_sources/full/result.json'); export=read('reports/m8j11_sources/full_export.json')
    assert run['status']=='ReadOnlySequencePassed' and run['checkpoint_sha256']==c['initial_asset_sha256']
    assert sha(c['eval_manifest'])==c['evaluation']['manifest_sha256']==prepared['source_manifest_sha256']
    assert len(prepared['frames'])==len(mf['frames'])==len(run['frames'])==81
    for f,p in zip(mf['frames'],prepared['frames']):
        for k in f: assert f[k]==p[k]
        assert sha(p['path'])==p['sha256']
    cache=[]
    for f in mf['frames']:
        path=old/'full'/('{:02}.npz'.format(f['index']))
        assert sha(path)==export['source_cache_hashes'][str(path)]
        cache.append(dict(index=f['index'],path=str(path),sha256=sha(path)))
    cp=rt/'artifacts/stream_petr_r50_flash_704_bs2_seq_90e.pth'
    assert sha(cp)==c['initial_asset_sha256']
    schedule,scenes=schedules(mf['frames'])
    assert [len(schedule[t]) for t in ('T1','T2','T3')]==[81,81,41]
    fixed=['configs/m8j11_protocol.json',c['eval_manifest'],'configs/m8j9_retry1_protocol.json',
           'configs/m8j7_protocol.json','tools/m8j11_gpu.py','datasets/m8j11_contract.py',
           'datasets/m8j9_official_export.py','training/m8j7_metrics.py','tools/m8j8_report.py',
           'tools/m8j10_report.py','reports/m8j11_project_comparison.json',
           'reports/m8j11_sources/metrics_summary.json','reports/m8j11_sources/metrics_details.json',
           'reports/m8j11_sources/full/result.json','reports/m8j11_sources/full_export.json',
           str(old/'prepared/manifest.json'),str(old/'full/resolved_config.py'),str(cp)]
    source=rt/'runtime-source'
    fixed += [str(p) for p in sorted((source/'projects').rglob('*.py'))]
    fixed += [str(rt/'env/lib/python3.8/site-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so')]
    assert subprocess.check_output(['git','-C',str(rt/'official-source'),'rev-parse','HEAD'],text=True).strip()==c['source_commit']
    config=dict(stage='M8-J11A',base_commit='1d7472dea3941132a0e5946dc7f80feddc26b4f6',
        branch='feat/streampetr-temporal-sensitivity',client_date='2026-11-01',
        host_recorded_utc=datetime.now(timezone.utc).isoformat(),
        date_note='Client task date and execution host clock retained separately; host date not substituted for client date',
        private_root='/root/gpufree-data/hs-drive-private/m8j11a-temporal-20261101',
        source_protocol='configs/m8j11_protocol.json',runtime_root=str(rt),
        prepared_manifest=str(old/'prepared/manifest.json'),T0_cache=cache,T0='StreamPETR_only_M8J11_full_normal_sequence',
        fixed_assets_sha256={p:sha(p) for p in fixed},scenes=scenes,schedule=schedule,
        budget=dict(primary=203,extra_max=40,total_max=243,warmup=0,per_condition=dict(T1=81,T2=81,T3=41)),
        run_ids={t:'m8j11a_20261101_'+t for t in ('T1','T2','T3')},execution_order=['T2','T1','T3'],
        atol=c['evaluation']['repeat_atol'],rtol=c['evaluation']['repeat_rtol'],seed=20261029,
        parameters='FP32_no_AMP_no_TF32_eval_bs1_frozen',backend=c['backend'],
        evaluation=dict(T1='official_full81_and_project',T2='official_full81_and_project',
                        T3='project_only41_same_T0_subset',output_policy='unchanged_J11_J9',threshold_search=False),
        visual_groups=[dict(condition='T1',index=10),dict(condition='T2',index=19),dict(condition='T2',index=20),
                       dict(condition='T2',index=59),dict(condition='T2',index=60),dict(condition='T3',index=70)],
        allowed_inputs='Existing RGB/calibration/real timestamps/dataset ego pose; no GT/LiDAR/Radar payload',
        pose_source='nuScenes ego_pose plus calibrated_sensor; LIDAR_TOP reference-to-global; NOT visual odometry',
        limits=dict(optimizer_steps=0,backward=0,CARLA=False,dependency_update=False,main_update=False))
    write(PROTOCOL,config)
    private=Path(config['private_root']);private.mkdir(exist_ok=False)
    write(private/'frozen_protocol.json',config)
    print('Frozen:',[(s['scene_name'],s['samples'],s['reset_scene_index']) for s in scenes], 'primary forwards=203')


if __name__=='__main__':main()
