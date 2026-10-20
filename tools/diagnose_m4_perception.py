"""Read-only fixed-checkpoint M4 diagnostics; refuses all existing output paths."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from diagnostics.experiment import RUNS,CONDITIONS,read,fit_priors,evaluate_checkpoint
from training.runner import seed_all
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/m4_depth_distribution_diagnostics.json')
    p.add_argument('--artifacts',default='reports/m4_depth_distribution_artifacts')
    p.add_argument('--smoke-frames',type=int,help='Engineering smoke only; incomplete diagnostic dataset')
    a=p.parse_args();out=Path(a.output);root=Path(a.artifacts)
    if out.exists() or root.exists():raise FileExistsError('Outputs must be new')
    manifests={n:read(Path('runs')/n/'manifest.json') for n in RUNS}
    if len({m['index_sha256'] for m in manifests.values()})!=1:raise ValueError('Checkpoints use different indices')
    train_ids=manifests[RUNS[0]]['train_sample_ids']
    index=manifests[RUNS[0]]['config']['index']
    for n,m in manifests.items():
        if m['status']!='complete' or m['train_sample_ids']!=train_ids:raise ValueError('Run status or training subset mismatch')
        if file_hash(Path('runs')/n/'best.pt')!=m['best_sha256']:raise ValueError('Checkpoint changed')
        if file_hash(Path(index)/'index.json')!=m['index_sha256']:raise ValueError('Index changed')
    # Protect all pre-existing model and report files, not only selected checkpoints.
    protected={str(p):file_hash(p) for folder in ('runs','reports') for p in Path(folder).rglob('*') if p.is_file()}
    root.mkdir(parents=True);seed_all(7);prepare_cuda()
    source={str(p):file_hash(p) for folder in ('diagnostics','models','training','datasets') for p in Path(folder).rglob('*.py')}
    source['tools/diagnose_m4_perception.py']=file_hash('tools/diagnose_m4_perception.py')
    protocol={'schema_version':'hsd-m4-perception-diagnostics-v1','split':'val','runs':list(RUNS),'conditions':list(CONDITIONS),
        'no_training':True,'original_reports_and_models_immutable':True,'common_road_mask':'legacy static lane v1',
        'smoke_frames':a.smoke_frames,'source_sha256':source,'protected_files_sha256':protected,
        'checkpoint_configuration':{n:{'config':m['config'],'best_step':m['best_step'],'last_step':m['last_step'],
            'checkpoint_sha256':m['best_sha256'],'road_overlay_manifest_sha256':m.get('road_overlay_manifest_sha256')} for n,m in manifests.items()},
        'metric_semantics':{'signed_bias':'prediction minus continuous GT; negative means underestimation',
            'median':'first discrete center with CDF >= 0.5','mode':'maximum bin probability center, not maximum density for unequal width bins',
            'entropy':'-sum(p log p)/log(48), categorical entropy with fixed LID bins',
            'window_mass':'sum of probability at centers within inclusive GT +/- window; uniform-bin variant separately named',
            'distribution_expected_absolute_error':'sum_k p_k abs(center_k-GT); distinct from abs(sum_k p_k center_k-GT)',
            'multimodality':'heuristic: two local maxima >=0.05, separation >=5m, secondary >=25% primary, valley <=50% smaller peak',
            'in_bev':'GT continuous point x[-20,60),y[-30,30),z[-2,4); same masks/rays across models',
            'bev':'matched valid rays and frozen context; continuous oracle, quantized oracle, probability splat, point decoders, pixel prior',
            'oracle_intervention':'privileged diagnostic, never model input or deployable score; frozen heads may be out-of-distribution',
            'surface':'sampled visible point counts only; zero-count cells are not Free and no semantic Occupancy claimed'},
        'perturbation_semantics':{'front_local_mask':'front camera central 50% width x 50% height replaced by that image channel mean, all other inputs fixed',
            'rgb_spatial_mean':'each view replaced with its own channel mean; destroys spatial evidence but retains color/weather',
            'feature_spatial_shuffle':'fixed seed91 pixel permutation after image encoder, shared across cameras, channel vectors preserved; camera MLP unchanged',
            'feature_zero':'post-encoder features set zero; camera MLP and downstream biases/priors remain'},
        'references':['https://arxiv.org/abs/2008.05711','https://arxiv.org/abs/2206.10092']}
    atomic_json(root/'protocol.json',protocol)
    (root/'environment.lock').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    mean,prob,road,prior=fit_priors(index,train_ids,root/'training_priors.npz');atomic_json(root/'prior_manifest.json',prior)
    results={n:evaluate_checkpoint(n,root,manifests[n],(mean,prob,road),a.smoke_frames) for n in RUNS}
    assert len({r['target_depth_uv_valid_sha256'] for r in results.values()})==1
    assert all(r['sample_ids']==results[RUNS[0]]['sample_ids'] for r in results.values())
    differences={k:{n:m['config'].get(k) for n,m in manifests.items()} for k in set().union(*(m['config'] for m in manifests.values())) if len({json.dumps(m['config'].get(k),sort_keys=True) for m in manifests.values()})>1}
    checks={'same_validation_samples':True,'same_depth_sampling_and_valid_mask':True,'no_training':True,
            'old_reports_and_models_unchanged':all(Path(p).exists() and file_hash(p)==h for p,h in protected.items())}
    # Match historical clean-RGB expectation metrics before interpreting interventions.
    if not a.smoke_frames:
        errors={n:{b:abs(r['distribution']['all_valid']['all'][b]['expected_mae_m']-read(Path('reports')/(n+'_val')/'metrics.json')['depth'][b]['mae_m']) for b in ('all','1.5-10m','10-30m','30-60m','60-80m')} for n,r in results.items()}
        checks['reproduces_historical_expected_mae_within_1e_4m']=max(v for r in errors.values() for v in r.values())<1e-4
    else:errors={}
    scene_index=read(Path(index)/'index.json')
    scenes=[{k:s[k] for k in ('scene_id','map','split','weather','frames','group_id','calibration_id')} for s in scene_index['scenes']]
    summary={'schema_version':'hsd-m4-perception-diagnostics-v1','status':'smoke' if a.smoke_frames else 'complete',
        'engineering_checks':checks,'historical_mae_absolute_reproduction_error_m':errors,'m4_quality_gate_closed':False,
        'experiment_differences':differences,'checkpoint_configuration':protocol['checkpoint_configuration'],'protocol':protocol['metric_semantics'],
        'perturbation_semantics':protocol['perturbation_semantics'],'scenes':scenes,'training_prior':prior,'results':results,
        'artifacts':str(root),'protocol_sha256':file_hash(root/'protocol.json'),'references':protocol['references']}
    atomic_json(out,summary)
    print('Completed diagnostics',json.dumps(checks),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
