"""Audit completed expansion artifacts, separating engineering completion from quality."""
import argparse,json
from pathlib import Path
import yaml
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='m4_expanded')
    p.add_argument('--output',default='reports/m4_expansion_acceptance.json');a=p.parse_args()
    selection=read(f'reports/{a.prefix}_selection.json');campaign=yaml.safe_load(Path('configs/capture_expanded.yaml').read_text())
    motion=read('reports/m4_driving_audit.json');checks={};runs={};evs={};evidence={}
    first=None
    for kind in ('ce','depth','joint'):
        path=Path(f'runs/{a.prefix}_{kind}');m=read(path/'manifest.json');runs[kind]=m
        cfg=m['config'];root=Path(cfg['index']);index=read(root/'index.json')
        checks[kind+'_complete']=m['status']=='complete' and m['last_step']==cfg['steps']==1200
        checks[kind+'_finite_reload']=m['all_gradients_finite'] and m['checkpoint_reload_exact']
        checks[kind+'_memory']=m['peak_reserved_gib']<20
        checks[kind+'_checkpoint_identity']=all(file_hash(path/(k+'.pt'))==m[k+'_sha256'] for k in ('best','last'))
        checks[kind+'_index_identity']=file_hash(root/'index.json')==m['index_sha256']
        checks[kind+'_selection_identity']=selection['experiments'][kind]['checkpoint_sha256']==m['best_sha256']
        train=set(m['train_sample_ids']);all_train={r['sample_id'] for r in index['samples'] if r['split']=='train'}
        checks[kind+'_train_ids']=train<=all_train and len(train)==len(m['train_sample_ids'])>0
        for split in ('val','test'):
            f=Path(f'reports/{a.prefix}_{kind}_{split}/metrics.json')
            if split=='test' and kind not in selection['test_runs']:continue
            r=read(f);evs[kind+'_'+split]=r;evidence[str(f)]=file_hash(f)
            expected=[row['sample_id'] for row in index['samples'] if row['split']==split]
            checks[kind+'_'+split+'_full_split']=r['sample_ids']==expected and r['samples']==len(expected)
            checks[kind+'_'+split+'_no_overlap']=not train&set(expected)
            checks[kind+'_'+split+'_identities']=r['index_sha256']==m['index_sha256'] and r['checkpoint_sha256']==m['best_sha256']
            checks[kind+'_'+split+'_rgb_only']=r['inference_without_supervision']
            checks[kind+'_'+split+'_bev_interface']=r['output_shapes']['bev_current']==[1,128,160,120]
            if kind=='joint':checks[kind+'_'+split+'_head_interface']=r['output_shapes']['drivable_logits']==[1,160,120]
        if first is None:first=m
        else:
            checks[kind+'_equal_sample_plan']=m['train_sample_ids']==first['train_sample_ids'] and m['val_sample_ids']==first['val_sample_ids'] and cfg['seed']==first['config']['seed']
        evidence[str(path/'manifest.json')]=file_hash(path/'manifest.json')
    expected={r['run_id'] for r in campaign['runs'] if r.get('include_training',True)}
    included={s['scene_id'] for s in index['scenes'] if s['scene_id'].startswith('m4drive')}
    checks['accepted_campaign_exact']=expected==included and motion['all_motion_accepted']
    checks['raw_integrity_all']=all(s['integrity']['recording_integrity_checks_passed'] for s in index['scenes'])
    checks['map_test_only']=all((s['split']=='test')==(s['map'] in index['split_policy']['test_maps']) for s in index['scenes'])
    groups={}
    for s in index['scenes']:groups.setdefault(s['group_id'],set()).add(s['split'])
    checks['group_disjoint']=all(len(v)==1 for v in groups.values())
    checks['static_label_geometry']=index['drivable_bev_config_sha256']==file_hash(runs['joint']['config']['bev_config'])
    checks['real_inference_target_io_forbidden']=read('reports/m4_static_leakage_real.json')['inference_npz_io_forbidden_passed']
    for name in ('val','drive','turn','val_retry'):
        checks['alignment_'+name]=read('reports/m4_static_alignment_'+name+'/report.json')['alignment_passed']
    joint_val=evs['joint_val'];joint_test=evs['joint_test']
    quality={'joint_val_depth_beats_pixel_prior':joint_val['depth']['all']['mae_m']<joint_val['camera_pixel_prior_depth']['all']['mae_m'],
        'joint_test_depth_beats_pixel_prior':joint_test['depth']['all']['mae_m']<joint_test['camera_pixel_prior_depth']['all']['mae_m'],
        'joint_val_static_beats_location_prior':read('reports/m4_drivable_prior_val.json')['model_beats_location_prior'],
        'joint_test_static_beats_location_prior':read('reports/m4_drivable_prior_test.json')['model_beats_location_prior'],
        'joint_val_static_beats_all_positive':joint_val['drivable_static']['iou_positive']>joint_val['drivable_static']['all_positive_prior_iou'],
        'joint_test_static_beats_all_positive':joint_test['drivable_static']['iou_positive']>joint_test['drivable_static']['all_positive_prior_iou']}
    report={'schema_version':'hsd-m4-expansion-audit-v1','engineering_passed':all(checks.values()),'engineering_checks':checks,
        'quality_diagnostics':quality,'full_m4_closed':False,
        'scope':'Post-run artifact/engineering audit; quality comparisons are diagnostics, not retrospective full-M4 acceptance thresholds',
        'evidence':evidence,'source_sha256':source_hashes()}
    atomic_json(a.output,report);print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','evidence')},indent=2))
    if not report['engineering_passed']:raise SystemExit(1)

if __name__=='__main__':main()
