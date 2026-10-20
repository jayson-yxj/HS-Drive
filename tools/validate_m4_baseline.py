"""Engineering gate for the requested GPU/small-data baseline, not full M4."""
import argparse,json
from pathlib import Path
import yaml
from simulator.recorder import atomic_json,file_hash
from tools.provenance import source_hashes


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--overfit',required=True)
    p.add_argument('--eval-val',required=True);p.add_argument('--eval-test',required=True)
    p.add_argument('--gpu',default='reports/m4_gpu.json');p.add_argument('--resume',default='reports/m4_resume.json')
    p.add_argument('--thresholds',default='configs/m4_acceptance.yaml');p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);fit=Path(a.overfit);r=read(run/'manifest.json');f=read(fit/'manifest.json')
    limits=yaml.safe_load(Path(a.thresholds).read_text());gpu=read(a.gpu);resume=read(a.resume)
    val,test=read(a.eval_val),read(a.eval_test)
    before,after=read(fit/'initial_train.json'),read(fit/'final_train.json')
    checks={'cuda_bf16':gpu['passed'] and gpu['checks']['bf16_supported'],
        'completed':r['status']==f['status']=='complete',
        'finite_gradients':r['all_gradients_finite'] and f['all_gradients_finite'],
        'memory_budget':max(r['peak_allocated_gib'],f['peak_allocated_gib'])<limits['max_peak_allocated_gib'],
        'checkpoint_reload':r['checkpoint_reload_exact'] and f['checkpoint_reload_exact'],
        'resume':resume['passed'],
        'overfit_depth_reduction':after['loss']['depth_ce']<before['loss']['depth_ce']*(1-limits['minimum_overfit_depth_ce_relative_drop']),
        'overfit_surface_reduction':after['loss']['surface_mse']<before['loss']['surface_mse']*(1-limits['minimum_overfit_surface_mse_relative_drop']),
        'heldout_eval':val['split']=='val' and test['split']=='test' and val['samples']>0 and test['samples']>0,
        'same_selected_checkpoint':val['checkpoint_sha256']==test['checkpoint_sha256']==r['best_sha256']==file_hash(run/'best.pt'),
        'same_index':val['index_sha256']==test['index_sha256']==r['index_sha256']==f['index_sha256'],
        'sample_sets_disjoint':not set(r['train_sample_ids'])&set(val['sample_ids']) and
                              not set(r['train_sample_ids'])&set(test['sample_ids']) and not set(val['sample_ids'])&set(test['sample_ids']),
        'rgb_only_forward':val['inference_without_supervision'] and test['inference_without_supervision'],
        'bev_interface':test['output_shapes']['bev_current']==[1,128,160,120] and test['output_shapes']['height_bev']==[1,64,4,160,120]}
    evidence=[a.gpu,a.resume,a.thresholds,a.eval_val,a.eval_test,str(run/'manifest.json'),str(fit/'manifest.json'),
              str(fit/'initial_train.json'),str(fit/'final_train.json'),'requirements-train.lock.txt']
    report={'schema_version':'hsd-m4-baseline-gate-v1','passed':all(checks.values()),'checks':checks,
        'scope':'Requested GPU and small-sample Depth/BEV engineering baseline only',
        'full_m4_closed':False,
        'quality_comparison':{split:{'model_mae_m':ev.get('depth',{}).get('all',{}).get('mae_m'),
            'camera_pixel_prior_mae_m':ev.get('camera_pixel_prior_depth',{}).get('all',{}).get('mae_m'),
            'constant_prior_mae_m':ev.get('constant_prior_depth',{}).get('all',{}).get('mae_m')}
            for split,ev in [('val',val),('test',test)]},'remaining_full_m4':['diverse driving corpus','Drivable supervision and metrics',
           'reliable depth quality across distance/weather','BEV task quality criteria and ablations'],
        'evidence':{path:file_hash(path) for path in evidence},'source_sha256':source_hashes()}
    atomic_json(a.output,report);print(json.dumps(report,indent=2));return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
