"""Run the predeclared equal-budget M4 comparison after capture/index acceptance."""
import argparse,json,subprocess,sys
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',default='m4_expanded')
    a=p.parse_args()
    if Path(a.prefix).name!=a.prefix:raise ValueError('Prefix must be a name')
    audit=json.loads(Path('reports/m4_driving_audit.json').read_text())
    if not audit['all_motion_accepted']:raise ValueError('Driving campaign not accepted')
    results={}
    for suffix in ('ce','depth','joint'):
        run=Path('runs')/(a.prefix+'_'+suffix);config=Path('configs')/('m4_expanded_'+suffix+'.yaml')
        if run.exists():raise FileExistsError(run)
        with Path('logs',run.name+'.log').open('x') as log:
            subprocess.run([sys.executable,'-m','tools.train_bev','--config',str(config),'--output',str(run)],stdout=log,stderr=subprocess.STDOUT,check=True)
        out=Path('reports')/(run.name+'_val')
        with Path('logs',out.name+'.log').open('x') as log:
            subprocess.run([sys.executable,'-m','tools.eval_bev','--run',str(run),'--split','val','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((out/'metrics.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
        results[suffix]={'run':str(run),'validation':str(out/'metrics.json'),
            'validation_sha256':file_hash(out/'metrics.json'),'depth_mae_m':metrics['depth']['all']['mae_m'],
            'depth_abs_rel':metrics['depth']['all']['abs_rel'],'drivable_iou':metrics['drivable_static']['iou_positive'],
            'checkpoint_sha256':manifest['best_sha256'],'step':manifest['best_step']}
        print(suffix,json.dumps(results[suffix]),flush=True)
    selected=min(('ce','depth'),key=lambda k:results[k]['depth_mae_m'])
    selection={'schema_version':'hsd-m4-comparison-v1','status':'frozen_before_test','experiments':results,
        'depth_selected_by_validation':selected,'test_runs':[selected,'joint'],
        'policy':'CE vs log-depth variant selected using full validation only; joint always evaluated as predeclared secondary experiment; test results never alter selection'}
    out=Path('reports')/(a.prefix+'_selection.json');atomic_json(out,selection)
    for suffix in selection['test_runs']:
        run=results[suffix]['run'];dest=Path('reports')/(a.prefix+'_'+suffix+'_test')
        with Path('logs',dest.name+'.log').open('x') as log:
            subprocess.run([sys.executable,'-m','tools.eval_bev','--run',run,'--split','test','--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed validation selection and frozen test evaluations',flush=True)

if __name__=='__main__':main()
