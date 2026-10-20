"""Compare interrupted/resumed training against an uninterrupted short control."""
import argparse,json,subprocess,sys
from pathlib import Path
import torch
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/m4_baseline.yaml');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    def run(name,*extra):
        subprocess.run([sys.executable,'-m','tools.train_bev','--config',a.config,'--steps','8',
                        '--output',str(root/name),*extra],check=True)
    run('part','--stop-after','4');run('finish','--resume',str(root/'part/last.pt'));run('control')
    one=torch.load(root/'finish/last.pt',map_location='cpu',weights_only=True)
    two=torch.load(root/'control/last.pt',map_location='cpu',weights_only=True)
    max_diff=max(float((one['model'][k]-v).abs().max()) for k,v in two['model'].items() if v.dtype.is_floating_point)
    def rows(name):return [json.loads(x) for x in (root/name/'metrics.jsonl').read_text().splitlines()]
    part,finish,control=rows('part'),rows('finish'),rows('control')
    checks={'continued_step':one['step']==two['step']==8,
        'optimizer_step_restored':all(int(s['step'])==8 for s in one['optimizer']['state'].values()),
        'sample_order':[x['sample_id'] for x in part+finish]==[x['sample_id'] for x in control],
        'varied_samples':len({x['sample_id'] for x in control})>1,'close_to_uninterrupted':max_diff<.005}
    report={'passed':all(checks.values()),'checks':checks,'max_abs_parameter_difference':max_diff,
        'tolerance':.005,'evidence':{str(root/n/'manifest.json'):file_hash(root/n/'manifest.json') for n in ('part','finish','control')},
        'note':'Numerical tolerance; CUDA scatter/grid_sample backward are not bitwise deterministic.'}
    atomic_json(root/'report.json',report);print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
