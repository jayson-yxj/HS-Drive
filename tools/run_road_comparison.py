"""Equal-budget point vs context head; freeze validation choice before regression test."""
import argparse,json,subprocess,sys
from pathlib import Path
from simulator.recorder import atomic_json,file_hash


def call(module,args,log):
    with Path(log).open('x') as stream:subprocess.run([sys.executable,'-m',module,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='m4_road');a=p.parse_args()
    results={}
    for variant in ('point','context'):
        run=f'runs/{a.prefix}_{variant}';out=f'reports/{a.prefix}_{variant}_val'
        call('tools.train_bev',['--config',f'configs/m4_road_{variant}.yaml','--output',run],f'logs/{a.prefix}_{variant}.log')
        call('tools.eval_bev',['--run',run,'--split','val','--output',out],f'logs/{a.prefix}_{variant}_val.log')
        m=json.loads(Path(out,'metrics.json').read_text());scenes=m['drivable_static']['per_scene']
        results[variant]={'val_metrics':out+'/metrics.json','sha256':file_hash(Path(out,'metrics.json')),
            'macro_scene_iou':sum(s['iou_positive'] for s in scenes.values())/len(scenes),'checkpoint_sha256':m['checkpoint_sha256']}
        print(variant,json.dumps(results[variant]),flush=True)
    selected=max(results,key=lambda k:results[k]['macro_scene_iou'])
    atomic_json(f'reports/{a.prefix}_selection.json',{'schema_version':'hsd-road-comparison-v1','status':'frozen_before_regression_test',
        'experiments':results,'selected':selected,'selection':'validation macro-scene static road IoU; individual checkpoints selected by validation macro-depth AbsRel',
        'test_scope':'Town05/Town10 historically observed; regression only, not fresh blind test'})
    run=f'runs/{a.prefix}_{selected}';out=f'reports/{a.prefix}_{selected}_test'
    call('tools.eval_bev',['--run',run,'--split','test','--output',out],f'logs/{a.prefix}_{selected}_test.log')
    print('Completed',selected,flush=True)

if __name__=='__main__':main()
