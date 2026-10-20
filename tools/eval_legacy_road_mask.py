"""Evaluate v2-trained checkpoint on both versioned road masks, same RGB/split."""
import argparse,json
from pathlib import Path
import torch
from datasets.training import HSDriveDataset
from simulator.recorder import atomic_json,file_hash
from training.runner import seed_all,initialize,load_checkpoint,evaluate
from training.runtime import prepare_cuda


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run=Path(a.run);m=json.loads((run/'manifest.json').read_text());cfg=m['config']
    if m['status']!='complete' or file_hash(run/'best.pt')!=m['best_sha256']:raise ValueError('Run identity')
    seed_all(cfg['seed']);prepare_cuda();model=initialize(cfg,pretrained=False).cuda();load_checkpoint(run/'best.pt',model)
    ds=HSDriveDataset(cfg['index'],a.split) # deliberately legacy v1 supervision
    if file_hash(Path(cfg['index'])/'index.json')!=m['index_sha256']:raise ValueError('Index changed')
    report,_=evaluate(model,ds,list(range(len(ds))),cfg,workers=cfg['workers'])
    report.update(schema_version='hsd-road-legacy-mask-v1',split=a.split,checkpoint_sha256=m['best_sha256'],
        scope='RGB unchanged; v2-trained model evaluated on original v1 Drivable mask for valid historical comparison')
    atomic_json(a.output,report);print(json.dumps(report['drivable_static'],indent=2))

if __name__=='__main__':main()
