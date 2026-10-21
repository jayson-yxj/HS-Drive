"""Freeze RGB backbone features, not targets, for matched cheap FPN studies."""
import json,argparse,time
from pathlib import Path
import torch
from datasets.m5a_training import VisualEvidenceDataset
from tools.m5a_learning import encoder_load,stack
from models.m5f_pixel import backbone_levels,pyramid
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash
from tools.m5f_dataset import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default=ROOT);p.add_argument('--output',default='runs/m5f_cache');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);net=encoder_load();rows=json.loads(Path(a.index,'index.json').read_text())['records'];lookup={};records=[];start=time.perf_counter()
    for row in rows:
        from tools.m5f_common import inputs
        inp=inputs(row)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            levels=backbone_levels(net.encoder,inp['images']);features={str(s):pyramid(net.encoder,levels,s).cpu() for s in [16,8]}
        path=out/(row['sample_id'].replace('/','_')+'.pt');torch.save(dict(levels=[x.cpu() for x in levels],features=features),path);records.append(dict(sample_id=row['sample_id'],path=str(path),sha256=file_hash(path)));print(row['sample_id'],flush=True)
    atomic_json(out/'index.json',dict(records=records,elapsed_s=time.perf_counter()-start,source_sha256=file_hash(__file__),encoder_checkpoint_sha256=file_hash('runs/m4_expanded_joint/best.pt'),contract='Backbonefeatures RGBonly, noGT cachedinmodelinput. Labels separate adapterfiles.'))

if __name__=='__main__':main()
