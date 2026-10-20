"""Exercise signal supervision through the training loader and RGB-only inference."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from datasets.training import HSDriveDataset
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();reports={};torch.set_num_threads(2)
    for split in ('train','val','test'):
        dataset=HSDriveDataset(a.index,split);valid=0;values=set();states=set()
        if not len(dataset):raise ValueError('Empty signal split')
        for item in dataset:
            target=item.targets;valid+=int(target['signal_stop_valid']);values.add(int(target['signal_stop_required']));states.add(int(target['signal_state']))
            if any('signal' in key or 'stop' in key for key in item.inputs.as_model_kwargs()):raise ValueError('Rule field leaked into model inputs')
        if values!={-1,0,1} or states!={-1,0,2}:raise ValueError('Missing controlled states or unknown mask')
        with patch('numpy.load',side_effect=AssertionError('Inference opened supervision array')):
            item=HSDriveDataset(a.index,split,supervision=False,road_overlay='/intentionally-missing-privileged-overlay')[0]
            if item.targets is not None:raise ValueError('Inference returned targets')
        reports[split]={'samples':len(dataset),'valid_stop_targets':valid,'stop_values':sorted(values),'states':sorted(states),'rgb_only_no_npz':True}
    atomic_json(a.output,{'passed':True,'splits':reports,'index_sha256':file_hash(Path(a.index)/'index.json'),
        'scope':'Traffic targets load; learned input whitelist unchanged; no signal classifier trained'})


if __name__=='__main__':main()
