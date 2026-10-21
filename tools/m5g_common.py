"""M5G readers and fixed reproducible schedules, old experiment data read-only."""
import json
from pathlib import Path
import torch,numpy as np
from tools.m5f_dataset import ROOT
from tools.m5f_common import inputs,pixel_label
from models.m5f_pixel import PixelHead,pyramid,backbone_levels
from tools.m5a_learning import encoder_load
from simulator.recorder import file_hash


def rows(root=ROOT):return json.loads(Path(root,'index.json').read_text())['records']


def schedule(records,labels,seed,count):
    rng=np.random.default_rng(seed);groups=sorted({r['group']for r in records if r['split']=='train'});s=[]
    while len(s)<count:
        for g in rng.permutation(groups):
            i=int(rng.choice([i for i,r in enumerate(records)if r['group']==g]));positive=[c for c in range(6)if labels[i]['target'][c].any()];cam=int(rng.choice(positive))if positive and rng.random()<.5 else int(rng.integers(6));s.append((i,cam))
    assert s[:400]==[tuple(x)for x in json.loads(Path(f'runs/m5f_learning/m5f_schedule_{seed}.json').read_text())['schedule']]
    return s[:count]


def load_pixel(record):
    if file_hash(record['checkpoint'])!=record['sha256']:raise ValueError('pixelSHA')
    ck=torch.load(record['checkpoint'],weights_only=True);net=encoder_load();head=PixelHead().cuda().eval();head.load_state_dict(ck['head'])
    net.encoder.lateral.load_state_dict(ck['lateral']);net.encoder.output.load_state_dict(ck['output'])
    if 'layer3'in ck:net.encoder.backbone.layer3.load_state_dict(ck['layer3']);net.encoder.backbone.layer4.load_state_dict(ck['layer4'])
    return net,head,ck


def verify_freeze():
    frozen=json.loads(Path('reports/m5g_pretest_freeze.json').read_text())
    assert file_hash('configs/m5g_protocol.json')==frozen['protocol_sha256']
    for path,sha in frozen['checkpoints'].items():
        if file_hash(path)!=sha:raise ValueError('changed frozencheckpoint '+path)
    return frozen
