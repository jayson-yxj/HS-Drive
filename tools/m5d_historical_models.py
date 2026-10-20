"""Frozen new models on opened historical M5C point witnesses; no model selection."""

import json
from pathlib import Path
import numpy as np
import torch
from tools.m5a_learning import encoder_load,stack
from tools.m5d_learning import predict
from datasets.m5a_training import VisualEvidenceDataset
from models.m5c_obstacle_head import VisibleObstacleHead
from models.m5d_stride8 import cache_p8
from models.m5b_cached_encoder import cache_rgb
from diagnostics.m5d_metrics import witness_metrics,aggregate_witness
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    dest=Path('reports/m5d_historical_new_models.json')
    if dest.exists():raise FileExistsError(dest)
    freeze=json.loads(Path('reports/m5d_pretest_decision.json').read_text())
    prepare_cuda();torch.set_num_threads(2)
    root=Path('/root/gpufree-data/hs-drive-data/processed/m5c_test_index')
    ds=VisualEvidenceDataset(root,'test',supervision=False);lookup={r['sample_id']:i for i,r in enumerate(ds.rows)}
    items=[ds[i] for i in range(len(ds))];net=encoder_load();cache={s:[fn(net,stack([it],'inputs')) for it in items] for s,fn in [(16,cache_rgb),(8,cache_p8)]}
    pairs=[]
    for r in json.loads((root/'m5c_pairs.json').read_text())['pairs']:
        with np.load(r['path']) as z:pairs.append((r,{k:z[k] for k in ['appeared','vanished']}))
    records=[]
    for r in freeze['models']:
        if r['name']=='P_prior':continue
        assert file_hash(r['checkpoint'])==freeze['checkpoints'][r['checkpoint']]
        ck=torch.load(r['checkpoint'],weights_only=True);net=encoder_load();head=VisibleObstacleHead().cuda().eval();head.load_state_dict(ck['head']);net.context_head.load_state_dict(ck['context']);score=predict(net,head,cache[r['stride']]);th=ck['selection']['threshold']
        rr=[dict(group=p['group'],witness=witness_metrics(score[lookup[p['a']]],score[lookup[p['b']]],d,th)) for p,d in pairs]
        records.append(dict(name=r['name'],seed=r['seed'],threshold=th,metrics=aggregate_witness(rr),checkpoint_sha256=file_hash(r['checkpoint'])))
    atomic_json(dest,dict(kind='actual_GPU_historical_openedtest_regression',records=records,source_sha256=file_hash(__file__),comparison='Same old M5C raw frames and exact original point masks as m5d_historical_same_witness.json. Different training data and selection policy; not strict crossround ablation. No tuning.'))


if __name__=='__main__':main()
