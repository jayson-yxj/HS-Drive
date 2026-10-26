"""No real forward/backward: replay all six layers of frozen J5B predictions."""
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
import torch

from datasets.m8j5b_loss_adapter import load_coverage_supervision
from training.m8j5bv_validated_loss import ValidatedCoveragePETRLoss
from tools.m8j5bv_run import ROOT,PUBLIC


def main():
    import mmcv
    import projects.mmdet3d_plugin
    from mmdet3d.models import build_head
    torch.manual_seed(850541)
    runtime=Path(os.environ['M8J5_RUNTIME']);old=Path(os.environ['M8J5B_PRIVATE'])
    cfg=mmcv.Config.fromfile(str(runtime/'sources/PETR/projects/configs/petr/petr_r50dcn_gridmask_c5.py'))
    head=build_head(dict(cfg.model.pts_bbox_head,train_cfg=cfg.model.train_cfg.pts)).cuda().eval().requires_grad_(False)
    loss=ValidatedCoveragePETRLoss(head);rows=[];full=[]
    before=json.loads((old/'predictions_run2/run.json').read_text())
    for reference in before['rows']:
        n=reference['ordinal'];path=old/'predictions_run2'/(n+'_raw.npz')
        with np.load(path,allow_pickle=False) as z:
            predictions=dict(all_cls_scores=torch.from_numpy(z['cls']).cuda(),all_bbox_preds=torch.from_numpy(z['box']).cuda())
        s,record=load_coverage_supervision(old,n)
        with torch.no_grad(),warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always');losses,audit=loss(predictions,[s])
        for i,(a,b) in enumerate(zip(audit,reference['layers'])):
            assert a['positive']+a['ignore']+a['negative']==900
            assert [a[k] for k in ('positive','ignore','negative')]==[b[k] for k in ('positive','ignore','negative')]
            assert a['samples'][0]['state']==b['samples'][0]['state']
            assert a['classification_weight_sum']==a['positive']
            assert a['classification_loss_negative']==a['classification_loss_ignore']==0.
        assert all(abs(float(v)-reference['losses'][k])<1e-5 for k,v in losses.items())
        full.append(dict(ordinal=n,layers=audit))
        public=json.loads(json.dumps(audit))
        for a in public:
            for sample in a['samples']:
                sample.pop('state');sample.pop('matched_gt_indices');sample.pop('positive_class_ids')
        rows.append(dict(ordinal=n,layers=public,raw_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            historical_counts_and_query_states_equal=True,losses={k:float(v) for k,v in losses.items()},
            warnings=sorted({str(w.message) for w in warns}),negative_certificate='NotAvailable_FrameFallback'))
    suffix=os.environ['M8J5BV_SUFFIX']
    with (ROOT/('replay_private_'+suffix+'.json')).open('x') as f:json.dump(full,f,indent=2)
    result=dict(evidence='ActualFrozenPredictionReplay_ActualSidecar_NoRealForwardNoBackward',rows=rows,
        real_model_calls=0,real_backward_calls=0,optimizer_steps=0,GT_private=True,
        header='P/I/N are per layer; correlated observations, not independent objects')
    with (PUBLIC/'m8j5bv_decoder_statistics.json').open('x') as f:json.dump(result,f,indent=2)
    lines=['# 六层 Decoder 实际缓存回放统计','',
        '每行900 Query；I=Ignore，不是独立Actor数。P/N Loss使用同一分类归一化因子。I直接损失贡献均为0。',
        '', '|序号|层|P|I|N|分类权重和|有效比例|分类P Loss|分类N Loss|回归Loss|分类/回归分母|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        for a in r['layers']:
            lines.append('|%s|%d|%d|%d|%d|%d|%.4f%%|%.6f|%.6f|%.6f|%.1f / %.1f|'%(r['ordinal'],a['layer'],a['positive'],a['ignore'],a['negative'],
                a['classification_weight_sum'],a['effective_fraction']*100,a['classification_loss_positive'],a['classification_loss_negative'],a['regression_loss'],a['cls_avg_factor'],a['reg_avg_factor']))
    with (PUBLIC/'m8j5bv_decoder_statistics.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(frames=len(rows),layers=24,P=66,I=21534,N=0,counts_equal=True)))


if __name__=='__main__':main()
