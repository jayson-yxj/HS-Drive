"""Pixel-weighted depth metrics and sampled surface diagnostics, not occupancy."""
import torch
from training.losses import logits_at_target_uv


class DepthMetrics:
    def __init__(self):
        self.ranges={'all':(1.5,80),'1.5-10m':(1.5,10),'10-30m':(10,30),'30-60m':(30,60),'60-80m':(60,80)}
        self.values={k:[0,0.,0.,0.,0] for k in self.ranges}

    def update(self,prediction,target,valid):
        error=(prediction-target).abs()
        for name,(low,high) in self.ranges.items():
            mask=valid&(target>=low)&(target<high);count=int(mask.sum())
            if not count:continue
            ratio=torch.maximum(prediction[mask]/target[mask],target[mask]/prediction[mask].clamp_min(1e-6))
            sums=[count,float(error[mask].sum()),float((error[mask]/target[mask]).sum()),
                  float(error[mask].square().sum()),int((ratio<1.25).sum())]
            self.values[name]=[a+b for a,b in zip(self.values[name],sums)]

    def result(self):
        return {name:{'count':v[0],'mae_m':v[1]/v[0] if v[0] else None,
            'abs_rel':v[2]/v[0] if v[0] else None,'rmse_m':(v[3]/v[0])**.5 if v[0] else None,
            'delta1':v[4]/v[0] if v[0] else None} for name,v in self.values.items()}


def depth_prediction(output,targets,inputs,centers):
    logits=logits_at_target_uv(output['depth_logits'],targets['depth_uv'],inputs['images'].shape[-2:])
    return (logits.softmax(2)*centers.view(1,1,-1,1,1)).sum(2)
