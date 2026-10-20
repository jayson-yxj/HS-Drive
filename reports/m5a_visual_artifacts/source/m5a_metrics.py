"""Observation-event scores, exact step PR integral (AP), never safety probabilities."""
import numpy as np


def binary_metrics(prob,target,mask,threshold=.5):
    p=np.asarray(prob)[mask].astype(float);y=np.asarray(target)[mask].astype(bool);b=p>=threshold;n=len(y)
    tp=int((b&y).sum());fp=int((b&~y).sum());fn=int((~b&y).sum());tn=n-tp-fp-fn
    ap=None
    if y.any():
        order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)!=0),n-1];ct=ys.cumsum()[ends];rec=ct/y.sum();prec=ct/(ends+1);ap=float(np.sum(np.diff(np.r_[0.,rec])*prec))
    ece=0.
    if n:
        for low in np.linspace(0,.9,10):
            ix=(p>=low)&(p<low+.1 if low<.89 else p<=1)
            if ix.any():ece+=ix.mean()*abs(p[ix].mean()-y[ix].mean())
    return dict(n=n,positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,iou=tp/(tp+fp+fn) if tp+fp+fn else None,pr_auc_ap=ap,brier=float(np.mean((p-y)**2)) if n else None,ece10=float(ece) if n else None,threshold=threshold)


def temperature_fit(logits,target,mask):
    # Scalar temperature + bias, event-label calibration; val ONLY.
    import torch
    x=torch.as_tensor(np.asarray(logits)[mask],dtype=torch.float64);y=torch.as_tensor(np.asarray(target)[mask],dtype=torch.float64)
    if not len(y) or y.min()==y.max():return dict(temperature=1.,bias=0.,fitted=False)
    lt=torch.zeros((),dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([lt,bias],lr=.3,max_iter=40,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();z=x/lt.clamp(-3,3).exp()+bias.clamp(-15,15);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward();return loss
    opt.step(closure);return dict(temperature=float(lt.detach().clamp(-3,3).exp()),bias=float(bias.detach().clamp(-15,15)),fitted=True)
