"""Discrete-center depth distribution metrics, independent of network execution."""
import math
import numpy as np
import torch

BANDS={'all':(1.5,80.),'1.5-10m':(1.5,10.),'10-30m':(10.,30.),'30-60m':(30.,60.),'60-80m':(60.,80.)}


def distribution_metrics(probability,centers,target,gt_bin,edges):
    """[N,D] probabilities; all point estimates use the same discrete bin centers.

    Window masses use center atoms. Also report uniform-within-bin integration
    separately: this is a diagnostic assumption, not added network resolution.
    """
    p=probability.float();c=centers.float();gt=target.float()
    if p.ndim!=2 or p.shape[1]!=len(c) or not torch.isfinite(p).all() or (p<0).any():raise ValueError('Invalid probability array')
    if not torch.allclose(p.sum(-1),torch.ones_like(gt),atol=2e-5):raise ValueError('Probability mass must sum to one')
    mean=(p*c).sum(-1);mode=c[p.argmax(-1)];median=c[(p.cumsum(-1)>=.5).long().argmax(-1)]
    diff=c[None]-gt[:,None];std=((p*c.square()).sum(-1)-mean.square()).clamp_min(0).sqrt()
    out={'gt_depth_m':gt,'expected_depth_m':mean,'mode_depth_m':mode,'median_depth_m':median,
         'normalized_entropy':-(p*p.clamp_min(1e-30).log()).sum(-1)/math.log(len(c)),
         'gt_bin_probability':p.gather(1,gt_bin[:,None]).squeeze(1),
         'distribution_expected_absolute_error_m':(p*diff.abs()).sum(-1),
         'distribution_std_m':std,'max_probability':p.max(-1).values,
         'mean_mode_gap_m':(mean-mode).abs(),
         'shortfall_contribution_m':(p*(-diff).clamp_min(0)).sum(-1),
         'overshoot_contribution_m':(p*diff.clamp_min(0)).sum(-1),
         'mass_more_than_5m_nearer':(p*(diff < -5)).sum(-1),
         'mass_more_than_5m_farther':(p*(diff > 5)).sum(-1)}
    for name,value in [('expected',mean),('mode',mode),('median',median)]:
        out[name+'_mae_m']=(value-gt).abs();out[name+'_signed_bias_m']=value-gt
    for window in (1,3,5):
        out[f'mass_within_{window}m']=(p*(diff.abs()<=window)).sum(-1)
        overlap=(torch.minimum(edges[None,1:],gt[:,None]+window)-torch.maximum(edges[None,:-1],gt[:,None]-window)).clamp_min(0)
        out[f'mass_within_{window}m_uniform_bin_assumption']=(p*overlap/edges.diff()[None]).sum(-1)
    # A transparent, deliberately conservative two-peak heuristic; not a formal mixture fit.
    left=torch.cat((torch.full_like(p[:,:1],-1),p[:,:-1]),1)
    right=torch.cat((p[:,1:],torch.full_like(p[:,:1],-1)),1)
    peaks=(p>left)&(p>=right);peak_p=p.masked_fill(~peaks,-1)
    first=peak_p.argmax(-1);sep=(c[None]-c[first,None]).abs()>=5
    second=peak_p.masked_fill(~sep,-1).argmax(-1)
    p1=p.gather(1,first[:,None]).squeeze(1);p2=peak_p.gather(1,second[:,None]).squeeze(1)
    ids=torch.arange(len(c),device=p.device)[None]
    between=(ids>torch.minimum(first,second)[:,None])&(ids<torch.maximum(first,second)[:,None])
    valley=p.masked_fill(~between,float('inf')).min(-1).values
    two=(p1>=.05)&(p2>=.05)&(p2>=.25*p1)&(valley<=.5*torch.minimum(p1,p2))
    out['separated_two_peak_fraction']=two.float()
    out['confident_wrong_gt_bin_fraction']=((p.max(-1).values>=.5)&(p.argmax(-1)!=gt_bin)).float()
    out['gt_bin_center_quantization_mae_m']=(c[gt_bin]-gt).abs()
    return {k:v.detach().cpu().numpy() for k,v in out.items()}


class GroupedMetrics:
    def __init__(self):self.sums={}

    def update(self,values,*,camera,scene,motion,in_bev,xy_in_bev=None):
        gt=values['gt_depth_m'];groups={'all':np.ones(len(gt),bool),'scene/'+scene:np.ones(len(gt),bool),'motion/'+motion:np.ones(len(gt),bool)}
        for cam in np.unique(camera):
            groups['camera/'+str(cam)]=camera==cam;groups['scene_camera/'+scene+'/'+str(cam)]=camera==cam
        for scope,inside in [('all_valid',np.ones(len(gt),bool)),('gt_in_bev_3d',in_bev),('gt_in_bev_xy',in_bev if xy_in_bev is None else xy_in_bev)]:
            for group,gm in groups.items():
                for band,(lo,hi) in BANDS.items():
                    mask=inside&gm&(gt>=lo)&(gt<hi);count=int(mask.sum());key=(scope,group,band)
                    if key not in self.sums:self.sums[key]={'count':0,**{k:0. for k in values}}
                    row=self.sums[key];row['count']+=count
                    if count:
                        for k,v in values.items():row[k]+=float(v[mask].sum(dtype=np.float64))

    def result(self):
        out={}
        for (scope,group,band),row in self.sums.items():
            n=row['count'];out.setdefault(scope,{}).setdefault(group,{})[band]={'count':n,**{k:v/n if n else None for k,v in row.items() if k!='count'}}
        return out


def js_divergence(p,q):
    middle=(p+q)/2
    return .5*((p*(p.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log())).sum(-1)+(q*(q.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log())).sum(-1))
