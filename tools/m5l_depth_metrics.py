import torch
from models.m5e_paths import geometry
from datasets.labels import depth_bin_edges
from tools.m5k_alignment import RANGES

def depth_records(net,c,gt,pr,method,row):
    dep=gt['depth'];g=geometry(net,c,dep);centers=net.splat.depth_centers[None,None,:,None,None];edges=torch.tensor(depth_bin_edges(),device='cuda');bid=torch.bucketize(dep.contiguous(),edges,right=True)-1;ids=torch.arange(48,device='cuda')[None,None,:,None,None];exp=(pr*centers).sum(2);mode=net.splat.depth_centers[pr.argmax(2)]
    fields=dict(correct_layer=(pr*g['same_layer']).sum(2),expected_mae=(exp-dep).abs(),mode_mae=(mode-dep).abs(),bias=exp-dep,gt_bin=(pr*(ids==bid[:,:,None])).sum(2),gt_bin_pm1=(pr*((ids-bid[:,:,None]).abs()<=1)).sum(2),inside_mass=(pr*g['bin_inside']).sum(2),outside_mass=(pr*~g['bin_inside']).sum(2))
    for radius in [.5,1,2]:fields['space'+str(radius)]=(pr*g['same_layer']*(g['xy']<=radius)).sum(2)
    records=[]
    for scope in ['all','actor','vehicle','person','cone']:
      scope_mask=torch.ones_like(dep,dtype=torch.bool)if scope=='all'else gt['actor']
      if scope in ['vehicle','person','cone']:scope_mask=scope_mask&(gt['classes']=={'vehicle':1,'person':2,'cone':3}[scope])
      for band,lo,hi in RANGES:
       # One identical strict GT-in-BEV denominator for all reported depth fields.
       m=gt['valid']&g['inside']&scope_mask&(dep>=lo)&(dep<hi)
       for ci,cam in enumerate(row['camera_names']):
        mask=m[0,ci];n=int(mask.sum())
        if n:records.append(dict(method=method,group=row['group'],sample_id=row['sample_id'],scope=scope,band=band,camera=cam,n=n,sums={k:float(value[0,ci][mask].double().sum())for k,value in fields.items()}))
    return records
