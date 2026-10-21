"""Coverage-guaranteed, multi-stratum training-view sampler and exposure audit."""
from collections import Counter
import numpy as np

def metadata(rows,labels):
    views=[]
    for i,(r,d)in enumerate(zip(rows,labels)):
        if r['split']!='train':continue
        for c,name in enumerate(r['camera_names']):
            valid=d['valid'][c];t=d['target'][c]&valid
            if not valid.any():continue
            classes=[str(k)for k in [1,2,3]if (t&(d['classes'][c]==k)).any()];dist=[f'{lo}_{hi}'for lo,hi in [(0,10),(10,20),(20,35),(35,80)]if(t&(d['radial'][c]>=lo)&(d['radial'][c]<hi)).any()];sizes=['small'if a['pixels']<=32 else'medium'if a['pixels']<=512 else'large'for a in r['actors'][c]['actors']if a['pixels']>=3]
            tags=[f'group:{r["group"]}',f'camera:{name}',f'positive:{bool(t.any())}',f'weather:{r["weather"]}',f'emptyremoved:{r["configuration"] in [min(a["configuration"]for a in rows if a["group"]==r["group"]),max(a["configuration"]for a in rows if a["group"]==r["group"])]}']+[f'class:{k}'for k in classes]+[f'distance:{k}'for k in dist]+[f'size:{k}'for k in set(sizes)]
            views.append(dict(index=i,camera_index=c,sample_id=r['sample_id'],camera=name,group=r['group'],positive=int(t.sum()),negative=int((valid&~t).sum()),tags=tags))
    return views

def schedule(views,seed,count):
    if count<len(views):raise ValueError('Budget cannot cover all eligible views')
    rng=np.random.default_rng(seed);freq=Counter(t for v in views for t in v['tags']);weights=np.array([sum(1/freq[t]for t in v['tags'])for v in views]);weights/=weights.sum()
    order=list(rng.permutation(len(views)))+rng.choice(len(views),count-len(views),p=weights).tolist()
    counts=Counter(order);return [(views[j]['index'],views[j]['camera_index'])for j in order],[dict(v,updates=counts[j])for j,v in enumerate(views)]
