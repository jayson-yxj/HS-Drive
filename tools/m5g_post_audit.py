"""Post-freeze data, exposures and raw hashes; no label or selection changes."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m5g_common import verify_freeze
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5g_test_audit.json')
    if out.exists():raise FileExistsError(out)
    freeze=verify_freeze();root=Path('/root/gpufree-data/hs-drive-data/processed/m5g_test_pixels');rows=json.loads((root/'index.json').read_text())['records'];quality=json.loads((root/'quality.json').read_text());groups=[];sync=[]
    for g in sorted({r['group']for r in rows}):
        rr=[r for r in rows if r['group']==g];cameras={};types={};positive=negative=0;actorcfg=[]
        for r in rr:
            frame=json.loads(Path(r['raw_frame']).read_text());checks=[e[k+'_frame_id']==frame['frame_id']and abs(e[k+'_timestamp_s']-frame['timestamp_s'])<1e-5 for e in frame['cameras'].values()for k in ['rgb','depth','instance']];sync.append(dict(sample_id=r['sample_id'],all_matched=all(checks),sensors=len(checks)))
            with np.load(r['pixel_label'])as z:
                for c,name in enumerate(r['camera_names']):
                    n=int(z['target'][c].sum());v=int(z['valid'][c].sum());m=cameras.setdefault(name,dict(target_pixels=0,known_pixels=0,positive_images=0));m['target_pixels']+=n;m['known_pixels']+=v;m['positive_images']+=n>0;positive+=n;negative+=v-n
            for aid in sorted({a['actor_id']for c in r['actors']for a in c['actors']}):
                aa=[a for c in r['actors']for a in c['actors']if a['actor_id']==aid];n=sum(a['pixels']for a in aa);typ=aa[0]['type_id'];types.setdefault(typ,[]).append(n);actorcfg.append(dict(configuration=r['configuration'],actor_id=aid,type_id=typ,pixels=n,distance_m=aa[0]['distance_m']))
        groups.append(dict(group=g,frames=len(rr),positive=positive,negative=negative,cameras=cameras,types={k:dict(configurations=len(v),eligible=sum(x>=3 for x in v),small=sum(3<=x<=32 for x in v),pixel_support=v)for k,v in types.items()},actors=actorcfg))
    training=json.loads(Path('reports/m5g_backbone_ablation.json').read_text());exposures=[]
    for r in training['records']:
        if r['name']!='fpn_bce':continue
        c=Counter(tuple(x)for x in r['schedule']);exposures.append(dict(seed=r['seed'],steps=len(r['schedule']),unique_views=len(c),available=1152,never_sampled=1152-len(c),repeat_updates=len(r['schedule'])-len(c)))
    atomic_json(out,dict(kind='offline_actual_new_CARLA_capture_audit_and_schedule_counts',groups=groups,synchronization=sync,extension_exposure=exposures,frozen_checkpoint_unchanged=True,frozen_source_comparison={p:file_hash(p)==sha for p,sha in freeze['sources'].items()},source_sha256=file_hash(__file__),index_sha256=file_hash(root/'index.json'),quality_sha256=file_hash(root/'quality.json'),limits='Atmost3newpositions; supports reportednotassumed. Staticposecars mayoffroad in sideevents, perceptionstress notdrivingscenarios. Render/AABBidentityproxy and selfmask inherited.'))

if __name__=='__main__':main()
