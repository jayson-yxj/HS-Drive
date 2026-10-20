"""Review certified direct negatives, zero supports and old-vs-new witness coverage."""
import argparse,json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.index);dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads((root/'index.json').read_text())['samples'];rowmap={r['sample_id']:r for r in rows};pairs=json.loads((root/'m5d_pairs.json').read_text())['pairs'];labels={}
    for r in rows:
        with np.load(r['label']) as z:labels[r['sample_id']]={k:z[k] for k in z.files}
    records=[];xx=-20+(np.arange(160)+.5)*.5;yy=-30+(np.arange(120)+.5)*.5;dist=np.hypot(xx[:,None],yy[None,:])
    for r in pairs:
        with np.load(r['path']) as z:d={k:z[k] for k in z.files}
        counts={};violations=0
        for key,opp in [('appeared','a'),('vanished','b')]:
            m=d[key];lab=labels[r[opp]];veto=lab['surface_semantic'][1]|lab['ego_mask']|lab['semantic_conflict_mask']|lab['evidence_conflict_mask'];violations+=int((m&veto).sum());counts[key]=dict(certified=int(m.sum()),legacy=int(d['legacy_'+key].sum()),removed=int((d['legacy_'+key]&~m).sum()),unknown_opposite=int((m&~lab['observability']).sum()),mixed_opposite=int((m&lab['mixed_cell_mask']).sum()),by_layer=[int(m[z].sum()) for z in range(4)],by_distance={f'{lo}_{hi}':int((m&((dist>=lo)&(dist<hi))[None]).sum()) for lo,hi in [(0,10),(10,15),(15,20),(20,35),(35,80)]})
        records.append(dict(group=r['group'],split=r['split'],a=r['a_configuration'],b=r['b_configuration'],counts=counts,semantic_veto_violations=violations))
    summary={}
    for split in ['train','val','test']:
        rr=[r for r in records if r['split']==split];summary[split]=dict(groups=sorted(set(r['group'] for r in rr)),certified=sum(r['counts'][k]['certified'] for r in rr for k in ['appeared','vanished']),legacy=sum(r['counts'][k]['legacy'] for r in rr for k in ['appeared','vanished']),veto_violations=sum(r['semantic_veto_violations'] for r in rr),zero_event_pairs=sum(sum(r['counts'][k]['certified'] for k in ['appeared','vanished'])==0 for r in rr),by_distance={b:sum(r['counts'][k]['by_distance'][b] for r in rr for k in ['appeared','vanished']) for b in ['0_10','10_15','15_20','20_35','35_80']})
    atomic_json(dest,dict(kind='offline_actual_CARLA_certified_point_witness_contract',records=records,summary=summary,source_sha256=file_hash(__file__),recipe_sha256=file_hash('datasets/m5d_witness.py'),safe_for_research=all(x['veto_violations']==0 for x in summary.values()) and sum(x['certified'] for x in summary.values())>0,limits='Unknown volumetric label may coexist with a certified particular point absence. Never equate to all-cell free. Known opposite obstacle from any reliable dense ray vetoes even if excluded from V2 training pixels. Same-renderer references, no independent Mesh. Legacy mask sidecars retained for comparable historical definition.'))

if __name__=='__main__':main()
