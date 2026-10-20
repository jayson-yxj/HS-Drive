"""Gold versus presence-only coverage, all independent-group and distance support."""
import argparse,json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',nargs='+',required=True);p.add_argument('--output',default='reports/m5e_dataset_quality.json');a=p.parse_args();dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    records=[];groups=[]
    x=-20+(np.arange(160)+.5)*.5;y=-30+(np.arange(120)+.5)*.5;rad=np.hypot(x[:,None],y[None,:])
    for index in a.indices:
        root=Path(index);rows=json.loads((root/'index.json').read_text())['samples'];supervision=json.loads((root/'m5e_supervision.json').read_text());surface=json.loads((root/'m5e_surface_index.json').read_text());ss={r['sample_id']:r for r in supervision['records']};sr={r['sample_id']:r for r in surface['records']};pairs=json.loads((root/'m5e_pairs.json').read_text())['pairs']
        for row in rows:
            s=ss[row['sample_id']];r=sr[row['sample_id']]
            with np.load(row['label']) as z:lab={k:z[k] for k in z.files}
            with np.load(s['path']) as z:presence=z['presence']
            records.append(dict(sample_id=row['sample_id'],group=row['group'],split=row['split'],configuration=row['configuration'],presence_cells=s['presence_cells'],by_distance=s['by_distance'],cameras=s['cameras'],actor_types=[a['type_id'] for a in r['actors']],actor_surface_counts=r['actor_visible_cells'],presence_by_layer=presence.sum((1,2)).tolist(),mixed_presence=int((presence&lab['mixed_cell_mask']).sum()),ground=int((lab['surface_semantic'][0]&lab['surface_semantic_valid']).sum()),obstacle=int((lab['surface_semantic'][1]&lab['surface_semantic_valid']).sum()),unknown=int((~lab['observability']).sum()),total_cells=int(presence.size)))
        for group in sorted({r['group'] for r in rows}):
            rr=[r for r in records if r['group']==group];pp=[p for p in pairs if p['group']==group];bands={}
            for lo,hi in [(0,10),(10,15),(15,20),(20,25),(25,35),(35,80)]:
                n=0;unique=np.zeros((4,160,120),bool);pix=0
                for pair in pp:
                    with np.load(pair['path']) as z:
                        m=(z['appeared']|z['vanished'])&((rad>=lo)&(rad<hi))[None];n+=int(m.sum());unique|=m
                        pix+=int(((z['witness_count_a']+z['witness_count_b'])*m).sum())
                band=f'{lo}_{hi}';bands[band]=dict(presence_cells=sum(r['by_distance'][band] for r in rr),visible_target_pixels=sum(c['by_distance'][band] for r in rr for c in r['cameras']),gold_paired_events=n,gold_unique_cells=int(unique.sum()),gold_support_pixel_votes=pix)
            counts=[n for r in rr for n in r['actor_surface_counts']];groups.append(dict(group=group,split=rr[0]['split'],frames=len(rr),actor_configurations=len(counts),zero_support_actors=sum(n==0 for n in counts),eligible_actors=sum(n>=3 for n in counts),by_distance=bands))
    summary={}
    for split in ['train','val','test']:
        rr=[g for g in groups if g['split']==split];summary[split]=dict(groups=[g['group'] for g in rr],frames=sum(g['frames'] for g in rr),by_distance={b:dict(presence_cells=sum(g['by_distance'][b]['presence_cells'] for g in rr),gold_events=sum(g['by_distance'][b]['gold_paired_events'] for g in rr),gold_groups=sum(g['by_distance'][b]['gold_paired_events']>0 for g in rr),presence_groups=sum(g['by_distance'][b]['presence_cells']>0 for g in rr)) for b in ['0_10','10_15','15_20','20_25','25_35','35_80']})
    atomic_json(dest,dict(kind='offline_actual_CARLA_new_group_supervision_contract',records=records,groups=groups,summary=summary,source_sha256=file_hash(__file__),candidate_negatives=False,limits='Gold same M5D; presence doesnotrequireabsence. Pixelvotes/configurationevents correlated; actorconfigurations notunique actoridentities. Other/unknown labels notnegative. No candidateupgrade. Rawcamera frames synchronized, immutablecalibration, sameegoPose validatedbyindex.'))


if __name__=='__main__':main()
