"""All 104 audit observations: represent both measurements without relabeling GT."""
import json
from pathlib import Path
import numpy as np
from datasets.m48_evidence import build_evidence
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m48_audit104_representation.json')
    if out.exists():raise FileExistsError(out)
    source=Path('reports/m47_false_free_audit.json');audit=json.loads(source.read_text());rows=[]
    for r in audit['records']:
        rays=r['free_rays'];observations=r['heldout_surface']
        d,stats=build_evidence([x['origin'] for x in rays],[x['endpoint'] for x in rays],
            [x['endpoint'] for x in observations],[x['surface_class'] for x in observations])
        rank=r['rank'];rows.append(dict(id=r['id'],surface=bool(d['surface_evidence'].flat[rank]),free=bool(d['free_ray_evidence'].flat[rank]),mixed=bool(d['mixed_cell_mask'].flat[rank]),fine_conflict=bool(d['evidence_conflict_mask'].flat[rank]),surface_valid=bool(d['surface_valid'].flat[rank]),free_valid=bool(d['free_evidence_valid'].flat[rank])))
    atomic_json(out,dict(kind='offline_local_observation_contract_replay',records=rows,count=len(rows),both=sum(r['surface'] and r['free'] for r in rows),source_audit_sha256=file_hash(source),source_sha256=file_hash(__file__),scope='Development audit replay incorporates audited surface points by design; not heldout performance or generator repair. No volume-free assertion.'))
if __name__=='__main__':main()
