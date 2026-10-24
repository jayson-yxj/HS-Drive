"""Descriptive support inventory, frozen smoke rule, and baseline preservation."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8b_common import *
def main():
    ledger=json.loads((ROOT/'evaluation_ledger.json').read_text())['frames'];idx=json.loads((ROOT/'evaluation_index.json').read_text());rows={r['token']:r['row'] for r in idx}
    buckets={}
    for f in ledger:
        if f['core40'] or f['split']!='train':continue
        bb={'town/'+f['map']}
        for g in f['gts']:
            bb.add('class/'+g['type']);bb.add('range/'+('near_under10' if np.linalg.norm(g['box'][:2])<10 else 'far20+' if np.linalg.norm(g['box'][:2])>=20 else 'mid10to20'))
            for v in g['views']:bb.add('camera/'+v['camera'])
        for b in sorted(bb):buckets.setdefault(b,f['token'])
    old=json.loads(Path('reports/m8a_p1_release_manifest.json').read_text());bad=[a['path'] for a in old['files'] if sha(a['path'])!=a['sha256']];assert not bad
    counts=Counter()
    for f in ledger:
        for g in f['gts']:counts[g['type']]+=1
    write('reports/m8b_dataset_support.json',dict(**provenance('OfflineFrozenInputSupportInventory'),frames=len(ledger),groups=len({f['group'] for f in ledger}),split_frames=dict(Counter(f['split'] for f in ledger)),maps=dict(Counter(f['map'] for f in ledger)),GT_actors=sum(counts.values()),class_counts=dict(counts),unique_group_actor_id=len({(f['group'],g['actor_id']) for f in ledger for g in f['gts']}),GT_views=sum(len(g['views']) for f in ledger for g in f['gts']),inserted_target_counts_per_frame=dict(Counter(len(f['gts']) for f in ledger)),fixed_smoke_coverage=buckets,smoke_rule='Rule frozen before inference; first all40, plus earliest supplemental Train observation per class/camera/range/Town; only highlights subset of the same192 run, not a separate rerun',core40=dict(frames=40,annotated_targets=sum(len(f['gts']) for f in ledger if f['core40']),limitations='38/40 empty recorded-target frames; no supported Person/Cone within core40 alone'),slope_support=dict(frames_slope=sum('slope' in str(rows[f['token']]['geometry']) for f in ledger),pitch_deg=[max([abs(v['pitch_deg']) for g in f['gts'] for v in g['views']] or [0]) for f in ledger]),gaps=['No >=40m target support in selected subset','Right/front_right12 GT, rear_right4 GT; insufficient separate quality conclusion','Only one frame with2 recorded visible actors; no strong multi-object occlusion evidence','Calibration24 objects, maximum18 projection associations: fails predeclared20 matched support even before quality','No fresh blind positions','Actor-view overlap, recycled actors/layouts and same-road frames are correlated; not265 independent scenes'],historical_baseline_files_reverified=len(old['files']),baseline_mismatches=bad))
if __name__=='__main__':main()
