"""Condense existing twelve-case records and compare one diagnostic working point."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from training.m8j8a_cache_diagnostics import match_record


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg=read('configs/m8j8a_protocol.json');root=Path(cfg['private_cache_root']);private=Path(cfg['private_output'])
    cases=read(private/'case_details.json');assert len(cases)<=12
    observations=[]
    for case in cases:
        row=dict(case_id=case['case_id'],fixed_slot_changes=[])
        for pair in case['fixed_query_slot_comparisons']:
            a,b=pair['A'],pair['B']
            row['fixed_slot_changes'].append(dict(slot=pair['query'],
                correct_class_score_A=a['correct_class_score'],correct_class_score_B=b['correct_class_score'],
                distance_to_same_GT_A_m=a['xy_distance_m'],distance_to_same_GT_B_m=b['xy_distance_m'],
                argmax_class_changed=a['argmax_class']!=b['argmax_class'],
                correct_class_decoded_A=a['correct_class_decoded'],correct_class_decoded_B=b['correct_class_decoded'],
                distance_crosses_outward=a['xy_distance_m']<2<=b['xy_distance_m'],
                distance_crosses_inward=b['xy_distance_m']<2<=a['xy_distance_m'],
                identity_caveat='SlotCorrespondenceOnly_NotTrackedPhysicalIdentity'))
        observations.append(row)
    # Coupled cases: check whether the same selected slot changes its assigned GT.
    coupled=[]
    for left in cases:
        if left['bucket']!='A_only':continue
        for right in cases:
            if right['bucket']!='B_only' or left['frame']!=right['frame']:continue
            ma=left['models']['A']['matched'];mb=right['models']['B']['matched']
            if ma['query']==mb['query']:
                coupled.append(dict(A_only=left['case_id'],B_only=right['case_id'],frame=left['frame'],query=ma['query'],
                    observation='SameDecoderSlotAssignedToDifferentGroundTruthAcrossCheckpoints',
                    not_physical_tracking=True))
    pr=read('reports/m8j8a_PR_diagnostic.json');point=pr['B_relative_to_original_A']['recall_first_reaching']
    transition=read(private/'all_GT_transitions.json');frame_rows={}
    for row in transition:frame_rows.setdefault(row['frame'],[]).append(row)
    counts=dict(common=0,A_only=0,B_only=0,neither=0,recovered_formal_A_only=0,
                recovered_formal_neither=0,formal_B_only_still_matched=0)
    matched_b_total=0
    if point is not None:
        for i,rows in frame_rows.items():
            with np.load(root/'B/mini_val'/f'{i:02}.npz',allow_pickle=False) as z:r={k:z[k].copy() for k in z.files}
            matched=match_record(r,point['score_threshold']);ids={j for _,j in matched['pairs']}
            matched_b_total+=len(ids)
            for row in rows:
                aa=row['A_prediction'] is not None;bb=row['GT_index'] in ids
                key=('common' if bb else 'A_only') if aa else ('B_only' if bb else 'neither')
                counts[key]+=1
                counts['recovered_formal_A_only']+=int(row['state']=='A_only' and bb)
                counts['recovered_formal_neither']+=int(row['state']=='neither' and bb)
                counts['formal_B_only_still_matched']+=int(row['state']=='B_only' and bb)
        assert matched_b_total==point['TP']
    write('reports/m8j8a_evidence_notes.json',dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        original_twelve_case_file_sha256=sha(private/'case_details.json'),
        cases=observations,coupled_selected_cases=coupled,
        diagnostic_equal_recall_GT_overlap=dict(threshold=point['score_threshold'],counts=counts,
            warning='SameTPCountDoesNotMeanSameGTSet. PosthocOnly; fixed0.2results remain unchanged.'),
        new_cases_added=0,new_model_forwards=0,new_optimizer_steps=0))
    print(json.dumps(dict(cases=len(cases),coupled=coupled,equal_recall_overlap=counts),indent=2))


if __name__=='__main__':main()
