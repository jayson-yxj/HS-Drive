"""Read-only extensions to the frozen J7 matching rules."""
import numpy as np
from training.m8j7_metrics import match, distribution, prediction_mask


def vertical_errors(record,pairs):
    b,g=record['boxes'],record['gt']
    dz=[float(b[i,2]+b[i,5]/2-g[j,2]-g[j,5]/2) for i,j in pairs]
    dh=[float(b[i,5]-g[j,5]) for i,j in pairs]
    return dict(signed_dz_m=distribution(dz),abs_dz_m=distribution(np.abs(dz)),
        signed_dh_m=distribution(dh),abs_dh_m=distribution(np.abs(dh)),
        abs_width_m=distribution([abs(float(b[i,3]-g[j,3])) for i,j in pairs]),
        abs_length_m=distribution([abs(float(b[i,4]-g[j,4])) for i,j in pairs]))


def duplicate_groups(record,query_indices):
    """Query identity is only intra-forward; never a temporal object identity."""
    m=match(**{k:record[k] for k in ('boxes','scores','labels','gt','gt_labels')})
    groups={};details=[]
    for i in m['duplicates']:
        ids=np.where(record['gt_labels']==record['labels'][i])[0]
        close=[int(j) for j in ids if np.linalg.norm(record['boxes'][i,:2]-record['gt'][j,:2])<2.]
        nearest=min(close,key=lambda j:float(np.linalg.norm(record['boxes'][i,:2]-record['gt'][j,:2])))
        matched=[p for p,j in m['pairs'] if j==nearest]
        row=dict(prediction_index=int(i),query_index=int(query_indices[i]),nearest_GT_index=nearest,
            candidate_GT_indices=close,ambiguous_GT_count=len(close),
            matched_query_indices=[int(query_indices[p]) for p in matched])
        groups.setdefault(nearest,[]).append(row);details.append(row)
    # Same query, multiple class outputs are separate from same-class duplicates.
    by_query={}
    for i in m['selected']:by_query.setdefault(int(query_indices[i]),[]).append(int(i))
    multiclass=[dict(query_index=q,prediction_indices=ids,classes=sorted({int(record['labels'][i]) for i in ids}))
                for q,ids in by_query.items() if len({int(record['labels'][i]) for i in ids})>1]
    return dict(duplicate_predictions=len(details),groups=groups,details=details,
        cross_class_same_query_groups=multiclass,
        ambiguous_duplicate_predictions=sum(r['ambiguous_GT_count']>1 for r in details))
