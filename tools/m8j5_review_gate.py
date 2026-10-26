"""Validate user-supplied human review records, never create approvals.

Machine validation cannot authenticate a person. The invocation must refer to
the original human-supplied local record; an explicit source reference is kept.
"""
import argparse
import hashlib
import json
from pathlib import Path

from datasets.m8j5_supervision import validate_signoff


def review_gate(candidates, responses):
    lookup={r['candidate_id']:r for r in candidates}
    seen=set();approved=[];rejected=[];pending=[]
    for e in responses:
        key=e.get('candidate_id')
        if key not in lookup or key in seen:raise ValueError('Duplicate/unknown candidate')
        seen.add(key)
        if validate_signoff(lookup[key],e):approved.append(key)
        elif e['decision']=='Reject':rejected.append(key)
        else:pending.append(key)
    pending+=sorted(set(lookup)-seen)
    return dict(approved_ids=approved,rejected_ids=rejected,pending_ids=pending,
                golden_count=len(approved),safe_negative_count=0,
                human_identity_authentication='NotProvidedByJSONValidator',
                training_readiness='NoGo_NoNegativeCertification_NoRealUpdateChainYet')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--responses',type=Path,required=True)
    ap.add_argument('--human-source-reference',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=Path(json.loads(Path('configs/m8j5_protocol.json').read_text())['private_root']).resolve()
    output=args.output.resolve()
    if root not in output.parents:raise ValueError('Review results must stay private')
    if not args.human_source_reference.strip():raise ValueError('Original human source required')
    package=json.loads((root/'golden_candidates.json').read_text())
    responses=json.loads(args.responses.read_text())
    result=review_gate(package['candidates'],responses)
    result.update(response_source=str(args.responses.resolve()),human_source_reference=args.human_source_reference,
                  response_sha256=hashlib.sha256(args.responses.read_bytes()).hexdigest())
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('golden_count','safe_negative_count','training_readiness')}))


if __name__=='__main__':main()
