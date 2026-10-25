"""Offline explanation of failed support observations; no new ray or actor experiment."""
import json
from pathlib import Path
import numpy as np
from tools.m8h1_common import *

def main():
 rows=[]
 for p in sorted((ROOT/'capture').glob('*/ego_present_review.json')):
  r=json.loads(p.read_text());g=r['geometry'];support=g['observed_pose_support_result'];prior=r['preflight'];pointrows=[]
  for i,(rr,old) in enumerate(zip(support['rays'],prior['rays'])):
   ground=old['decision'].get('selected');floor=support['bottom_samples'][i];pointrows.append(dict(sample=i,current_bottom=floor,current_query=rr,pre_actor_selected_surface=ground,current_bottom_minus_prior_ground_m=None if ground is None else floor[2]-ground['xyz'][2],ray_start_minus_prior_ground_m=None if ground is None else rr['start'][2]-ground['xyz'][2],XY_surface_sample_shift_m=None if ground is None else float(np.linalg.norm(np.array(floor[:2])-ground['xyz'][:2])),reference='Prior surface at nearly sameXY, before actor. Not a replacement current support hit.'))
  actual=np.array(g['actual_local_box']['T_world_from_box']);old=np.array(prior['asset']['local_box_matrix']);dim=np.array(g['actual_local_box']['extent_m'])-prior['asset']['extent'];rows.append(dict(template=p.parent.name,review=asset(p),settling_status=r['settling']['status'],point_rows=pointrows,local_box_matrix_difference=(actual-old).tolist(),extent_difference_m=dim.tolist(),max_local_matrix_difference=float(abs(actual-old).max()),local_box_interpretation='Frozen old local matrix was reconstructed via inverse world transform; tiny translation discrepancy consistent with precision, not proven changed mesh. Exact provenance mismatch preserved; no tolerance changed.',support_interpretation='Frozen ray starts below actual OBB floor; floor lies slightly below historical road proxy so downward ray can miss road. NoRayHit is not proof of floating; contract remains unresolved.',acceptance_override=False))
 write('reports/m8h1_support_diagnostic.json',dict(**provenance('OfflineActualSettlingVersusHistoricalSupport'),rows=rows,no_new_queries=True,no_rule_change=True,no_second_actor_attempt_after_genuine_gate_failure=True))
 print('offline support diagnosis',len(rows),'templates',flush=True)
if __name__=='__main__':main()
