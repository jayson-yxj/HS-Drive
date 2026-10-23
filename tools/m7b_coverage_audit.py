"""Post-evaluation coverage limitations, necessary stop bounds and failure ledger."""
import numpy as np
from tools.m7b_common import *


def main():
    results={};bounds=[]
    for split,file in [('development','development_v2'),('validation','validation_frozen'),('fresh_final','fresh_final_frozen'),('historical_regression','historical_regression_frozen')]:
        data=read('reports/m7b_'+file+'.json');ref_feasible_rejected=[];decision_counts={};names=[]
        for r in data['records']:
            if r['fixture']['status']!='FixtureValid':continue
            chosen=r['decisions']['B_hypotheses'];decision_counts[str(chosen)]=decision_counts.get(str(chosen),0)+1
            # Independent future-only feasibility is weaker than robust feasibility.
            # Reveal conservatism instead of defining all missed witnesses away.
            witness=[]
            for e in r['candidate_evaluations']:
                stop=e['name']=='D_stop' or e['name'].startswith('E_stop');gap=e['reference']['min_OBB_gap_m']
                if e['geometry']['planning_feasible'] and not e['reference']['collisions'] and (not stop or gap is None or gap>=3):witness.append(e['name'])
            if witness and chosen is None:ref_feasible_rejected.append(dict(scene_id=r['scene_id'],future_only_witnesses=witness,prediction_unknown=bool(r.get('prediction_unknown')),meaning='A single realized future admits a candidate; not certified under all predeclared hypotheses'))
            if r['spec']['name']=='blocked_close':
                # Necessary longitudinal straight-road bound; never asserts global 2D no-solution.
                scene=r['request']['scene'];p=np.array(r['request']['route']);yaw=np.unwrap(p[:,2]);straight=np.ptp(yaw)<1e-4
                front=scene['body_half_extent'][0]+scene['body_center_offset'][0];obj=scene['boxes'][0]
                available=r['spec']['d']-obj['half_length']-front-3
                lower=scene['ego_xy_yaw_speed'][3]**2/(2*1.8)
                bounds.append(dict(scene_id=r['scene_id'],straight_route=bool(straight),available_longitudinal_stop_distance_m=available,optimistic_max_deceleration_stop_distance_m=lower,no_feasible_straight_lane_stop=bool(straight and lower>available),B_rejected=chosen is None,scope='Necessary bound ignores jerk, so failure certifies no legal in-lane longitudinal stop; does not prove no lateral escape anywhere'))
        results[split]=dict(B_choices=decision_counts,realized_future_feasible_but_robust_rejected=ref_feasible_rejected,count=len(ref_feasible_rejected))
    write('reports/m7b_coverage_limits.json',dict(kind='PostEvaluationNoRetuning',provenance=provenance(),per_split=results,necessary_stop_bounds=bounds,certified_no_straight_lane_stop=sum(x['no_feasible_straight_lane_stop'] for x in bounds),correct_reject_of_certified_no_straight_lane_stop=sum(x['no_feasible_straight_lane_stop'] and x['B_rejected'] for x in bounds),general_no_solution_count=None,explanation='Global no-solution not established; finite candidate library, limited road geometry and finite motion hypotheses. Unknown is never a safe free-space certificate.'))
    write('reports/m7b_execution_logs/failure_ledger.json',dict(kind='PreservedFailuresAndNotRun',failures=[dict(file='reports/m7b_historical_seven.json',classification='InvalidDiagnosticOutput',reason='JSON candidate stamp remained dict; dynamic checker returned invalid; profile then raised AttributeError',replacement='reports/m7b_historical_seven_v2.json',original_preserved=True),dict(file='reports/m7b_execution_logs/history_profile.txt',classification='ActualFailedCommandTraceback',retained=True),dict(classification='FixtureInvalid',scenes=['validation_road19_lead_accelerating','historical_regression_road18_lead_accelerating'],reason='Reference reaches unsupported road end',retained=True),dict(classification='WhitespaceCheck',reason='Raw cProfile trailing blank line; code-only diff check passed, original log retained')],new_control='NotRun',C_generation='NotRun_ConditionalGateNotMet',GPU='NotRun_NoNewInferenceNeeded',Bench2Drive='NotRun',training='NotRun'))
if __name__=='__main__':main()
