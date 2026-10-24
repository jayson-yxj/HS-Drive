"""Predeclared gate, no tuning from final data."""
from tools.m7e_common import *


def main():
    episodes=[]
    for ep in read('configs/m7e/protocol.json')['development']:
        r=read(f'reports/m7e_replays/{ep}_B.json')['records'];ev=read(f'reports/m7e_evaluations/{ep}.json')['records']
        added=[x for x in r if x['selected']=='F_state_anchored_slow' and x['baseline_selected'] in [None,'D_stop']]
        fs=[e for rr in ev for e in rr['evaluations'] if e['name']=='F_state_anchored_slow' and e['accepted']]
        episodes.append(dict(episode=ep,added_continue=len(added),span_s=(added[-1]['input_index']-added[0]['input_index'])*.05 if added else 0,
            accepted_F=len(fs),accepted_F_conflicts=sum(bool(e['measured_sample_OBB_conflicts']) for e in fs),accepted_F_road_violations=sum(bool(e['full_body_road_violation_samples']) for e in fs),accepted_F_recorded_reserve_failures=sum(not e['recorded_slow_reserve_passed'] for e in fs)))
    passed=sum(e['added_continue'] for e in episodes)>=5 and max(e['span_s'] for e in episodes)>=1 and not any(e['accepted_F_conflicts'] or e['accepted_F_road_violations'] or e['accepted_F_recorded_reserve_failures'] for e in episodes)
    write('reports/m7e_development_gate.json',dict(kind='ActualServerOfflineDevelopmentDecision',provenance=provenance(),episodes=episodes,passed=passed,decision='FreezeSingleCandidateAndEvaluateIndependentEpisodes' if passed else 'StopF_NoSecondAlgorithm',params=asset('configs/m7e/protocol.json')))
    if passed:write('reports/m7e_execution_logs/pre_final_freeze.json',dict(provenance=provenance(),gate=asset('reports/m7e_development_gate.json'),collection=asset('configs/m7e/collection.json'),tuning_after_final=False))
    print(passed,episodes)
if __name__=='__main__':main()
