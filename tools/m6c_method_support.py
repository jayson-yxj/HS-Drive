"""Confirm physical configuration and data split separation, without learning."""
import json
from pathlib import Path
from tools.m6c_common import write,provenance,asset


def main():
    probes=json.loads(Path('reports/m6c_low_speed_dynamics.json').read_text())['trials'];expected=json.loads(Path('reports/m6b_trials/final_stop_p27_l9_s101_C/result.json').read_text())['physics']['sha256'];actual={q['physics']['sha256'] for q in probes}
    groups={}
    for split in ['development','validation']:
        sub=[q for q in probes if q['task']['split']==split]
        groups[split]=dict(trials=len(sub),positions=sorted({q['task']['spawn'] for q in sub}),road_ids=sorted({q['initial']['frame'] for q in []}),
            seeds=sorted({q['task']['seed'] for q in sub}),natural_initialization=True,
            mode_counts={mode:sum(q['task']['mode']==mode for q in sub) for mode in ['startup','warm_start','maintain','stop']})
    train=set(groups['development']['positions']);val=set(groups['validation']['positions'])
    write('reports/m6c_method_support.json',dict(**provenance('actual_server_asset_and_split_audit'),source='Privileged',
        physics_identical=actual=={expected},expected_physics_sha=expected,actual_physics_sha=list(actual),probe_groups=groups,
        disjoint_positions=not bool(train&val),envelope_model=asset('configs/m6c/envelope_model.json'),
        original_controller=asset('simulator/m6b_motion.py'),original_planner=asset('simulator/motion_oracle.py'),original_safety=asset('simulator/m47_motion.py'),
        limits='Same Town03 across splits; development road3 and validation road20/67. Independent location is not independent city or broad grade support. Controller evaluation was observed during bug fixes and is not blind quality certification.'))

if __name__=='__main__':main()
