"""Separate ten stage gates; short research success never grants safety."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from simulator.recorder import atomic_json,file_hash


def read(name):return json.loads(Path('reports',name+'.json').read_text())


def main():
    dest=Path('reports/m5e_acceptance.json')
    if dest.exists():raise FileExistsError(dest)
    asset=read('m5e_assets');freeze=read('m5e_pretest_decision');train=read('m5e_model_ablation');runtime=read('m5e_runtime_attribution');gpu=read('m5e_gpu_contracts');tests=[]
    for f in ['reports/m5e_final_tests.xml','reports/m5e_carla_api_tests.xml','reports/m5e_runtime_unit_tests.xml']:
        suites=list(ET.parse(f).getroot().iter('testsuite'));tests.append(dict(path=f,**{k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']},sha256=file_hash(f)))
    ck={p:file_hash(p)==h for p,h in freeze['checkpoints'].items()};code={p:file_hash(p)==h for p,h in freeze['code_sha256'].items()};sameprotocol=file_hash('configs/m5e_protocol.json')==freeze['protocol_sha256'];engineering=all(t['failures']==t['errors']==0 for t in tests) and gpu['passed'] and asset['old_assets_preserved'] and all(ck.values()) and all(code.values()) and sameprotocol
    rr=[r['result'] for r in runtime['records']+runtime['faults']];errors={k:sum(r[k] for r in rr) for k in ['collisions','lane_invasions','road_boundary_frames','normal_control_expired','frame_advancement_errors']}
    gates=dict(
        engineering_correctness=dict(status='Passed' if engineering else 'Partial',scope='Tested contracts, gradient masks, RGB-only inference, checkpoints and historical assets intact'),
        trusted_supervision_coverage=dict(status='Partial',reason='Reliable presence-only mid/far surfaces added; far Goldabsence sparse and lacks independent-group coverage. Candidate notpromoted.'),
        presence_absence_learning=dict(status='Partial',research_numerical_status='Passed' if all(r['finite_gradients'] and r['frozen_gradients_absent'] for r in train['records']) else 'Partial',reason='8 finite 200-step runs; no joint scene constraints feasible, newtest tradeoffs remain.'),
        actor_spatial_localization=dict(status='Partial',reason='Positive-only improves visible Actorrecall but absence/background errors persist, smalltargets and far support limited.'),
        depth_spatial_probability=dict(status='Partial',reason='Two matched seeds show small validation distribution improvement; error remainslarge, sparse S16actor rays and no pureCE control.'),
        high_resolution_benefit=dict(status='Partial',reason='P8/context interventions improve someActorrecall, potentialabsence costs; cannot uniquely separate features/raycenters/distribution. DefaultS16kept.'),
        multiscene_generalization=dict(status='Partial',reason='Three newtest positions only, no joint feasibility or farnegative assurance; no tuning ontest.'),
        runtime_stability=dict(status='Passed' if all(r['period_gate'] for r in runtime['records']) else 'Partial',scope='Repeated two-order controlled resource treatments',reason='Strictcycle gates separately fromaverage20Hz; finite2repeats nothardrealtime proof.'),
        oracle_driving_safety=dict(status='Partial',actual_counts=errors,reason='Finite route/fault/movingActor regression only, no generaltraffic or continuousmesh/comfort assurance.'),
        pure_visual_driving_safety=dict(status='NotValidated',reason='No predicted visualevidence connectedto safety/planner; Oracle privilegedstates retained.'),
    )
    atomic_json(dest,dict(task='M5-E',status='Partial',base=asset['base'],source_commit=asset['source_commit'],branch=asset['branch'],gates=gates,tests=tests,all_training_steps=sum(r['steps'] for r in train['records']),runtime_ticks=sum(r['ticks'] for r in rr),validation_joint_feasible_models=sum(r['selected']['feasible'] for r in train['records']),checkpoint_freeze_matches=ck,scientific_code_freeze_matches=code,protocol_hash_matches=sameprotocol,old_assets_preserved=asset['old_assets_preserved'],candidate_upgraded=False,expand_formal_occupancy_training=False,test_reused_for_tuning=False,source_sha256=file_hash(__file__),evidence={k:f'reports/{v}.json' for k,v in dict(coverage='m5e_dataset_quality',witness='m5e_witness_attrition',oldfunnel='m5e_gold_funnel_review',visibility='m5e_visibility_validation',collision='m5e_collision_visibility',models='m5e_model_ablation',presence='m5e_presence_absence',actor='m5e_actor_localization',depth='m5e_depth_spatial_probability',depth_metric='m5e_depth_metric_review',stride8='m5e_stride8_factorization',runtime='m5e_runtime_isolation',runtime_attribution='m5e_runtime_attribution',assets='m5e_assets').items()},next_steps=['New protocol for independently corroborated farnegative support and matched positive-background hardnegatives; never promoteCandidate fromplane fit alone','Improve sparse S16 object spatial supervision; preserve original48bins/defaultSplat; future CE-versusspatial objective control and matched modelselection steps','Deterministic aggregation research to quantify B/Cconstant negativecontrol variability without claiming gain fromconstant loss','Do not increase modelscale or useTemporal/SparseQuery to conceal unresolved singleframe localization','Continue server-level/tick+worker scheduling instrumentation; affinity alone not realtime guarantee']))

if __name__=='__main__':main()
