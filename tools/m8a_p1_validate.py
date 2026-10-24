"""Validate delivered evidence, not merely file existence."""
import json
from pathlib import Path
from tools.m8a_p1_common import LOG,CFG,write,asset,provenance,sha

def main():
    def load(name):return json.loads(Path('reports/m8a_p1_'+name+'.json').read_text())
    acceptance=load('acceptance');assert acceptance['overall']=='FeasibilityPassed_QualityNotValidated'
    audit=load('checkpoint_key_audit');assert audit['checkpoint_tensor_count']==audit['model_tensor_count']==558
    assert not audit['missing'] and not audit['unexpected'] and not audit['shape_mismatches']
    forward=load('forward_results');assert forward['input_shape']==[1,6,3,512,1408] and forward['all_finite']
    all_outputs=json.loads(Path(forward['complete_outputs']['path']).read_text())
    assert len(all_outputs['outputs'])==24 and len(all_outputs['full_model_dcn_calls'])==9
    assert all(len(x['boxes'])==len(x['scores'])==len(x['labels'])==300 for x in all_outputs['outputs'])
    assert sha(forward['complete_outputs']['path'])==forward['complete_outputs']['sha256']
    model=load('model_load');assert model['strict'] and model['all_loaded_tensors_bitwise_equal']
    assert model['checkpoint']['sha256']==CFG['checkpoint_sha256']
    memory=load('gpu_memory_latency');assert len(memory['measured_ms'])==20 and not memory['p99_certified'] and not memory['shared_gpu_20hz_certified']
    assert memory['max_memory_allocated_bytes']<24*2**30
    camera=load('camera_contract');assert camera['status']=='Passed_BasicReadOnlyContract' and not camera['carla_model_forward'] and not camera['GT_model_input']
    assert len(camera['camera_checks'])==6
    lock=load('environment_lock');assert all(x['unchanged'] for x in lock['historical_assets'])
    assert all(lock['main_environments']['equal'].values())
    assert acceptance['separate_status']['perception_quality']=='NotValidated' and acceptance['training_steps']==0
    assert load('failure_ledger')['failures']
    required=['summary.md','acceptance.json','environment_lock.json','dependency_matrix.json','cuda_operator_tests.json','model_load.json','checkpoint_key_audit.json','forward_results.json','gpu_memory_latency.json','camera_contract.json','failure_ledger.json']
    files=[Path('reports/m8a_p1_'+x) for x in required]+[Path('docs/M8A_P1_IMPLEMENTATION.md'),Path('docs/M8A_P1_ARCHITECTURE_DECISION.md')]
    write(LOG/'delivery_validation.json',dict(**provenance('OfflineDeliveryCrossChecks'),status='Passed',files=[asset(x) for x in files],
        unit_tests=asset(LOG/'unit_tests_final.txt'),GPU_run='probe13',camera_run='camera01',quality_and_safety_gates_open=True))
    print('Delivery and measured-evidence cross-checks Passed')
if __name__=='__main__':main()
