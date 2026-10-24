"""Materialize P0 conclusions from retained measured results; no training or CARLA."""
import json
from pathlib import Path
import re
import shutil
from tools.m8a_p1_common import ROOT,LOG,CFG,write,asset,provenance


def main():
    source=ROOT/'artifacts/probe13';camera=ROOT/'artifacts/camera01'
    execution=json.loads((LOG/'gpu_probe13.json').read_text())
    assert execution['exit_code']==0
    p=provenance('ActualRTX4090_SyntheticInputSmoke')
    p['execution_git_commit']=execution['git_head'];p['execution_source_hashes']=execution['sources']
    out=Path('reports/m8a_p1_results');out.mkdir(exist_ok=False)
    for path in sorted(source.iterdir()):
        if path.suffix=='.json':shutil.copy2(path,out/path.name)
    shutil.copy2(camera/'camera_contract.json',out/'camera_contract.json')
    def get(name):return json.loads((source/(name+'.json')).read_text())
    def report(name,kind,data):write('reports/m8a_p1_'+name+'.json',dict(provenance=dict(p,kind=kind),**data))
    operators=get('cuda_operators');model=get('model_load');keys=get('checkpoint_key_audit');forward=get('forward_results');memory=get('gpu_memory_latency')
    assert all(x['status']=='Passed' for x in [operators,model,forward,memory])
    assert not keys['missing'] and not keys['unexpected'] and not keys['shape_mismatches']
    assert len(forward['outputs'])==24 and len(memory['measured_ms'])==20 and len(forward['full_model_dcn_calls'])==9
    assert forward['input_shape']==CFG['input_shape']
    report('checkpoint_key_audit','ActualCompleteModelTensorAudit',dict(status='Passed',**keys,source=asset(source/'checkpoint_key_audit.json'),
        safe_extraction=json.loads((ROOT/'artifacts/safe_extract/safe_extract.json').read_text())))
    report('model_load','ActualStrictModelLoad',dict(**model,checkpoint=asset(CFG['checkpoint']),model_keys_total=keys['model_tensor_count'],
        missing_keys=[],unexpected_keys=[],shape_mismatches=[],torch_strict_load=True,optimizer_loaded=False,
        original_pickle_read='CPU weights_only=True inside bwrap network/filesystem namespace; GPU reads hash-checked numeric NPZ allow_pickle=False',
        GPU_isolation='New venv, clean environment, unshare --net. GPU subprocess has host filesystem; original checkpoint never unpickled there.',
        original_PETR_code_changed=False))
    arch=sorted(set(re.findall(r'sm_\d+', (LOG/'mmcv_binary_architecture.txt').read_text())))
    report('cuda_operator_tests','ActualCUDAKernelExecution',dict(**operators,mmcv_binary_architectures=arch,
        full_model_dcn_calls=forward['full_model_dcn_calls'],head_output_shapes=forward['head_shape_trace'],
        interpretation='MMCV cubins include sm86, usable on sm89 through same-major binary compatibility; actual DCN CUDA trace confirms execution. Driver-chosen cubin not directly traced.',
        separately_compiled_mmdet3d_architecture='sm_89',
        scope='Torch matmul and DCNv2 explicitly GPU tested; full PETR backbone/head/transformer executed. Other11 legacy extensions compile/load only; unused lidar kernels not all runtime-tested.',
        profiler_trace=asset(out/'dcn_cuda_trace.json'),reference='https://docs.nvidia.com/cuda/ada-compatibility-guide/'))
    first=forward['outputs'][0]
    report('forward_results','ActualRTX4090_SyntheticInputSmoke',dict(status='Passed',input_shape=forward['input_shape'],precision=forward['precision'],
        official_input_available=False,input_kind='SyntheticInputSmoke_NotNuScenesEvaluation',synthetic_calibration=asset(out/'synthetic_input.json'),
        no_GT_input=True,all_finite=True,frames=24,measured_frames=20,queries=900,decoder_layers=6,classes=10,
        head_shape_trace=forward['head_shape_trace'],DCNv2_modules_executed=9,box_shape=first['shape'],bbox_count=len(first['boxes']),
        output_box_semantics='old MMDetection3D LiDAR bottom-center x,y,z,width,length,height,clockwise yaw,vx,vy',
        all_outputs_bitwise_equal=all(x['boxes']==first['boxes'] and x['scores']==first['scores'] and x['labels']==first['labels'] for x in forward['outputs']),
        score_range=[min(first['scores']),max(first['scores'])],label_indices=sorted(set(first['labels'])),
        output_interpretation='300 decoder top-k hypotheses at very low scores on random imagery, not300 validated physical detections',
        complete_outputs=asset(out/'forward_results.json'),CARLA_detection_quality='NotRun',nuScenes_quality='NotRun'))
    attribution=json.loads((LOG/'gpu_process_memory_attribution.json').read_text())
    report('gpu_memory_latency','ActualSynchronizedCUDAForwardMeasurements',dict(**memory,
        process_memory_samples=attribution,process_sampled_max_MiB=1752,process_sampled_max_GiB=1752/1024,
        allocated_peak_GiB=memory['max_memory_allocated_bytes']/2**30,reserved_peak_GiB=memory['max_memory_reserved_bytes']/2**30,
        initialization_ms=operators['device']['cuda_initialization_ms'],
        cold_definition='First complete model forward in this process; Torch/DCN probes already loaded some CUDA kernels. Includes first-forward finite/DCN hooks; not a pristine driver-cache cold-start benchmark.',
        steady_timing_excludes=['RGB generation and CPU preprocessing','H2D input transfer','JSON serialization','output CPU serialization','CARLA and planner'],
        steady_timing_includes=['full RGB backbone','PETR head/transformer','box decoding','CUDA synchronize'],
        caveats=['20 samples do not certify P99 or worst-case latency','No concurrent CARLA/HS model load; baseline GPU process436MiB remained','P50 already exceeds50ms; strict20Hz and sharedGPU safety stay open','Process sample peak is temporal hostPID attribution at250ms, not exact instantaneous peak']))
    c=json.loads((camera/'camera_contract.json').read_text());assert c['status']=='Passed_BasicReadOnlyContract'
    report('camera_contract','OfflineM7ARecordedCalibrationAndSyntheticBoxTests',dict(**c,
        actual_camera_images=6,new_CARLA_execution=False,scope='Minimal contract only; full CARLA inference adapter and domain quality not run',
        yaw_scope='local ego yaw; arbitrary pitched/rolled3D Actor orientation not certified by this flat synthetic box',
        RGB_channel_check='official NormalizeMultiviewImage sentinel test; to_rgb=False after RGB->BGR',
        camera_order_scope='HS order frozen and matrices paired; no claim of canonical nuScenes dataset serialization order'))
    report('dependency_matrix','ActualIsolatedInstallationBuildAndInference',dict(status='Passed_ControlledCompatibility',route_count=1,
        selected=CFG['single_route'],additional_runtime='einops0.4.1; full package list in environment lock',
        original_reference='Torch1.9/cu111/MMCV1.4 historical only, not attempted on4090',
        CUDA_toolchains={'host':'12.8 untouched','torch_runtime':'11.8','mmcv_build':'11.8/GCC9.3','local_mmdet3d_build':'11.8.89/GCC11.4 sm89'},
        extension_build_status='11/11 built; original PETR inference path tested',pip_check='No broken requirements found',
        patch_files=[asset(x) for x in sorted(LOG.glob('*.patch'))],
        extra_packaging_patch=asset(LOG/'ninja_wheel_metadata_patch.json'),
        limitations=['Controlled inference compatibility, not bitwise replication of historical Torch1.9','No training/backward validation','Not all unused legacy lidar operators runtime-tested'],
        official_sources=['https://github.com/megvii-research/PETR/tree/'+CFG['official_commit'],
            'https://pytorch.org/get-started/previous-versions/',CFG['single_route']['mmcv_wheel'],
            'https://github.com/open-mmlab/mmdetection3d/tree/f1107977dfd26155fc1f83779ee6535d2468f449'],
        no_global_CUDA_or_HS_environment_modifications=True))
    failures=[]
    for path in sorted(LOG.glob('*.json')):
        data=json.loads(path.read_text())
        if data.get('exit_code',0)!=0:failures.append(dict(attempt=path.stem,exit_code=data['exit_code'],evidence=asset(path),log=data.get('log')))
    for path in sorted((ROOT/'artifacts').glob('probe*/failure.json')):
        failures.append(dict(attempt=path.parent.name,failure=json.loads(path.read_text()),evidence=asset(path)))
    report('failure_ledger','ActualRetainedFailedAttemptsAndResolution',dict(status='ResolvedWithinSingleVersionRoute',failures=failures,
        additional_failures=[
          {'issue':'Initial direct script invocation ModuleNotFoundError: tools','resolution':'Use python -m tools.m8a_p1_setup --step torch'},
          {'issue':'CUDA nvcc first full transfer truncated; SHA failed','evidence':asset(ROOT/'wheels/cuda_nvcc-linux-x86_64-11.8.89-archive.tar.xz.truncated'),'resolution':'Official checksum-verified bounded ranged download'},
          {'issue':'Official GitHub tree API403 rate limit','resolution':'No access bypass; same-tag official sparse git used'},
          {'issue':'GPU bwrap empty proc CUDA error304 / proc mountEPERM','resolution':'Restricted CPU numeric extraction; separate no-network GPU process'},
          {'issue':'cuBLAS header first build raced unfinished download','resolution':'Wait for verified distribution; build wrapper requires all headers'},
          {'issue':'Sparse patch loop assertion at undecorated pool.py','resolution':'conv-only patch audited against official source; pool unchanged'},
          {'issue':'einops proxy TLS EOF','resolution':'One official direct TLS-verified PyPI fetch'},
          {'issue':'ContainerPID did not match NVML hostPID','resolution':'Explicit temporal new-process attribution; exact PID namespace proof not claimed'}],
        downloaded_checkpoint_again=False,model_replacements=0,training_steps=0,all_old_failure_records_preserved=True))
    report('acceptance','EvidenceBasedP0GateDecision',dict(overall='FeasibilityPassed_QualityNotValidated',
        gates={'A_AssetAccess':'Passed_ReverifiedSameOfficialSHA','B_Dependency':'Passed_ControlledCompatibility',
               'C_Operator':'Passed_ActualDCNv2AndCompleteModelCUDA','D_ModelLoad':'Passed_Strict558Tensors',
               'E_Forward':'Passed_OriginalSpecification_SyntheticInputSmoke','F_Resource':'Passed_InferenceFits24GB',
               'G_Contract':'Passed_BasicReadOnlyCARLACameraContract'},
        separate_status={'engineering_inference_feasibility':'Passed','perception_quality':'NotValidated','CARLA_detection_benchmark':'NotRun',
            'training_feasibility':'NotRun','default_model_adoption':'NotApproved','pure_visual_driving_safety':'NotValidated',
            'Occupancy_safety_free':'NotValidated','shared_GPU_strict20Hz':'NotValidated','historical_Runtime':'Partial_Unchanged'},
        units_test_result='19 passed',runtime_contract_change=False,new_CARLA_closed_loop=False,training_steps=0,
        no_default_changes=True,checkpoint=asset(CFG['checkpoint']),
        next_step='Only separately approved read-only CARLA adapter/zero-shot quality pilot; no automatic training, model replacement or control',
        source_commit=execution['git_head'],run='probe13',contract_run='camera01',historical_AssetNoGo_preserved=True))
    write(LOG/'final_result_artifacts.json',dict(**provenance('CompleteResultArtifactManifest'),files=[asset(x) for x in sorted(out.iterdir())]))

if __name__=='__main__':main()
