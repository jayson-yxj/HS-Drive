"""P0 outcome and read-only CARLA field readiness, NOT a PETR dataset adapter."""
import json
from pathlib import Path
import numpy as np
from tools.m6a_common import write, provenance, asset
from tools.m5o_common import rows
from tools.m5f_common import input_path
from datasets.m5n_reference import synchronized, actor_reference


def main():
    review=[]
    # Existing development data only. No test or risk-calibration selection.
    for split in ('train','val'):
        entries=rows(split)
        cameras=[];frames_ok=0;refs=[];snap_ok=0
        for row in entries:
            inp=json.loads(input_path(row).read_text())
            frame=json.loads(Path(row['raw_frame']).read_text())
            frames_ok+=synchronized(frame,inp)
            cameras.append(len(inp['intrinsics'])==6 and len(inp['cam_to_ego'])==6)
            for a in frame['actors']:
                if a['is_ego']:continue
                snap_ok+=bool(a.get('actor_snapshot_present')) and a['actor_snapshot_frame_id']==frame['frame_id'] and abs(a['actor_snapshot_timestamp_s']-frame['timestamp_s'])<1e-5
                ref=actor_reference(a,inp['ego_pose'],inp['cam_to_ego'][0],inp['intrinsics'][0])
                refs.append(ref)
        review.append(dict(split=split,frames=len(entries),positions=len({r['group'] for r in entries}),
            synchronous_frames=int(frames_ok),six_camera_calibration_frames=sum(cameras),
            actor_configurations=len(refs),matching_actor_snapshot_frames=int(snap_ok),
            max_obb_vertex_closure_m=max(r['vertex_reconstruction_residual_m'] for r in refs),
            max_world_closure_m=max(r['world_closure_residual_m'] for r in refs),
            reference='simulator_actor_OBB_center_not_mesh_or_contact',
            index=asset(f'/root/gpufree-data/hs-drive-data/processed/m5o_{split}_pixels/index.json')))
    source=Path('/root/gpufree-data/hs-drive-experiments/m6a-petr/source')
    attempt=json.loads(Path('reports/m6a_petr_attempt_status.json').read_text())
    write('reports/m6a_perception_framework_feasibility.json',dict(
        **provenance('actual_server_asset_attempt_and_offline_official_source_review'),
        status='NoGo_P0_AssetAccess', decision='StopThisBoundedAttempt_ArchitectureQualityUnknown',
        attempt=attempt, official_repository='https://github.com/megvii-research/PETR',
        official_commit=attempt['official_source_commit'],
        source_license='Apache-2.0; retained LICENSE. No checkpoint redistribution; no separate verified checkpoint license found in inspected listing.',
        official_source_evidence=[asset(source/p) for p in ['LICENSE','README.md','install.md','requirements.txt','projects/configs/petr/petr_r50dcn_gridmask_c5.py','projects/mmdet3d_plugin/models/dense_heads/petr_head.py','projects/mmdet3d_plugin/datasets/nuscenes_dataset.py','projects/mmdet3d_plugin/core/bbox/util.py']],
        dependencies=dict(readme_python='3.6.8',attempt_python='3.8.20',readme_cuda='11.2',wheel_cuda='11.1',torch='1.9.0',mmcv_full='1.4.0',mmdet='2.24.1',mmseg='0.20.2',mmdet3d='0.17.1',numpy='<1.20',numba='0.48.0'),
        dependency_status='Python installed; matching cp38 MMCV binary downloaded. Torch download incomplete when asset-stop triggered. DCNv2/CUDA ABI/model import untested; do not infer incompatible.',
        official_io=dict(input='RGB image batch [B,6,3,512,1408], img_metas with calibrated lidar2img and augmented image shapes; test wrapper is augmentation-list',
            normalization='BGR caffe mean [103.530,116.280,123.675], std=1, to_rgb=False; must convert stored RGB consistently',
            model='ResNet50 DCNv2 C5, 900 queries, 6 decoder layers, original PETR single frame',
            classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
            output='bbox coder top 300 boxes, class labels/scores; gravity center internally, bottom-center z on exported LiDARInstance3DBoxes; dimensions and old yaw convention require pinned-version corner checks',
            ego_motion='No temporal ego motion needed by selected original single-frame head. Static sensor calibration to reference frame required; global pose used for dataset evaluation, not Actor GT features.',
            lidar_input=False,depth_input=False,
            positional_depth='PETR has 64 internal positional coordinate samples in this config; not the HS-Drive 48-bin Depth Head, which is untouched'),
        resource_status=dict(gpu='RTX4090 24GB',official_forward='NotRun',peak_memory_mib=None,six_camera_latency_ms=None,reason='No official checkpoint; no fabricated random-weight performance'),
        carla_field_readiness_review=review,
        adapter='NotRun_P0_gate; read-only field audit is not a conversion or a framework forward',
        adaptation_risks=['nuScenes camera order/optics/resolution and CARLA domain differ.',
            'CARLA left-handed world must convert to HS right-handed ego; optical projection K @ inverse(T_ego_from_opt).',
            'mmdet3d 0.17.1 old yaw convention differs from modern versions; no blind copying current adapters.',
            'Upright yaw-only boxes lose actor roll/pitch on slopes; quantify corner residual and keep Unsupported.',
            'M5-O references are six camera actor-views; PETR emits global objects; deduplicate actor IDs per frame, retain visibility union.',
            'Person/Cone official class support does not establish CARLA support; predictions and class confusion NotRun.',
            'Full physical-scene precision unavailable for unannotated static objects; unmatched is unknown burden, not automatic physical FP.'],
        formal_migration='NotApproved', training_steps=0))
    write('reports/m6a_3d_detection_benchmark.json',dict(
        **provenance('NotRun_conditioned_on_failed_P0'), status='NotRun',reason='Official pretrained PETR checkpoint not obtained; P0 incomplete',
        prediction_count=None,recall=None,precision=None,xy_error=None,dimension_error=None,yaw_error=None,
        latency_ms=None,peak_memory_mib=None,checkpoint_sha256=None,
        historical_baselines=[asset('reports/m5o_localization_benchmarks.json'),asset('reports/m5o_paired_method_analysis.json')],
        historical_numbers_are_not_petr_results=True,
        future_contract=dict(reference='Actor Box Center',frame='hs_ego',units='m,radians',
            association='one-to-one class-compatible global-object XY matching; report all output candidates and misses; predeclare radius/score thresholds before execution',
            localization='same visible GT scope, P50/P90/P95/Max XY, >5m severe errors, dimension and wrapped yaw separately',
            coverage='all eligible visible actors; unmatched outputs retained as unknown burden unless independent negative reference exists',
            baseline_scope='Geometry Prior/Direct same evaluable Vehicle/Person/Cone targets; camera-view vs global object accounting cannot be mixed',
            selection='M5-O Train/Val development; all opened tests historical only; any adaptation needs new independent final groups',
            inference_allowlist=['six_RGB','K','T','original_timestamp','image_preprocessing_metadata'],
            forbidden_features=['GT_actor_transform','GT_center','depth','instance_mask','GT_visibility_mask']),
        class_results={k:dict(status='NotRun',official_output_class_exists=True) for k in ['Vehicle','Person','Cone']},
        conclusion='No evidence to claim superiority, infer failure of PETR, or authorize migration'))
    print('field audit',review,flush=True)


if __name__=='__main__':main()
