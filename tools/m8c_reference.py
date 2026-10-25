"""Source/calibration distribution review, conditional B gate and actual-hook analysis."""
import json,inspect
from pathlib import Path
import numpy as np
from tools.m8c_common import *
from diagnostics.m8c_geometry import camera_in_reference,upright_boxes_to_ego
from datasets.m8b_petr_adapter import corners_hs,ORDER,A

def resume_write(path,value):
    p=Path(path)
    if p.exists():
        old=json.loads(p.read_text());new=dict(value)
        for k in ['sources','commit']:
            old.pop(k,None);new.pop(k,None)
        assert old==new, 'Existing evidence differs: '+str(path)
    else:write(path,value)

def main():
    source=RUNTIME/'sources/PETR';hook=json.loads((ROOT/'hook/run.json').read_text());index=json.loads((BASELINE/'evaluation_index.json').read_text());ledger=json.loads((BASELINE/'evaluation_ledger.json').read_text())['frames'];assoc=json.loads((ROOT/'association_ledger.json').read_text())
    camera=[];rigs=[]
    for x in index:
        p=json.loads(Path(x['packet']['path']).read_text());rigs.append([c['T_cam_to_ego'] for c in p['cameras']])
    p=json.loads(Path(index[0]['packet']['path']).read_text())
    for c in p['cameras']:
        K=np.array(c['K']);T=np.array(c['T_cam_to_ego']);fovx=np.degrees(np.arctan((640-K[0,2])/K[0,0])-np.arctan(-K[0,2]/K[0,0]));fovy=np.degrees(np.arctan((384-K[1,2])/K[1,1])-np.arctan(-K[1,2]/K[1,1]));cropv=np.degrees(np.arctan((384-K[1,2])/K[1,1])-np.arctan((332/2.2-K[1,2])/K[1,1]))
        camera.append(dict(name=c['name'],K=K.tolist(),T_E_C=T.tolist(),height_m=float(T[2,3]),heading_deg=float(np.degrees(np.arctan2(T[1,2],T[0,2]))),horizontal_fov_deg=float(fovx),vertical_fov_deg=float(fovy),cropped_vertical_fov_deg=float(cropv),augmented_focal_px=float(K[0,0]*2.2)))
    positions=np.array([c['T_cam_to_ego'] for c in p['cameras']])[:,:3,3];baselines=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
    paths=[source/'projects/mmdet3d_plugin/models/dense_heads/petr_head.py',source/'projects/mmdet3d_plugin/models/utils/positional_encoding.py',source/'projects/configs/petr/petr_r50dcn_gridmask_c5.py',source/'projects/mmdet3d_plugin/datasets/nuscenes_dataset.py',source/'tools/data_converter/nuscenes_converter.py',source/'projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py']
    official_order=['front','front_right','front_left','rear','rear_left','rear_right']
    comparison=dict(**provenance('OfflineOfficialSourceAndActualCalibrationReview'),PETRTrainingReferenceDistribution=dict(frame='nuScenes reference sample LIDAR_TOP; camera sensor2top composes calibration and ego motion/time',metadata_status='NotAvailable_FullCalibratedSensorTables',lidar_to_ego=None,camera_to_ego=None,camera_to_lidar=None,heights=None,intrinsics=None,FOV=None,baseline_matrix=None,empirical_target_reference_distribution=None,source_image_size=[900,1600],official_converter_order=official_order,model_position_range=[-61.2,-61.2,-10,61.2,61.2,10],model_pc_range=[-51.2,-51.2,-5,51.2,51.2,3],depth_samples=64,depth_start=1.,depth_end=float(hook['records'][0]['depth_samples'][-1]),public_tutorial_calibration=dict(sensor='RADAR_FRONT',translation=[3.412,0,0.5],sensor_token='47fcd48f71d75e0da5c8c1704a9bfe0a',not_lidar_height=True)),HSCARLACameraDistribution=dict(frame='HS ego actororigin',cameras=camera,baseline_matrix_m=baselines.tolist(),rig_max_variation=float(np.max(abs(np.array(rigs)-rigs[0]))),source_image_size=[384,640],order=ORDER,target_center_z=stats([g['box'][2] for f in ledger for g in f['gts']]),target_dims_lwh=[stats([g['box'][i] for f in ledger for g in f['gts']]) for i in range(3,6)]),source_evidence=[asset(v) for v in paths],camera_order_confounds=dict(status='ReferenceDistributionMismatchSupported_CameraIndexEncoding',geometry_pairing_correct=True,indices_mismatch=[i for i in range(6) if ORDER[i]!=official_order[i]],evidence='SinePositionalEncoding3D normalizes cumsum along camera dimension and embeds it; pairingK/T correctly does not erase learned camera-index semantic difference.',candidate_reordering='NotRun_NoUndeclaredAblation',checkpoint_actual_training_info_order='InferredFromFixedOfficialConverter_NotVerifiedFromOriginalTrainingPKL'),limitations=['nuScenes fullsensor calibration not locally present; officialtutorial contains RADAR_FRONT example not LIDAR_TOP','Officialdatasetrequiresaccount/terms; no accountassets available, no fabricatedheight or thirdpartydataset','Empiricaltrainingtarget Z and cameracolor distributions NotAvailable','No claim that cameraorder causes measured errors until onefactorcontrolledtest'])
    resume_write('reports/m8c_camera_distribution_comparison.json',comparison)
    records=[];top_z=[]
    for r in hook['records']:
        token=r['token'];base=json.loads((BASELINE/'forward'/(token+'.json')).read_text());assert np.array_equal(r['scores'],base['scores']) and np.array_equal(r['raw_boxes'],base['raw_old_framework_boxes'])
        with np.load(r['all_query_tensors']['path'],allow_pickle=False) as z:
            norm=z['normalized_xyz'][0];actual_logits=z['position_encoder_logits'];mask=z['geometric_mask'][0];depth=z['depth_m'];expected=((norm>1)|(norm<0)).reshape(6,44,16,-1).sum(-1)>32
            assert np.array_equal(expected.transpose(0,2,1),mask)
            clipped=np.clip(norm,0,1);expected_logits=np.log(np.maximum(clipped,1e-5)/np.maximum(1-clipped,1e-5)).transpose(0,3,4,2,1).reshape(6,192,16,44)
            residual=float(np.max(abs(expected_logits-actual_logits)));assert residual<2e-5
            strata={}
            for name,lo,hi in [('near',0,10),('mid',10,20),('far',20,40),('beyond40',40,80)]:
                n=norm[:,:,:, (depth>=lo)&(depth<hi),:];strata[name]=dict(depth_bins=int(((depth>=lo)&(depth<hi)).sum()),outside_xyz_fraction=float(((n<0)|(n>1)).any(-1).mean()))
        records.append(dict(token=token,split=next(x['row']['split'] for x in index if x['token']==token),frame_id=r['frame_id'],timestamp_s=r['timestamp_s'],camera=r['camera'],depth_range=strata,clamp_logits_max_residual=residual,hook_matches_M8B=True,raw_query_axes_per_layer=r['query_center_axes_per_layer'],raw_tensor=r['all_query_tensors']))
        top_z.extend(np.array(r['boxes'])[:,2].tolist())
    resume_write('reports/m8c_position_embedding_analysis.json',dict(**provenance('ActualRTX4090Tensors_OfflineAnalysis'),frames=len(records),records=records,position_depth_samples=hook['records'][0]['depth_samples'],topK_gravity_z=stats(top_z),hook_overhead=hook['hook_overhead_paired'],hook_on_off_bitwise_equal=True,baseline_output_bitwise_equal=True,computed_mask_consumed=False,computed_mask_definition='Count out-of-range coordinate components over D*3, threshold D*0.5; not number of invalid depth points. Returned geometry mask discarded; transformer uses padmask.',clamping='inverse_sigmoid clips outside normalized[0,1], then epsilon1e-5. Coordinates beyond bounds collapse to edge logits; not automatically ignored.',interpretation='23-27% out-of-range geometry samples are not evidence of domainmismatch without nuScenes calibration reference distribution. PrimarilyZ; geometry same across12rigs.',timing='Current uninstrumented ~503ms, hook~510ms; instrumentation delta paired4.6-10.1ms. Not comparable M8B53ms productionclaim: sysaudit/tracing harness, runtime conditions not controlled. No realtimeclaim.',source=hook['official_source']))
    # Synthetic contract validation only; hypothetical test transform is not a physical B candidate.
    maxproj=0.;maxcorner=0.;maxvelocity=0.
    for angle in [0,.37,-1.8]:
        T=np.eye(4);c,s=np.cos(angle),np.sin(angle);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[2,-1,1.3]
        for cc in p['cameras']:
            EC=np.array(cc['T_cam_to_ego']);RC=camera_in_reference(T,EC);point=np.array([12,3,1,1.]);newpoint=np.linalg.inv(T)@point
            maxproj=max(maxproj,float(np.max(abs(np.linalg.inv(RC)@newpoint-np.linalg.inv(EC)@point))))
        box=np.array([[10,-3,1,5,2,1.6,-.2,4,-2.]])
        converted=upright_boxes_to_ego(box,T);maxcorner=max(maxcorner,float(abs(corners_hs(converted[0])-(corners_hs(box[0])@T[:3,:3].T+T[:3,3])).max()))
        maxvelocity=max(maxvelocity,float(abs(upright_boxes_to_ego(converted,np.linalg.inv(T))-box).max()))
    resume_write('reports/m8c_reference_transform_contract.json',dict(**provenance('SyntheticGeometryContract_NotPhysicalBExperiment'),status='SyntheticContractPassed_PhysicalBNotRun',formula='T_R_C=inverse(T_E_R)@T_E_C; p_E=T_E_R@p_R; velocities rotation only; exactyawonly if no reference tilt',max_projection_optical_residual_m=maxproj,max_box_corner_residual_m=maxcorner,max_roundtrip_residual=maxvelocity,roll_pitch_reference='RejectedForUprightOutput_NoSilentYawApproximation',physical_T_E_R=None,reason='No traceable LIDAR_TOP calibration; RADAR_FRONTexample invalid. No pickedheight/yaw.',tests=asset('reports/m8c_logs/geometry_tests_v2.txt')))
    resume_write('reports/m8c_frozen_ablation.json',dict(**provenance('ConditionalGateDecision_NotRun'),A='FrozenM8B192frames/57600topK;12hookrerunsbitwiseequal',B='ReferenceVariantNotRun',B_runs=0,reason='Physicalbasis gate not satisfied: nuScenes LIDAR_TOP calibration and sourceegoalignment unavailable',camera_index_reordering='NotRun_SeparateSingleFactorProposal',threshold_selection=False,raw_A_reference=asset('reports/m8b_candidate_burden.json'),A_natural_reference=asset('reports/m8b_detection_metrics.json'),A_trusted_association=asset('reports/m8c_2d_3d_association_audit.json'),improvement_claim=False))
    # Correct a diagnostic mapping mistake with immutable original records preserved.
    old=json.loads(Path('reports/m8c_annotation_completeness.json').read_text());corrected=[]
    for r in old['records']:
        if r['coarse_class']!='unsupported_barrier':continue
        for v in r['views']:
            correct=sum(v['semantic_histogram'].get(str(i),0) for i in [4,5,28])/v['area']
            corrected.append(dict(token=r['token'],prediction_index=r['prediction_index'],camera=v['camera'],original_fraction=v['class_compatible_fraction'],correct_fraction=correct))
    resume_write('reports/m8c_annotation_semantic_correction.json',dict(**provenance('SDKEnumCrosscheck_Correction_NoStatusChange'),schema=asset('reports/m8c_semantic_schema.json'),supersedes_field='annotation_completeness.records[coarse_class=unsupported_barrier].views.class_compatible_fraction only',reason='InstalledCARLA0.9.15enum Walls4,Fences5,GuardRail28. Legacywebtable/assumed2,3notvalid. Rawsemanticcounts and classificationstatuses unchanged; no error-basedtuning.',corrected=corrected))
    print('reference analysis complete; physical B NotRun; camera order encoding discrepancy discovered')
if __name__=='__main__':main()
