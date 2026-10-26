"""Failure-aware offline closeout for the preserved, non-retried M8-I1 attempt."""
import ast
import json
import subprocess
from pathlib import Path
from tools.m8i1_common import ROOT, HISTORY, PROTOCOL, BASE, sha, asset, write, provenance, read_jsonl


def report(name,kind,**data):
    write('reports/m8i1_'+name+'.json',{**provenance(kind),**data})


def main():
    cfg=json.loads(PROTOCOL.read_text()); result=json.loads((ROOT/'session_result.json').read_text())
    supervisor=json.loads((ROOT/'supervisor_result.json').read_text()); events=read_jsonl(ROOT/'events.jsonl')
    cleanup=json.loads((ROOT/'cleanup.json').read_text()); post=json.loads((ROOT/'post_shutdown_process_probe.json').read_text())
    assert result['failure']=='RuntimeError: MapMismatch_NoReload'
    assert not result['vehicle_ids'] and not result['walker_ids'] and result['ticks']==0
    assert not list(ROOT.rglob('*.png')) and not list(ROOT.rglob('*sidecar.json'))
    executed=result['source_sha256']; executed_check={}
    for name,digest in executed.items():
        content=subprocess.check_output(['git','show',result['execution_commit']+':'+name])
        import hashlib
        executed_check[name]=hashlib.sha256(content).hexdigest()==digest
    assert all(executed_check.values())
    kind='ActualServerStartupFailure_NoTrafficRun'
    report('session_and_tick_audit',kind,status='PilotNoGo_MapIdentityMismatch',
        session_launch_attempts=1,server_restarts=0,owner_ticks=0,traffic_sim_seconds=0,
        expected_map='Town02',actual_map='NotRecorded_ExecutedFailurePathGap',world_episode='NotRecorded',
        bootstrap_world_ticks='NotMeasured_ServerAsyncStartupBeforeOwner',
        client_server_version_check='Both0.9.15_CodeCheckReached',
        source=asset(ROOT/'session_result.json'),commands=asset(ROOT/'execution_commands.json'),
        execution_commit=result['execution_commit'],executed_source_sha_verified=executed_check,
        no_tick_owner_created=True,no_TM_created=True,stop_condition_obeyed=True,
        failure=result['failure'],future_debugging='Only independently authorized startup/map identity repair; no second run in this stage')
    report('spawn_lifecycle',kind,status='NotRun_MapGateBeforeSpawn',spawn_attempts=0,vehicles=0,walkers=0,
        controllers=0,sensors=0,ego=None,navigation_requests=0,TM_registrations=0,
        events=events,original_physics='NotRead_NoVehicle',actual_spawn_order='NotRun',
        predeclared_source_order=cfg['spawn_order'],all_actual_spawn_calls_recorded=True)
    report('traffic_actor_snapshots',kind,status='NotRun',frames=[],source_actor_count=0,
        actual_frame_id='NotRecorded_NoOwnerTick',timestamp_s=None,source_box='NotAvailable_NoActor')
    report('sensor_sync',kind,status='NotRun_NoSensors',gate='SensorSyncPartial',
        required_keyframes=12,complete_keyframes=0,required_payloads=288,actual_payloads=0,
        sensor_tick_s=.1,world_dt_s=.05,image_size=[384,640],hfov=100,
        planned_period_offsets=cfg['save_offsets_sensor_periods'],actual_sensor_phase=None,
        actual_frame_schedule='NotInstantiated_NoSensorPhase',missing_frames='NotRun_NotSensorDrop',
        nearest_fill_used=False,synthetic_tests='reports/m8i1_logs/final_contract_and_history_tests.txt')
    report('gt_sidecar_contract','OfflineImplementedSchema_NotRuntimeValidated',status='NotRun_NoSidecars',
        directory_names=['RGBInferencePacket','EvaluatorSidecar'],
        RGB_allowlist=['RGB','K','T_cam_to_ego','frame_id','timestamp_s','camera_order','IdentityImageTransform','calibration_sha256','source'],
        evaluator_only=['ActorSnapshot','SourceOBB','Depth','Instance','Semantic','GroundRegion','EnvironmentInventory','TrafficLifecycle'],
        existing_source_contracts=['datasets/schema.py','geometry/reference.py','datasets/m8e_reference.py'],
        coordinate='CARLA world x forward y right z up; HS ego x forward y left z up; optical x right y down z forward',
        box='Actor transform composed with local BoundingBox transform, dimensions=2*extent; not mesh',
        timestamp_policy='original image/snapshot equality within1e-5s; no synthesized time',
        limitations=['Dynamic OBB/render closure untested','No source identity association executed',
          'Actual sidecar filesystem privilege separation not deployed; synthetic OS probe is separate',
          'Inference whitelist is an interface boundary, not a sandbox for arbitrary privileged code'])
    report('render_identity',kind,status='NotRun_NoRGBOrInstances',visible_actors=0,render_linked_objects=0,
        geometry_evaluable_objects=0,unique_Actor_renderID_equality_assumed=False,
        planned_rule=cfg['identity'],actual_association_rows=[],not_counted_as_detection_failures=True)
    report('vehicle_walker_20_40m',kind,status='SourceSupportNoGo_NotObserved',
        vehicle=dict(spawned=0,entered_range=0,visible=0,render_linked=0,geometry_evaluable=0,single_view=0,multiple_views=0),
        walker=dict(spawned=0,entered_range=0,visible=0,render_linked=0,geometry_evaluable=0,single_view=0,multiple_views=0),
        minimum='AtLeastOneVehicleAndOneWalkerAt20To40mWithVisibleInstanceAndSameFrameSource',
        reason='Map gate stopped execution before traffic; not evidence TM lacks opportunities',
        independent_location_groups=0,not_run_is_not_model_false_negative=True)
    report('background_unknown_audit',kind,status='NotRun',background_status='UnknownBackground',
        environment_inventory='NotQueried_MapGate',unlinked_pixel_count=None,
        static_identity='NotAvailable',occupancy_free_exported=False,missing_not_zero_negative=True)
    report('natural_scene_review',kind,status='NaturalScenePartial_NotRun',contact='ExactPhysicalContactNotCertified',
        actual_settling='NotRun',actual_movement='NotRun',ground_depth_semantic_support='NotRun',
        region_evidence='NotRun',static_competition='NotRun',visual_review='NotRun_NoImages',
        verified_float_or_penetration=None,legacy_15_ray_results_unchanged=True,
        stopped_reason=result['failure'],map_launch_root_cause='Unresolved_ActualNameNotRecorded')
    index=json.loads(HISTORY.read_text())
    from tools.m8g_history import preflight
    import numpy as np
    old=json.loads(Path('reports/m8f_history_recovery_supplement.json').read_text())['checks']
    req=json.loads(Path('reports/m8f_ray_hit_and_contact_audit.json').read_text())['requests']
    regs=[]
    for check in old:
        r=next(r for r in req if r['group']==check['group'] and r['key']==check['target'])
        poly=np.asarray(r['preflight']['present']['requested_obb']['corners_world_m'])[[0,4,6,2],:2]
        raw=dict(index,records=[x for x in index['records'] if x['index_origin']=='historical_raw_frame_supplement.json'])
        out=preflight(poly,'Town02',raw)
        delta=abs(out['actor']['distance_m']-check['nearest_raw_actor']['distance_m'])
        assert delta<1e-6
        regs.append(dict(group=check['group'],target=check['target'],distance_m=out['actor']['distance_m'],passed=True))
    report('historical_overlap','OfflineHistoricalIndexRegression_NoNewSpatialData',status='HistoricalCoverageIncomplete',
        index=asset(HISTORY),records=len(index['records']),source_count=len(index['historical_sources']),
        near_footprint_regressions=regs,new_ego_positions=0,new_target_footprints=0,
        actual_overlap='NotRun_NoNewPose',new_Validation=0,new_Calibration=0,new_Final=0,
        independent_groups=0,all_new_assets_use='DevelopmentFailureEvidenceOnly')
    gpu=read_jsonl(ROOT/'resource_samples.jsonl')
    report('resource_and_cleanup','ActualServerAndOSProcessEvidence',status='BudgetRespected_CleanupConfirmedByLaterOSProbe',
        wall_elapsed_s=supervisor['elapsed_s'],max_wall_s=1800,owner_ticks=0,max_ticks=700,
        final_external_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),max_bytes=1073741824,
        gpu_samples=gpu,gpu_peak='NotMeasured_StartupSamplesNotFullPeak',
        actor_cleanup=cleanup,immediate_supervisor_probe=supervisor['server'],later_OS_probe=post,
        world_settings_modified=False,weather_modified=False,TM_settings_modified=False,physics_modified=False,
        server_shutdown=True,unresolved_cleanup_processes=post['matching_processes'],
        abnormal_process_exit='SupervisorReturnCode0IsNotPilotSuccess; inspect session_result.failure')
    runtime_files=[Path(p) for p in executed if '/tests/' not in '/'+p]
    imports=[]
    for p in runtime_files:
        if p.suffix!='.py':continue
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
            elif isinstance(n,ast.ImportFrom):imports.append(n.module or '')
    report('training_readiness','OfflineSourceAndExecutionAudit',status='TrainingReadinessNoGo',
        model_forwards_actual=0,training_steps_actual=0,new_training_samples=0,
        model_or_checkpoint_imports=[i for i in imports if i.startswith(('torch','models','mmdet'))],
        proof_scope='Reviewed command graph, source hashes, imports, and execution stopped before actor/sensor code. Not a malicious-code sandbox.',
        RGB_visual_control=False,default_system='Unchanged',old_M8H1_NoGo='Unchanged',
        PETR_quality_and_online_realtime='NoGo_Unchanged',Occupancy_and_visual_safety='NotValidated_Unchanged')
    report('acceptance','ActualAttemptAndOfflineVerificationDecision',status='PilotNoGo',
        reason='MapMismatch_NoReload_BeforeTraffic',session_attempts=1,retries=0,
        gates=dict(A_TrafficGenerated='NotRun_NoActors',B_SensorSync='NotRun_0of12',
                   C_ActorIdentity='NotRun',D_GeometryEvaluable='NotRun',
                   E_Vehicle20To40mSupport='NotRun_0',F_Walker20To40mSupport='NotRun_0',
                   G_NaturalSceneReview='NaturalScenePartial_NotRun',H_ResourceAndCleanup='Confirmed_OSProbe'),
        synthetic_interface_tests='9 passed',historical_geometry_sync_tests='17 passed',
        historical_M8H1_evidence_tests='12 passed',actual_file_IO_probe='Passed_SyntheticFixturesOnly',
        real_packet_isolation='NotRun',actual_successful_science_keyframes=0,
        training_readiness='TrainingReadinessNoGo',real_scene_source_feasibility='Unanswered_NotTested',
        executed_code_commit=result['execution_commit'],relaunch_authorized=False,
        no_main_merge=True,no_default_changes=True)
    print('Wrote failure-aware reports; five historical footprint regressions preserved')


if __name__=='__main__':main()
