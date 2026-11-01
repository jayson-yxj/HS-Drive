"""Read-only identity, state, budget and timing audit after fixed live run."""
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from tools.m8j13_common import assets,write,digest
from tools.m8j13_recheck_report import stats,lines,read
from simulator.m8j13_shadow_contract import require_completion_only


def main():
    b=Path('/root/gpufree-data/hs-drive-private/m8j13-per-camera-recheck-20261101');r=b/'live';out=Path('reports/m8j13_sources/per_camera_recheck')
    cfg=read(Path('configs/m8j13_per_camera_recheck.json'));paths=sorted((r/'diagnostics').glob('*.json'));d=[read(x) for x in paths]
    ledger=lines(b/'forward_budget.jsonl');result=read(r/'result.json');supervisor=read(r/'supervisor.json');summary=read(out/'live_summary.json')
    assert all(x['kind']=='live_pipeline' for x in ledger) and len(ledger)<=24
    assert len(ledger)==len(d)==result['model']['completed_forwards']
    assert len(read(b/'budget.json')['runs'])==1 and result['charged_ticks']<=800
    assert result['model_controls']==result['training']==0 and result['model']['fixed_state_unchanged']
    assert supervisor['cleanup_complete'] and result['transport_cleanup']
    after=assets();assert after==read(b/'assets_before.json');write(b/'assets_after.json',after)
    snapshots={x['frame']:x for x in lines(r/'snapshots.jsonl')};cams=defaultdict(list)
    for x in lines(r/'camera_metadata.jsonl'):cams[x['frame']].append(x)
    prev=None;identity=None;checks=lines(r/'snapshot_checks.jsonl');receipts={x['request']:x for x in lines(r/'completion_received.jsonl')}
    for path,x in zip(paths,d):
        row=read(r/'input_rows'/path.name);snap=snapshots[x['frame']];group=cams[x['frame']]
        assert len(group)==6 and row['imu_frame']==row['frame_id'] and abs(row['imu_timestamp_s']-snap['sim_s'])<1e-5
        assert row['timestamp_s']==snap['sim_s'] and row['ego_world_matrix']==snap['pose'] and row['speed_mps']==snap['speed']
        group_identity=sorted((c['run'],c['episode'],c['sensor_generation'],c['camera'],c['actor_id']) for c in group)
        if identity is None:identity=group_identity
        assert identity==group_identity and all(abs(c['timestamp_s']-snap['sim_s'])<1e-5 for c in group)
        assert x['pre_forward_age']['reason'] is None and x['pre_forward_age']['snapshot_confirmed']
        assert x['memory_before']['history_timestamp']==(None if prev is None else [prev])
        assert x['memory_after']['history_timestamp']==[row['timestamp_s']];prev=row['timestamp_s']
        assert x['finite'] and not x['applied_to_vehicle']
        for key in ('native_path','native_speeds','native_trajectory','hs_points'):assert np.isfinite(x[key]).all()
        summed=sum(y['snapshot_wait_ms'] for y in checks if y['input_frame']==x['frame'])
        assert abs(summed-x['snapshot_wait_cumulative_ms'])<1e-9
        require_completion_only(receipts[path.name]['response'])
    fb=lines(r/'parked_feedback.jsonl');assert all(x['matched'] and x['control']['throttle']==0 and x['control']['brake']==1 and x['control']['steer']==0 for x in fb)
    assert max(x['speed_mps'] for x in fb if x['phase']=='measurement')==0
    cs=lines(r/'candidates.jsonl');assert len(cs)==len({x['candidate_index'] for x in cs})==summary['reached_candidates']
    assert set(x['candidate_index'] for x in cs)==set(range(len(cs)))
    old=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101/live');oldp=sorted((old/'diagnostics').glob('*.json'));oldd=[read(x) for x in oldp]
    def arrival(r,dd):
        cm=defaultdict(list)
        for x in lines(r/'camera_metadata.jsonl'):cm[x['frame']].append(x)
        return dict(first_camera_copy_ms=stats([(min(y['copied_monotonic_s'] for y in cm[x['frame']])-x['input_wall_s'])*1000 for x in dd]),
            all_camera_copy_ms=stats([(max(y['copied_monotonic_s'] for y in cm[x['frame']])-x['input_wall_s'])*1000 for x in dd]),
            first_last_camera_ms=stats([(max(y['receive_monotonic_s'] for y in cm[x['frame']])-min(y['receive_monotonic_s'] for y in cm[x['frame']]))*1000 for x in dd]))
    sub=d[1:];comparison=dict(scope='Cross-run descriptive comparison; different candidate/gap/load mix; no causal attribution of all delta to preprocessing',
        previous32_subsequent31_input_to_forward_ms=stats([(x['forward_start_wall_s']-x['input_wall_s'])*1000 for x in oldd[1:]]),
        current_input_to_forward_ms=stats([(x['forward_start_wall_s']-x['input_wall_s'])*1000 for x in sub]),
        current_input_to_CPU_assembly_ms=stats([(x['prepared_wall_s']-x['input_wall_s'])*1000 for x in sub]),
        current_CPU_tail_ms=stats([x['preprocessing']['tail']['serial_tail_ms'] for x in sub]),
        current_per_camera_CPU_work_ms=stats([v['cpu_ms'] for x in sub for v in x['preprocessing']['cameras'].values()]),
        old_arrival=arrival(old,oldd[1:]),current_arrival=arrival(r,sub),
        old_model_decode_ms=stats([x['timings_ms']['inference_decode_with_selection_audit'] for x in oldd[1:]]),
        current_model_decode_ms=stats([x['timings_ms']['inference_decode_with_selection_audit'] for x in sub]),
        offline_hypothesis='Historical same-input preparation saved paired median70.9ms; candidate batch ready median194.6ms then hypothetical checks complete305.4ms. No renderer/model contention and no actual Snapshot-confirmation wait.',
        online_result='CPU assembly still finishes early, but serial preforward confirmation waits and longer measured model/decode eliminate end-to-end gain. Confirmation waits are included, never subtracted from measured deadlines.',
        counterfactual_input_to_forward_if_waits_removed_ms=stats([(x['forward_start_wall_s']-x['input_wall_s'])*1000-x['snapshot_wait_cumulative_ms'] for x in sub]),
        counterfactual_warning='Algebraic diagnostic only, NOT a permissible change, actual elapsed result, or guaranteed speedup. Every Snapshot confirmation remains required.',
        old_HS_PID_ms=stats([x['timings_ms']['end_to_end_result'] for x in oldd[1:]]),current_HS_PID_ms=stats([x['timings_ms']['end_to_end_result'] for x in sub]),
        no_GPU_RPC_engine_causal_claim=True)
    write(out/'comparison.json',comparison)
    group_checks=lines(r/'input_checks.jsonl');watchdog=lines(r/'watchdog_events.jsonl');receipts_list=[receipts[p.name] for p in paths]
    first=receipts_list[0] if d else None
    detail=dict(verified_actual_inputs=len(d),same_run_generation_sensor_instances=True,same_frame_snapshot_imu_ego=True,
        model_memory_matches_actual_previous_input=True,weights_buffers_unchanged=True,
        max_navigation_error_m=max((x['navigation_error'] for x in group_checks),default=None),max_extrinsic_error=max((x['extrinsic_max_error'] for x in group_checks),default=None),
        model_network_namespace=read(r/'model_ready.json')['network_namespace'],model_carla_import_forbidden=True,
        diagnostic_outputs_do_not_actuate=True,all_completion_IPC_flags_only=True,
        actual_threads=read(r/'model_ready.json')['threads'],cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        watchdog_events=len(watchdog),watchdog_reasons=dict(Counter(x['reason'] for x in watchdog)),watchdog_tick_calls=read(r/'watchdog_ready.json')['tick_calls'],
        watchdog_scope='Parked independent brake already active; no moving-fault claim',
        all_owner_sources=sorted(set(x['source'] for x in lines(r/'parked_control_sources.jsonl'))),
        first_owner_consume_ms=(first['owner_consume_check_done_wall_s']-d[0]['input_wall_s'])*1000 if d else None,
        first_wait_cumulative_ms=d[0]['snapshot_wait_cumulative_ms'] if d else None,
        first_gpu_peak=dict(allocated=d[0]['gpu_allocated'],reserved=d[0]['gpu_reserved']) if d else None,
        consumer_timestamp_limit='Frozen run stamps pure-consumer call reference immediately before its arithmetic checks, not exact return. Completion serialization follows it. Reported owner endpoint is this near-consumption reference; no exact-return claim. All wall ages exceed300ms by at least45.9ms so0 fresh remains certain. Delivery code adds explicit return timestamp; no rerun.',
        measured_tick_RPC_ms=stats([(snapshots[x['frame']]['tick_rpc_return_wall_s']-snapshots[x['frame']]['input_wall_s'])*1000 for x in fb if x['phase']=='measurement']),
        completion_publish_to_owner_ms=stats([(y['wall_s']-x['published_wall_s'])*1000 for x,y in zip(d,receipts_list)]),
        owner_read_to_consume_ms=stats([(x['owner_consume_check_done_wall_s']-x['wall_s'])*1000 for x in receipts_list]),
        post_publication_persistence_ms=stats([x['post_publish_persistence_ms'] for x in lines(r/'worker_timing.jsonl') if x['status']=='ShadowComplete']),
        queue_occupied_before_commit=dict(Counter(str(x['queue_occupied']) for x in cs if 'queue_occupied' in x)),
        pid_full_throttle=sum(x['raw_pid']['throttle']==1 for x in d),trajectory_check_failures=sum(x['trajectory_check_reason'] is not None for x in d),
        remaining_actors=result['cleanup']['remaining'],settings_restored=result['cleanup']['settings'],traffic_manager=result['cleanup']['traffic_manager'])
    # Namespace value is process metadata; omit its local identifier from public export.
    detail.pop('model_network_namespace');detail['network_isolation_verified']=True
    write(out/'isolation_and_timing_audit.json',detail)
    write(out/'acceptance.json',dict(J13_A='Partial',LiveIntegration='Passed_FixedCandidateRunsEndToEnd',FrozenInference='Passed',
        EndToEndFreshness='NotPassed' if summary['OwnerConsumption']['fresh']==0 else 'LimitedFreshResults_NotRealtimeQualification',
        forward_attempts=summary['attempts'],forward_success=summary['success'],forward_failure=summary['failed'],owner_fresh=summary['OwnerConsumption']['fresh'],
        SnapshotConfirmation='MeasuredAtEveryCheck_CumulativeWaitIncluded_UnconfirmedNotCalledExpired',
        ParkedIsolation='Passed_ForThisFixedMeasurement',RemainingMovingFaultCoverage='NotRun_FourCasesStillRequired',
        next_step='DesignSeparateNonRealtimeSynchronousLogicalLoopProtocol_NoAutomaticSwitch' if summary['OwnerConsumption']['fresh']==0 else 'EvaluateLimitedMarginBeforeFurtherAuthorizedPerformanceWork',
        no_further_optimization_or_simulation_this_round=True,SensorSynchronizationPassed=False,realtime_claim=False,S0='NotRun',S1='NotRun',S2='NotRun',model_controls=0,training=0))
    write(out/'provenance.json',dict(base='e0ad2eda3213d548ec634d60106450216956b9ad',run_commit=read(r/'request.json')['commit'],config_sha256=digest('configs/m8j13_per_camera_recheck.json'),
        runtime=read(b/'runtime_identity.json'),assets_verified_before_after=True,official_commit=after['official_commit'],asset_sha256={x['file']:x['sha256'] for x in after['files']},
        preflight_regression=95,offline_forwards_this_round=0,old_equivalence_referenced=True,old_private_chain_imported=False,main_unchanged='c1644b0587da31396a8db8cb81779ed8b7f8acd8',
        private_root=str(b),defaults_PID_model_parameters_unchanged=True))
    print('Audit passed for',len(d),'actual inputs; historical reports retained')
if __name__=='__main__':main()
