"""Read-only aggregate reporting; no model, simulator or raw public payloads."""
import argparse
from collections import Counter,defaultdict,deque
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.m8j13_common import write,quantiles,assets,digest
from simulator.m8j13_shadow_contract import require_completion_only


def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def load(p):return json.loads(p.read_text())
def counts(items):return dict(Counter(items))
def stats(v):return dict(n=len(v),min=float(min(v)),**quantiles(v)) if v else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,default=Path('/root/gpufree-data/hs-drive-private/m8j13-input-pipeline-20261101'));p.add_argument('--output',type=Path,default=Path('reports/m8j13_sources/input_pipeline'));a=p.parse_args()
    b=a.private_root;r=b/'live';out=a.output;out.mkdir(exist_ok=True,parents=True)
    old=Path('/root/gpufree-data/hs-drive-private/m8j13-parked-shadow-20261101/live')
    paths=sorted((r/'diagnostics').glob('*.json'));d=[load(p) for p in paths];cs=lines(r/'candidates.jsonl');wt=lines(r/'worker_timing.jsonl');cr=lines(r/'completion_received.jsonl')
    ss=lines(r/'snapshots.jsonl');snaps={x['frame']:x for x in ss};ot=lines(r/'owner_timing.jsonl');fb=lines(r/'parked_feedback.jsonl');events=lines(r/'watchdog_events.jsonl')
    led=lines(b/'forward_budget.jsonl');result=load(r/'result.json');supervisor=load(r/'supervisor.json')
    assert len(led)==40 and sum(x['kind']=='live_pipeline' for x in led)==len(d)==32
    assert result['charged_ticks']<=800 and len(load(b/'budget.json')['runs'])==1
    assert len(cs)==result['reached_candidates']==37 and len({x['candidate_index'] for x in cs})==len(cs)
    assert len(cr)==33 and len({x['request'] for x in cr})==33
    assert result['model']['fixed_state_unchanged'] and result['model']['preprocessing_thread_stopped']
    assert supervisor['cleanup_complete'] and result['transport_cleanup']
    after_assets=assets();assert after_assets==load(b/'assets_before.json')
    if (b/'assets_after.json').exists():assert load(b/'assets_after.json')==after_assets
    else:write(b/'assets_after.json',after_assets)
    for x in cr:require_completion_only(x['response'])
    cams=defaultdict(list)
    for x in lines(r/'camera_metadata.jsonl'):cams[x['frame']].append(x)
    sensor_identity=None;last=None
    for path,x in zip(paths,d):
        row=load(r/'input_rows'/path.name);snap=snaps[row['frame_id']];group=cams[row['frame_id']]
        assert len(group)==6 and row['imu_frame']==row['frame_id']
        assert abs(row['imu_timestamp_s']-row['timestamp_s'])<1e-5 and row['timestamp_s']==snap['sim_s']
        assert row['ego_world_matrix']==snap['pose'] and row['speed_mps']==snap['speed'] and row['ego_state_source']=='Oracle Ego State'
        identity=sorted((y['run'],y['episode'],y['sensor_generation'],y['camera'],y['actor_id']) for y in group)
        if sensor_identity is None:sensor_identity=identity
        assert identity==sensor_identity
        assert all(y['frame']==row['frame_id'] and abs(y['timestamp_s']-row['timestamp_s'])<1e-5 for y in group)
        assert x['pre_forward_age']['reason'] is None and x['pre_forward_age']['wall_age_s']<=.300001
        assert x['pre_forward_age']['sim_age_s']<=.300001
        assert x['memory_before']['history_timestamp']==(None if last is None else [last])
        assert x['memory_after']['history_timestamp']==[row['timestamp_s']]
        assert not x['applied_to_vehicle'] and x['finite']
        for key in ('native_path','native_speeds','native_trajectory','hs_points'):assert np.isfinite(x[key]).all()
        last=row['timestamp_s']
    for x in fb:assert x['matched'] and x['control']['throttle']==0 and x['control']['brake']==1 and x['control']['steer']==0
    sources=lines(r/'parked_control_sources.jsonl');assert all(x['source']=='IndependentParkedBrake' and not x['model_control'] for x in sources)
    measure=[x for x in fb if x['phase']=='measurement'];start=snaps[measure[0]['frame']]['input_wall_s'];end=snaps[measure[-1]['frame']]['tick_end_wall_s'];duration=end-start
    assert all(x['speed_mps']==0 for x in measure)
    received={x['request']:x for x in cr};complete=[received[p.name] for p in paths]
    fresh_result=sum(x['result_age_reason'] is None for x in d);fresh_owner=sum(x['age_reason'] is None for x in complete)
    assert fresh_result==fresh_owner==0
    oldd=[load(p) for p in sorted((old/'diagnostics').glob('*.json'))]
    def transport(root,selected):
        candidates={x['request']:x for x in lines(root/'candidates.jsonl') if x['status']=='Submitted'}
        workers={x['request']:x for x in lines(root/'worker_timing.jsonl')}
        c=[candidates[n] for n in selected]
        return dict(tick_to_six_cameras_ms=stats([(max(x['camera_arrival_wall_s'].values())-x['input_wall_s'])*1000 for x in c]),
            six_cameras_to_client_ms=stats([(x['decision_wall_s']-max(x['camera_arrival_wall_s'].values()))*1000 for x in c]),
            group_checks_ms=stats([x['group_checks_ms'] for x in c]),image_copy_or_mmap_publish_ms=stats([x['image_copy_ms'] for x in c]),
            critical_path_npz_write_ms=stats([x['ipc_save_ms'] for x in c]),
            publish_to_worker_ms=stats([(workers[x['request']]['start_wall_s']-x['submitted_wall_s'])*1000 for x in c]),
            tick_to_worker_ms=stats([(workers[x['request']]['start_wall_s']-x['input_wall_s'])*1000 for x in c]))
    timings=dict(first_frame_ms=d[0]['timings_ms'],subsequent31_ms={k:stats([x['timings_ms'][k] for x in d[1:]]) for k in d[0]['timings_ms']},
        all32_endpoint1_ms=stats([x['timings_ms']['end_to_end_result'] for x in d]),
        first_owner_received_ms=complete[0]['wall_age_s']*1000,subsequent31_endpoint2_owner_received_ms=stats([x['wall_age_s']*1000 for x in complete[1:]]),
        all32_endpoint2_owner_received_ms=stats([x['wall_age_s']*1000 for x in complete]),
        result_sim_age_s=stats([x['completion_age']['sim_age_s'] for x in d]),owner_received_sim_age_s=stats([x['sim_age_s'] for x in complete]),
        preforward_wall_age_ms=stats([x['pre_forward_age']['wall_age_s']*1000 for x in d]),
        preprocessing_subsequent31_ms={k:stats([x['preprocessing'][k] for x in d[1:]]) for k in d[0]['preprocessing']},
        old_transport_subsequent15_ms=transport(old,[p.name for p in sorted((old/'diagnostics').glob('*.json'))][1:]),
        new_transport_subsequent31_ms=transport(r,[p.name for p in paths][1:]),
        world_tick_and_snapshot_measurement_ms=stats([x['tick_wall_ms'] for x in ot if x['phase']=='measurement']),
        owner_queue_drain_and_metadata_ms=stats([x['drain_and_metadata_write_ms'] for x in ot if x['phase']=='measurement']),
        camera_callback_copy_ms=stats([(x['copied_monotonic_s']-x['receive_monotonic_s'])*1000 for group in cams.values() for x in group]),
        post_publish_archive_and_logging_ms=stats([x['post_publish_persistence_ms'] for x in wt if x['status']=='ShadowComplete']),
        owner_poll_delay_ms=stats([(v['wall_s']-x['published_wall_s'])*1000 for x,v in zip(d,complete)]),
        measurement_wall_s=duration,measurement_sim_s=len(measure)*.05,physics_sim_hz=20,
        actual_wall_tick_hz=len(measure)/duration,candidate_wall_hz=len(cs)/duration,inference_wall_hz=len(d)/duration,
        fresh_owner_wall_hz=fresh_owner/duration,completion_interval_wall_s=stats(np.diff([x['wall_s'] for x in complete]).tolist()),
        completion_span_hz=(len(d)-1)/(complete[-1]['wall_s']-complete[0]['wall_s']),
        fresh_result_intervals_s=None,longest_no_fresh_result_wall_s=duration,no_fresh_window='Entire measurement window; initialization excluded',
        gpu_peak_allocated_bytes=max(x['gpu_allocated'] for x in d),gpu_peak_reserved_bytes=max(x['gpu_reserved'] for x in d),
        gpu_scope='PyTorch model process allocator only; not CARLA or whole GPU utilization',
        cpu_prepare_thread_ms=stats([x['timings_ms']['prepare_thread_cpu'] for x in d]),
        max_submitted_leases=2,leases_before_submission=counts(str(x['queue_occupied']) for x in cs if x['status']=='Submitted'),
        workload='One CPU preparation, serial CUDA/model/PID, synchronous post-publication archive. CPU quota14 cores, model affinity11-12/Torch2/OpenCV2/interop64 unchanged.',
        interpretation='Different live sequences and gaps, cross-run comparison, not a strict online single-factor experiment. Percentiles describe this small sample and cannot be added. No causal GPU/RPC/server attribution from wall-time data.')
    worker_by_name={x['request']:x for x in wt}
    timings['next_input_read_starts_overlapping_prior_inference']=sum(d[i-1]['forward_start_wall_s']<worker_by_name[paths[i].name]['start_wall_s']<d[i-1]['forward_start_wall_s']+d[i-1]['timings_ms']['inference_decode_with_selection_audit']/1000 for i in range(1,len(d)))
    write(out/'timing.json',timings)
    write(out/'budget_and_freshness.json',dict(starts=1,active_ticks=result['charged_ticks'],explicit_ticks=result['explicit_ticks'],settings_charge=30,tick_cap=800,
        offline_forward_attempts=8,offline_success=8,live_forward_attempts=32,live_success=32,live_failure=0,total_forward_attempts=40,
        fixed_candidates=120,reached_candidates=len(cs),not_reached_candidates=120-len(cs),owner_outcomes=counts(x['status']+':'+str(x.get('reason')) for x in cs),
        worker_outcomes=counts(x['status'] for x in wt),input_rejections=3,capacity_skips=1,reached_budget_end_not_submitted=1,
        fresh_at_HS_PID=fresh_result,fresh_at_checked_owner=fresh_owner,fresh_per_forward=fresh_owner/len(d),fresh_per_reached_candidate=fresh_owner/len(cs),
        results_wall_expired=sum(x['completion_age']['wall_age_s']>.300001 for x in d),results_sim_expired=sum(x['completion_age']['sim_age_s']>.300001 for x in d),
        received_wall_expired=sum(x['wall_age_s']>.300001 for x in complete),received_sim_expired=sum(x['sim_age_s']>.300001 for x in complete),
        result_reason=counts(x['result_age_reason'] for x in d),received_reason=counts(x['age_reason'] for x in complete),
        no_refreshed_observation_clock=True,input_wait_wall_s=.25,sim_age_limit_s=.3,wall_age_limit_s=.3,model_controls=0,training=0,
        old_budgets_and_failures_unchanged=True))
    checks=lines(r/'input_checks.jsonl');init=[x['speed_mps'] for x in fb if x['phase']=='initialization'];exits=[x['speed_mps'] for x in fb if x['phase']=='exit_brake_observation']
    first_watchdog_tick=next(x for x in ss if x['input_wall_s']<=events[0]['wall_s']<=x['tick_end_wall_s'])
    write(out/'isolation_temporal.json',dict(first_watchdog_event_inside_tick_ms=dict(from_tick_start=(events[0]['wall_s']-first_watchdog_tick['input_wall_s'])*1000,tick_and_snapshot_total=(first_watchdog_tick['tick_end_wall_s']-first_watchdog_tick['input_wall_s'])*1000),actual_forward_inputs_verified=len(d),same_generation_sensor_instances=True,same_frame_snapshot_ego_imu=True,
        max_extrinsic_error=max(x['extrinsic_max_error'] for x in checks),max_navigation_error_m=max(x['navigation_error'] for x in checks),
        navigation=load(r/'navigation_wiring.json'),memory_initial_empty=True,memory_timestamps_previous_actual_input=True,
        actual_input_gap_counts_s=counts(str(round(x['timestamp_gap_s'],3)) for x in d[1:]),fixed_state_unchanged=True,
        initialization_peak_speed_mps=max(init),admission_last21_speed_max_mps=max(init[-21:]),measurement_ticks=len(measure),measurement_speed_max_mps=max(x['speed_mps'] for x in measure),
        exit_observation_ticks=len(exits),exit_terminal_speed_mps=exits[-1],owner_brake_submissions=len(sources),all_actual_control_feedback_matches=True,
        watchdog_submissions=len(events),watchdog_reasons=counts(x['reason'] for x in events),watchdog_tick_calls=load(r/'watchdog_ready.json')['tick_calls'],
        watchdog_scope='A command-expiry latch only reinforces preexisting independent full brake; not a moving-fault stop test',
        collisions=len(lines(r/'collisions.jsonl')),model_network_isolated=True,completion_IPC_controls=False,
        cleanup_complete=supervisor['cleanup_complete'],mmap_removed=result['transport_cleanup'],owned_actors_remaining=result['cleanup']['remaining'],restored_settings=result['cleanup']['settings'],
        raw_tail='Full callback logs retained privately, including noncandidate and drain tail; no new500-tick sensor acceptance claimed'))
    # Reproduce longitudinal PID arithmetic from already saved outputs, no forward.
    pid={}
    for name,records in [('previous16',oldd),('current32',d)]:
        window=deque([0.]*10,maxlen=10);raw=[];errors=[]
        for x in records:
            target=x['native_speeds'][0];speed=x['pid_metadata']['speed'];assert target==x['pid_metadata']['desired_speed']
            delta=float(np.clip(target-speed,0,1));prior=window[-1];window.append(delta)
            u=5*delta+.5*np.mean(window)+(delta-prior);brake=target<.2 or speed/target>1.1
            throttle=0. if brake else float(np.clip(u,0,1));assert throttle==x['raw_pid']['throttle'];raw.append(float(u));errors.append(target-speed)
        pid[name]=dict(samples=len(records),desired_speed_min_mps=min(x['native_speeds'][0] for x in records),desired_speed_max_mps=max(x['native_speeds'][0] for x in records),
            error_min_mps=min(errors),error_max_mps=max(errors),reconstructed_unclipped_throttle_min=min(raw),reconstructed_unclipped_throttle_max=max(raw),
            full_throttle=sum(x['raw_pid']['throttle']==1 for x in records),max_future_speed_mps=max(max(x['native_speeds']) for x in records),
            planned_future_above_5mps=sum(max(x['native_speeds'])>5 for x in records),trajectory_check_failures=sum(x['trajectory_check_reason'] is not None for x in records))
    write(out/'pid_analysis.json',dict(target='First native lon_reg_final entry, not last or mean; window-average planned speed sample',
        speed_feedback='Signed forward speed m/s;0 for accepted parked inputs',gains=dict(Kp=5,Ki=.5,Kd=1,window=10,error_clip=[0,1],throttle_clip=[0,1]),
        first_sample_explanation='Target0.33375287 minus speed0; proportional term alone1.668764>1, final PID saturated. All saved longitudinal outputs reproduced without new forward.',
        records=pid,interpretation='Acceleration recommendation in parked empty-road scene and saturation are observed; neither proves moving controller failure. Future native speed exceeds5m/s driving limit. No PID/trajectory change, no driving admission. Different live memory gaps prevent attributing planning differences to algorithm changes.'))
    write(out/'acceptance.json',dict(J13_A='Partial',InputEquivalence='Passed_24ExactTensorPairs_8ModelForwardsExact',
        TransportAndPreprocessing='BoundedTransportPassed_PreprocessingComputeEssentiallyUnchanged',PipelineScheduling='BoundedOverlapRecorded_1QueuedInputExpired_1CapacitySkip',
        FrozenInference='Passed_32LiveAttempts32Success0Failure',EndToEndFreshness='NotPassed_0of32Fresh_0of37ReachedCandidates',
        ParkedIsolation='Passed_ForThisParkedMeasurement',RemainingDrivingGates='FourPlannedMovingFaultsNotRun_FreshnessNotPassed_ModelDrivingNotRun',
        realtime_pass=False,SensorSynchronizationPassed=False,S0='NotRun',S1='NotRun',S2='NotRun',model_controls=0,training=0,
        next_priority='BoundedCaptureToConsumerSchedulingAnalysis; moving-fault coverage remains separately mandatory before driving',
        reason='Transport cost reduced and throughput increased; every result still expired. Camera arrival + unchanged JPEG/transforms + inference dominate, owner polling adds delay. No evidence here justifies changing model/research route or PID. No further run authorized in this round.'))
    write(out/'regression_and_provenance.json',dict(regression_passed=83,regression_command='/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv-gpu/bin/python -m pytest tests/m8j13 tests/m8j12 --confcutdir=tests -q',
        run_commit=load(r/'request.json')['commit'],config_sha256=load(r/'request.json')['config_sha256'],runtime=load(b/'runtime_identity.json'),
        assets_before_after_equal=True,asset_source_commit=load(b/'assets_before.json')['official_commit'],asset_sha256={x['file']:x['sha256'] for x in load(b/'assets_before.json')['files']},
        actual_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),main_unchanged='c1644b0587da31396a8db8cb81779ed8b7f8acd8',
        public_content='Code/config/tests/aggregates only; original reports and private archive untouched; no original sensor identifiers/images/per-frame model outputs in this export'))
    print(json.dumps(dict(success=32,fresh=0,endpoint1_subsequent=timings['subsequent31_ms']['end_to_end_result'],endpoint2_subsequent=timings['subsequent31_endpoint2_owner_received_ms'],throughput=timings['inference_wall_hz']),indent=2))

if __name__=='__main__':main()
