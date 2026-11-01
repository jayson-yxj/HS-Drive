"""Read-only report for the bounded per-camera attempt, including failed run."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from tools.m8j13_camera_timeline import ROOT
from tools.m8j13_common import write,quantiles,assets,digest


def main():
    r=ROOT/'live';out=Path('reports/m8j13_sources/per_camera')
    def load(p):return json.loads(p.read_text())
    def lines(name):return [json.loads(x) for x in (r/name).read_text().splitlines()] if (r/name).exists() else []
    result=load(r/'result.json');supervisor=load(r/'supervisor.json');final=load(r/'model_final.json');before=load(ROOT/'assets_before.json');after=assets();assert before==after
    write(ROOT/'assets_after.json',after)
    assert final['completed_forwards']==0 and final['full_forward_attempts']==8 and final['fixed_state_unchanged']
    assert result['error']=="ValueError('RowCameraBindingMismatch')" and supervisor['cleanup_complete']
    feedback=lines('parked_feedback.jsonl');snaps=lines('snapshots.jsonl');byframe={x['frame']:x for x in snaps};schedule=load(r/'first_camera_schedule.json');measurement=[x for x in feedback if x['phase']=='measurement']
    reached=[schedule['first_frame']+4*i for i in range(120) if schedule['first_frame']+4*i<=max(x['frame'] for x in measurement)]
    assert len(reached)==1 and len(measurement)==3
    frame=reached[0];group=[x for x in lines('camera_metadata.jsonl') if x['frame']==frame];assert len(group)==6
    assert all(abs(x['timestamp_s']-byframe[frame]['sim_s'])<1e-5 for x in group)
    assert len(lines('input_checks.jsonl'))==1
    assert all(x['matched'] and x['control']['throttle']==0 and x['control']['brake']==1 for x in feedback)
    assert max(x['speed_mps'] for x in measurement)==0
    assert len(lines('collisions.jsonl'))==0
    # Preserve raw run; reconstruct missing failed-candidate disposition separately.
    write(ROOT/'failed_candidate_reconstruction.json',dict(frame=frame,first_candidate=schedule['first_frame'],status='RejectedAtCommit',
        reason='RowCameraBindingMismatch',metadata_group=group,snapshot=byframe[frame],source='Traceback + schedule + measured frames + input_checks; original candidates log absent because exception preceded append',
        raw_original_files_unchanged=True))
    rr=load(ROOT/'profile_result.json')['records'];sub=rr[3:];budget=load(ROOT/'budget.json')
    assert len(budget['runs'])==1 and result['charged_ticks']==119
    init=[x for x in feedback if x['phase']=='initialization'];exit_=[x for x in feedback if x['phase']=='exit_brake_observation']
    wall=byframe[measurement[-1]['frame']]['tick_end_wall_s']-byframe[measurement[0]['frame']]['input_wall_s']
    ticks_in_camera=[any(s['input_wall_s']<=c['receive_monotonic_s']<=s['tick_rpc_return_wall_s'] for s in snaps) for c in group]
    write(out/'failed_live_result.json',dict(status='Partial_LiveIntegrationFailureBeforeForward',category='B_OfflineLocalImprovement_OnlineFreshnessUnverified',
        explanation='Not a measured0/N inference freshness failure: no live forward. All old0/16 and0/32 results remain unchanged.',
        starts=1,start_cap=1,explicit_ticks=result['explicit_ticks'],settings_internal_charge=30,charged_ticks=119,tick_cap=800,
        offline_forwards=8,offline_success=8,live_forwards=0,live_success=0,live_failure_attempts=0,live_cap=24,total_forwards=8,total_cap=32,
        candidate_limit=120,reached_candidates=1,candidate_dispositions={'RejectedAtCommit':1},not_reached=119,submitted=0,
        rejected_reason='RowCameraBindingMismatch',candidate_log_gap='Exception occurred before append; one reached candidate reconstructed from frozen schedule/snapshots/input checks in separate private record. Old logs untouched. Delivery code logs commit errors before stopping.',
        fresh_HS_PID=0,fresh_required_checks=0,fresh_notification=0,fresh_owner_consumption=0,success_fraction=None,
        fresh_rate_hz=0,inference_rate_hz=0,frequency_scope='Three measured ticks only; no performance inference possible',
        measured_wall_s=wall,measured_sim_s=.15,longest_no_fresh_window_wall_s=wall,new_model_input_intervals=None,
        initialization_ticks=60,initialization_peak_speed_mps=max(x['speed_mps'] for x in init),admission_last21_max_speed_mps=max(x['speed_mps'] for x in init[-21:]),
        measurement_ticks=3,measurement_max_speed_mps=0,exit_ticks=len(exit_),exit_terminal_speed_mps=exit_[-1]['speed_mps'],
        model_controls=0,training=0,collisions=0,actual_brake_feedback_matches=True,watchdog_brake_submissions=len(lines('watchdog_events.jsonl')),
        same_frame_camera_group=True,callbacks_during_actual_tick_RPC=sum(ticks_in_camera),first_group_callbacks=6,
        first_group_arrival_spread_ms=(max(x['receive_monotonic_s'] for x in group)-min(x['receive_monotonic_s'] for x in group))*1000,
        measured_tick_RPC_ms=quantiles([(byframe[x['frame']]['tick_rpc_return_wall_s']-byframe[x['frame']]['input_wall_s'])*1000 for x in measurement]),
        model_state_unchanged=True,preprocessor_thread_exited=final['preprocessing_thread_stopped'],cleanup_complete=True,mmap_removed=result['transport_cleanup'],
        remaining_actors=result['cleanup']['remaining'],restored_settings=result['cleanup']['settings'],traffic_manager='NotCreated',
        no_retry=True,server_or_GPU_cause_claim=False))
    write(out/'failure_fix.json',dict(failure='In-memory tuple-of-tuples compared directly with JSON-deserialized list-of-lists',
        evidence='RawImageStamp.transform is tuple; record/asdict retains tuple. Inputs.row reuses it. PartialSlots writes same transform to JSON then compares list to original tuple; Python!= is true despite identical numbers. Historical offline rows were already JSON lists so earlier equivalence missed this representation boundary.',
        source='simulator/m8j13_partial_transport.py: commit',fix='Compare np.asarray transforms with np.array_equal; no tolerance change. A1e-12 element perturbation must still fail.',
        regression_before_live=88,regression_delivery=90,cross_python=load(ROOT/'cross_python_partial.json'),
        post_fix_live_test='NotRun_StartBudgetConsumed',post_fix_model_replay='NotRun_8OfflineAttemptsConsumed; comparison-only fix does not change preprocessing math',
        other_bounded_fixes=['Log failed commit disposition before rethrow','Reject-before-first-camera stays terminal','Missing or pending snapshot is explicitly unconfirmed, never a fresh simulation-age claim'],
        raw_failure_preserved=True))
    write(out/'reachability_decision.json',dict(category='B_OfflineImproved_LiveBlockedBeforeForward',
        observed_offline_subsequent_old_end_to_batch_ms=quantiles([x['old']['end_to_batch_ms'] for x in sub]),
        observed_offline_subsequent_candidate_end_to_batch_ms=quantiles([x['candidate']['end_to_batch_ms'] for x in sub]),
        paired_saved_subsequent_ms=quantiles([x['critical_path_saved_ms'] for x in sub]),
        candidate_process_cpu_ms=quantiles([x['candidate']['process_cpu_ms'] for x in rr]),old_process_cpu_ms=quantiles([x['old']['process_cpu_ms'] for x in rr]),
        counterfactual_checked_subsequent_ms=quantiles([x['counterfactual_checked_ms'] for x in sub]),
        counterfactual_gap_vs300ms=quantiles([x['counterfactual_checked_ms']-300 for x in sub]),
        counterfactual_best_ms=min(x['counterfactual_checked_ms'] for x in sub),counterfactual_under300=7,counterfactual_sample_count=21,
        warning='These are historical-paced CPU measurements joined to same-observation historical model/check durations, not realized end-to-end performance or a universal lower bound.21 measurements are3 repeats of7 inputs, not21 independent observations. First input retained separately and includes cold GPU transfer/inference.',
        actual_online_margin_ms=None,no_realtime_admission=True,
        current_contract_route=dict(requirement='Repair real representation boundary, then separately authorize a bounded validation of this same candidate; no extra startup this round. If pursuing stable300ms, need new evidence for capture delivery, CPU contention or serial model/check cost savings plus actual owner availability; no more blind small optimizations.',
            basis='Offline median is still5.45ms over with no renderer/model contention; P95 about23ms over. Owner wait and in-flight Snapshot confirmation add costs; existing model/official JPEG remain material.'),
        nonrealtime_route=dict(proposal='Separate opt-in synchronous logical-loop protocol if research priority is finite functional driving rather than real-time delivery. Do not silently switch this run.',
            simulation_age='Observation tied to actual frame; advance only explicitly owned ticks; define maximum physical-time gap from sampled frame to command apply.',
            wall_watchdog='Independent monotonic process/RPC/worker deadlines and stop latch; never disable watchdog because simulation is paused.',
            control_validity='Validity tied to original observation plus actual execution acknowledgement; no renewal by computation completion. During compute pause retain independent brake; if owner blocks, command delivery/physical response remain unverified.',
            claims='Logical closed loop only, no5Hz or300ms real-time claim; separate admission and metrics. Four moving-fault tests remain required before any model driving.'),
        recommendation='No automatic optimization round or protocol change. First decide whether strict real-time is the research objective. For functional research, draft separate non-real-time protocol and complete moving faults; for real-time, require new measured timing evidence.'))
    write(out/'acceptance.json',dict(J13_A='Partial',InputEquivalence='Passed_24PairsExact_8HistoricalModelForwardsExact',
        PerCameraPreprocessing='OfflineSingleObservationCriticalPathShortened',LiveIntegration='FailedBeforeFirstForward_TupleListRepresentationMismatch_FixedOffline',
        EndToEndFreshness='Unverified_NoLiveForward_OriginalRealtimeFailuresPreserved',ParkedIsolation='Passed_Only3MeasuredTicks_NoModelActuation',
        Regression='90Passed',RemainingDrivingGates='FourMovingFaultCasesNotRun_RealtimeNotPassed_S0S1S2NotRun',
        exit_category='B_WithExplicitLiveIntegrationBlock',SensorSynchronizationPassed=False,model_controls=0,training=0,S0='NotRun',S1='NotRun',S2='NotRun',
        live_retest_authorized_in_this_round=False,nonrealtime_protocol_activated=False))
    write(out/'provenance.json',dict(base='40f2ebcf1b3af1b6bacaf7e4690e5132ccbc5504',run_commit=load(r/'request.json')['commit'],config_sha256=load(r/'request.json')['config_sha256'],
        assets_before_after_equal=True,official_commit=after['official_commit'],asset_sha256={x['file']:x['sha256'] for x in after['files']},
        default_environment_model_PID_weights_unchanged=True,main='c1644b0587da31396a8db8cb81779ed8b7f8acd8',
        private_root=str(ROOT),private_archive_not_imported=True,regression_command='/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0/.venv-gpu/bin/python -m pytest tests/m8j13 tests/m8j12 --confcutdir=tests -q',
        simulation_command='PYTHONPATH=<existing Pillow dependency directory>:. <existing CARLA0.9.16 python> -m tools.m8j13_parked_shadow --run-config configs/m8j13_per_camera.json'))
    print('Recorded: local improvement, live integration Partial, no retry; all budgets retained')
if __name__=='__main__':main()
