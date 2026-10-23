"""Detailed offline interval/application audit, preserving first analysis files."""
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from tools.m6f_analyze import read,write,quant


def asset(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def main():
    rows_by_trial={};commands=[];runtime=[];peaks=[];startup=[];sham=[];badframes=0;global_api_diff=[];nrows=0;stale_control=0
    for p in sorted(Path('reports/m6f_trials_0916').glob('*/*/trace.jsonl')):
        rs=[json.loads(s) for s in p.read_text().splitlines()];rows_by_trial[str(p.parent)]=rs;nrows+=len(rs)
        config=read(p.with_name('config.json'));init=read(p.with_name('initialization.json'));result=read(p.with_name('result.json'))
        intervals=np.diff([r['timeline']['apply'] for r in rs])*1000
        runtime.append(dict(trial=str(p.parent),control_intervals_ms=quant(intervals),mean_Hz=1000/float(np.mean(intervals)),deadline_misses=sum(r['deadline_missed'] for r in rs),skipped_slots=sum(r['skipped_slots'] for r in rs),work_ms=quant([(r['timeline']['end']-r['timeline']['start'])*1000 for r in rs])))
        for i,r in enumerate(rs):
            after=r['telemetry_after']
            if after is None:continue
            badframes+=after['associated_frame']!=r['body']['frame']
            ks=[k for k in ['throttle','brake','steer'] if abs(r['requested'][k]-after['data'][k])>1e-5]
            stale_control+=any(abs(r['body']['applied'][k]-after['data'][k])>1e-5 for k in ['throttle','brake','steer'])
            if ks:
                delay=next((j-i for j in range(i+1,min(i+4,len(rs))) if rs[j]['telemetry_after'] and all(abs(rs[j]['telemetry_after']['data'][k]-r['requested'][k])<1e-5 for k in ks)),None)
                commands.append(dict(trial=str(p.parent),frame=r['body']['frame'],phase=r['phase'],phase_tick=r['phase_tick'],fields=ks,request=r['requested'],actual=after['data'],applied_match_delay_ticks=delay))
        for phase in ['cold_run','warm_run','normal_stop','final_stop','coast']:
            ids=[i for i,r in enumerate(rs) if r['phase']==phase]
            if not ids:continue
            begin=ids[0];end=ids[-1];t=np.array([r['body']['t'] for r in rs]);v=np.array([r['body']['long_speed'] for r in rs]);a=np.diff(v)/np.diff(t)
            if phase.endswith('run'):
                app=next((i for i in ids if rs[i]['telemetry_after']['data']['throttle']>.1 and rs[i]['telemetry_after']['data']['brake']<.001),None)
                moving=next((i for i in ids if v[i]>.08),None)
                gear1=next((i for i in ids if rs[i]['telemetry_after']['data']['gear']==1),None)
                startup.append(dict(trial=str(p.parent),phase=phase,interface=config['interface'],delay_from_request_s=None if moving is None else float(t[moving]-rs[begin]['body_before']['t']),
                    delay_after_first_applied_snapshot_s=None if app is None or moving is None else float(t[moving]-t[app]),gear1_delay_s=None if gear1 is None else float(t[gear1]-rs[begin]['body_before']['t']),
                    first_applied_frame=None if app is None else rs[app]['body']['frame'],moving_frame=None if moving is None else rs[moving]['body']['frame']))
            if phase in ['normal_stop','final_stop']:
                indices=np.arange(max(1,begin),end+1);i=int(indices[np.argmin(a[indices-1])]);r=rs[i];prev=rs[i-1]
                peaks.append(dict(trial=str(p.parent),phase=phase,raw_decel=float(-a[i-1]),frame=r['body']['frame'],body_before=prev['body'],body_after=r['body'],previous_wheels=prev['telemetry_after']['data']['wheels'],current_wheels=r['telemetry_after']['data']['wheels'],applied=r['telemetry_after']['data'],request=r['requested'],note='force is debug sampled state, not integrated interval force'))
        if '/sham/' in str(p):
            before=init['before_operation'];after=init['after_operation']
            sham.append(dict(trial=str(p.parent),config=config,physics_sha=result['physics_sha256'],cached_body_identical=before['body']==after['body'],frame_before=before['telemetry']['associated_frame'],frame_after=after['telemetry']['associated_frame'],live_speed_before=before['telemetry']['data']['speed'],live_speed_after=after['telemetry']['data']['speed'],speed_jump_mps=after['telemetry']['data']['speed']-before['telemetry']['data']['speed'],alignment='Physics_rebuild_barrier_not_same_physical_state_despite_same_frame'))
    write('reports/m6f_execution_logs/detailed_evidence.json',dict(kind='offline_raw_CARLA_trace_audit',source='Privileged',versions=['0.9.16'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       trial_count=len(rows_by_trial),rows=nrows,bad_frame_brackets=badframes,applied_control_vs_telemetry_mismatches=stale_control,command_transition_delays=commands,startup=startup,braking_peaks=peaks,physics_reapply=sham,runtime=runtime,
       independent_samples='3 response locations x2 deterministic repeats,1 sham location x2, one stationary overhead location; frames not independent statistical trials'))
    # Enrich current-stage reports, retaining byte-identical first analysis below.
    for name in ['actuation_causal_probe','physics_sham_comparison','runtime_overhead','guard_validation']:
        path=Path('reports/m6f_'+name+'.json');original=path.read_bytes();backup=Path('reports/m6f_execution_logs/analysis_v1_'+name+'.json');backup.write_bytes(original)
        d=json.loads(original)
        d['evidence_details']=asset('reports/m6f_execution_logs/detailed_evidence.json')
        d['analysis_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        d['input_trace_manifest']=[asset(p) for p in sorted(Path('reports/m6f_guard_0915' if name=='guard_validation' else 'reports/m6f_trials_0916').glob('p*/trace.jsonl' if name=='guard_validation' else '*/*/trace.jsonl'))]
        if name=='actuation_causal_probe':
            d['command_alignment']=dict(rows_with_delayed_request=len(commands),all_delays_one_tick=all(x['applied_match_delay_ticks']==1 for x in commands),impact='request-time pairings are not exact same applied-time interventions; records retain all25 delayed transitions; no dropped data',applied_vs_telemetry_mismatches=stale_control)
            d['causal_status']='Partial: gear0 wait supported by manual intervention; wheel lock/slip/force transitions observed; actuator torque and substep cause not identified'
        elif name=='physics_sham_comparison':
            d['initial_state_operations']=sham
            d['interpretation']='Sham resets live speed and changes subsequent response even with equal public Physics; async command timing also differs, so full decel delta cannot be uniquely assigned to rebuild or brake torque.'
        elif name=='runtime_overhead':d['all_trial_timing']=runtime
        elif name=='guard_validation':
            d['status']='Partial_NoIndependentDangerousPositive';d['protected_metrics_status']='NotRun_no_emergency_on_new_positions';d['risk_events']=sum(len(x['guard_events']) for x in d['trials'])
        path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
