"""Additional event details, thresholds fixed in protocol, no refitting."""
import json
from pathlib import Path
import numpy as np
from tools.m6g_analyze import read,write,asset,counts


def main():
    manifest=read('reports/m6g_execution_logs/trial_manifest.json');trials={t['trial_id']:t for t in manifest['trials']};series={k:[json.loads(l) for l in Path(t['timeline']['path']).read_text().splitlines()] for k,t in trials.items()}
    report=read('reports/m6g_braking_event_analysis.json');events=[]
    for e in report['events']:
        if e['category']!='ordinary_light_brake':continue
        samples=series[e['trial_id']];a=e['applied_index'];end=e['end_index'];wheels=[]
        for w in range(4):
            previous=samples[max(0,a-1)]['telemetry']['wheels'][w];already=abs(previous['long_slip'])>=.5
            j=next((i for i in range(a,end) if abs(samples[i]['telemetry']['wheels'][w]['long_slip'])>=.5),None)
            if j is None:wheels.append(dict(wheel_index=w,onset='NotObserved'));continue
            before=samples[max(0,j-1)]['telemetry']['wheels'][w];now=samples[j]['telemetry']['wheels'][w]
            wheels.append(dict(wheel_index=w,first_high_slip_frame=samples[j]['frame'],first_high_slip_index=j,already_high_before_braking=already,
              interpretation='left_censored_preexisting_slip' if already else 'first_observed_high_slip_within_brake',
              previous=before,current=now,load_delta_N=now['tire_load']-before['tire_load'],long_force_delta_N=now['long_force']-before['long_force'],
              evidence='Observed same outer frame; refresh/substep phase unknown'))
        peak=e['metrics']['peak_deceleration']['index'];s=samples[peak]
        events.append(dict(event_id=e['event_id'],trial_id=e['trial_id'],physics_sha256=e['physics_sha256'],source_trace_sha256=e['source_trace_sha256'],acquisition_git=e['acquisition_git'],
             per_wheel_slip_load_force=wheels,peak_frame=s['frame'],peak_index=peak,
             peak_force_zero=all(abs(w['long_force'])<=1 for w in s['telemetry']['wheels']),
             previous_peak_current=[samples[max(0,peak-1)],s],
             earliest_all_locked=e['metrics']['first']['all_wheels_locked'],outer_order=e['metrics']['ordering']))
    rebuild=read('reports/m6g_physics_rebuild_analysis.json')['events'];groups={x['trial_id']:x for x in rebuild};pairs=[]
    for rep in [401,402]:
        base=groups[f'sham__P0-unmodified_{rep}']
        for group in ['P0-sham-reapply','P1-wheel-brake-experimental']:
            other=groups[f'sham__{group}_{rep}'];pairs.append(dict(repeat_id=rep,group=group,baseline=base['trial_id'],other=other['trial_id'],
              first_tick_baseline=base['first_ticks'][0],first_tick_other=other['first_ticks'][0],
              first_tick_velocity_difference_world_mps=np.subtract(other['first_ticks'][0]['body_velocity_world'],base['first_ticks'][0]['body_velocity_world']).tolist(),
              first_tick_z_difference_m=other['first_ticks'][0]['body_position_world'][2]-base['first_ticks'][0]['body_position_world'][2],
              interpretation='Observed body vertical dynamics reset/initialization discontinuity; public Physics same only for Sham; exact internal state reset list unresolved'))
    write('reports/m6g_execution_logs/event_details.json',dict(kind='OfflineAnalysisOfActualM6F',source='Privileged',events=events,
        summary=dict(ordinary_events=len(events),P0_events=sum('P1-' not in e['trial_id'] for e in events),P0_order=counts(e['outer_order'] for e in events if 'P1-' not in e['trial_id']),peak_force_zero=sum(e['peak_force_zero'] for e in events),
          wheel_high_slip_observations=sum(w['onset']!='NotObserved' if 'onset' in w else True for e in events for w in e['per_wheel_slip_load_force']),
          wheel_slip_left_censored=sum(w.get('already_high_before_braking',False) for e in events for w in e['per_wheel_slip_load_force'])),rebuild_pairs=pairs,source_manifest=asset('reports/m6g_execution_logs/trial_manifest.json')))

if __name__=='__main__':main()
