"""Phase-resolved actuator diagnostics preserve all raw probe/prepare samples."""
import json
from pathlib import Path
import numpy as np
from tools.m6c_common import write,provenance,asset
from tools.m47_common import dynamics_metrics


def main():
    report=json.loads(Path('reports/m6c_low_speed_dynamics.json').read_text());trials=[]
    for d in report['trials']:
        r=[json.loads(s) for s in Path(d['trace']['path']).read_text().splitlines()];start=d['probe_start'];sub=r[start:] if start is not None else []
        firsthold=next((i for i,q in enumerate(sub) if q['phase']=='hold'),None)
        firstcmd=None;response=None
        if d['task']['mode']=='stop' and start is not None:
            firstcmd=sub[0]['t']
            # First measured brake>0 and measured longitudinal deceleration are
            # separate events; no invented actuator delay if snapshot misses it.
            response=next((q for q in sub if q['long_acceleration_api']<-.1),None)
        hold=sub[firsthold:] if firsthold is not None else []
        phases={}
        for phase in ['prepare','warm_stop','probe','coast','hold']:
            rr=[q for q in r if q['phase']==phase]
            if len(rr)>=8:phases[phase]=dynamics_metrics(rr)
        coast=[]
        for lo,hi in zip([0,.25,.5,1,1.5,2],[.25,.5,1,1.5,2,3.5]):
            rows=[q for q in sub if q['command']['throttle']==q['command']['brake']==0 and lo<=q['speed']<hi]
            if rows:coast.append(dict(speed_range=[lo,hi],n=len(rows),a_min=min(q['long_acceleration_api'] for q in rows),a_max=max(q['long_acceleration_api'] for q in rows),a_mean=float(np.mean([q['long_acceleration_api'] for q in rows]))))
        trials.append(dict(name=d['name'],task=d['task'],source='Privileged',trace=d['trace'],
            initial_speed=None if not sub else sub[0]['speed'],initial_signed_speed=None if not sub else sub[0]['long_speed'],
            initial_gear=None if not sub else sub[0]['applied']['gear'],
            initial_pitch=None if not sub else sub[0]['rotation'][1],initial_road=None if not sub else sub[0]['road'],
            actual_grade_min=min(q['road']['grade'] for q in r),actual_grade_max=max(q['road']['grade'] for q in r),
            probe_metrics=None if len(sub)<8 else dynamics_metrics(sub),phase_metrics=phases,
            first_measured_deceleration_delay_s=None if response is None else response['t']-firstcmd,
            first_applied_positive_brake_delay_s=next((q['t']-firstcmd for q in sub if q['applied']['brake']>0),None) if firstcmd is not None else None,
            hold_entry_speed=None if not hold else hold[0]['speed'],
            hold_drift_m=None if not hold else float(np.linalg.norm(np.array([q['position'] for q in hold])-hold[0]['position'],axis=1).max()),
            coast_acceleration_by_speed=coast))
    write('reports/m6c_probe_phase_analysis.json',dict(**provenance('offline_phase_analysis_actual_CARLA'),source='Privileged',trials=trials,
        limits='Response event threshold a<-.1 is a measurement definition, not comfort gate. Snapshot-applied control may precede/delay velocity response. Probe initialization phase is disclosed and never removed from whole-run metrics.'))

if __name__=='__main__':main()
