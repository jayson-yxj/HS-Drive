"""Logging-only outer timing audit, frozen plan function and decisions unchanged."""
import argparse,json,time,subprocess,os
from pathlib import Path


def child(episode):
    from tools import m7d_replay as replay
    from tools.m7d_common import write
    from datasets.m7d_temporal import result_clock
    from dataclasses import asdict
    original=replay.plan
    def timed(inp,*args,**kwargs):
        start=time.perf_counter();record=original(inp,*args,**kwargs);ready=time.perf_counter()
        # Serialize the exact original scientific result before timestamping delivery.
        payload=json.dumps(record,allow_nan=False).encode();serialized=time.perf_counter()
        record['core_clock']=record['clock']
        record['clock']=asdict(result_clock(inp,start,serialized,record['selected']))
        record['availability_audit']=dict(frozen_result_object_ready_wall_s=ready,serialized_result_ready_wall_s=serialized,
            complete_compute_and_serialization_ms=(serialized-start)*1000,serialization_ms=(serialized-ready)*1000,
            serialized_bytes=len(payload),delivery_semantics='Offline in-memory serialized payload ready, no IPC/actuator delivery; disk archive later',logging_only_no_decision_change=True)
        return record
    replay.plan=timed
    replay.write=lambda p,d:write(p.replace('_stream.json','_availability.json'),d)
    replay.run(episode,'stream')


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode');a=p.parse_args()
    if a.episode:return child(a.episode)
    for manifest in sorted(Path('reports/m7d_inputs').glob('*/manifest.json')):
        episode=manifest.parent.name
        with Path(f'reports/m7d_execution_logs/availability_{episode}.txt').open('x') as f:
            subprocess.run(['.venv/bin/python','-m','tools.m7d_availability','--episode',episode],env={**os.environ,'PYTHONPATH':'runs/m7a_geometry_deps:.'},stdout=f,stderr=subprocess.STDOUT,check=True)
        print(episode,'timing audited',flush=True)
if __name__=='__main__':main()
