"""Historical measured records; availability retained from original snapshot_end."""
import json, math
from pathlib import Path
from tools.m7d_common import *


def main():
    cfg=read('configs/m7d/protocol.json');body=read('reports/m6b_body_reference.json')
    for name in cfg['history_selection']:
        root=Path('reports/m6b_trials')/name;allrows=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        # Fixed first 18 seconds selected by protocol, not by result.
        rows=allrows[:361];route=read(root/'route.json')['selected'];episode='historical_'+name
        if isinstance(route,dict):raise ValueError('unsupported_route_schema')
        target=Path('reports/m7d_inputs')/episode;target.mkdir(parents=True,exist_ok=False)
        future=Path('reports/m7d_evaluator')/episode;future.mkdir(parents=True,exist_ok=False);records=[]
        def actors(r):
            return [{**a,'category':'Vehicle','role':'HistoricalRecordedActor','blueprint':'Unknown_HistoricalFieldAbsent'} for a in r['obstacles']]
        for i in range(0,len(rows)-120,4):
            r=rows[i];m=r['measurement'];p=dict(episode_id=episode,frame=r['frame'],sim_time=r['sim_time'],input_available_wall_s=r['timeline']['snapshot_end'],valid=True,missing=False,source='Privileged_ActualCARLAActorSnapshot',
                 ego=dict(xy=[m['position'][0],-m['position'][1]],z=m['position'][2],yaw=-math.radians(m['rotation'][2]),signed_speed=m['long_speed']),
                 actors=actors(r),body=dict(half_extent=body['extent'][:2],offset=[body['location'][0],-body['location'][1]]),route=route,
                 history_stamps=[[x['frame'],x['sim_time']] for x in rows[max(0,i-4):i]])
            name2=f'{i:05d}.json';write(target/name2,p);f=rows[i:i+121]
            write(future/name2,dict(source='EvaluatorOnly_RecordedActualCARLA',episode_id=episode,origin_frame=r['frame'],origin_timestamp=r['sim_time'],complete_six_seconds=True,
                                   samples=[dict(frame=x['frame'],sim_time=x['sim_time'],actors=actors(x)) for x in f]))
            records.append(dict(index=i,input=asset(target/name2)))
        write(target/'manifest.json',dict(episode_id=episode,split='historical_regression',records=records,map=asset('reports/m6b_execution_logs/Town03.xodr'),protocol=asset('configs/m7d/protocol.json'),
            raw=asset(root/'trace.jsonl'),source='Privileged_CurrentOnly_NoFuturePaths',limitations='Actor blueprint absent; category Vehicle known from original scene code; historical data never blind. 18s prefix chosen before replay.'))
        print(episode,len(records))
if __name__=='__main__':main()
