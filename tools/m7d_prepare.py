"""Read raw collection only here; write disjoint current-packet and future stores."""
import argparse,json
from pathlib import Path
from tools.m7d_common import *
from datasets.m7d_temporal import from_packet


def run(split):
 cfg=read('configs/m7d/protocol.json');out=[]
 for spec in cfg['episodes']:
  if spec['split']!=split:continue
  root=Path('reports/m7d_episodes')/spec['id']
  if not (root/'result.json').exists():continue
  rows=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()];route=read(root/'route.json')['points'];body=read(root/'body.json');packroot=Path('reports/m7d_inputs')/spec['id'];packroot.mkdir(parents=True,exist_ok=False)
  futurepath=Path('reports/m7d_evaluator')/spec['id'];futurepath.mkdir(parents=True,exist_ok=False)
  end=rows[-1]['sim_time'];records=[];previous=None
  for i,r in enumerate(rows):
   if i%4 or r['sim_time']+6>end+1e-6:continue
   p={k:r[k] for k in ['episode_id','frame','sim_time','input_available_wall_s','ego','actors','source','valid','missing']};p.update(body=body,route=route,history_stamps=[[x['frame'],x['sim_time']] for x in rows[max(0,i-4):i]])
   inp=from_packet(p);inp.validate(previous);previous=inp;name=f'{i:05d}.json';write(packroot/name,p)
   f=rows[i:i+121];complete=len(f)==121 and all(f[k+1]['frame']-f[k]['frame']==1 for k in range(120))
   write(futurepath/name,dict(source='EvaluatorOnly_RecordedActualCARLA',episode_id=spec['id'],origin_frame=r['frame'],origin_timestamp=r['sim_time'],complete_six_seconds=complete,samples=[dict(frame=x['frame'],sim_time=x['sim_time'],actors=x['actors']) for x in f]))
   records.append(dict(index=i,input=asset(packroot/name)))
  write(packroot/'manifest.json',dict(episode_id=spec['id'],split=split,records=records,map=asset('reports/m7d_execution_logs/Town03.xodr'),protocol=asset('configs/m7d/protocol.json'),source='Privileged_CurrentOnly_NoFuturePaths'))
  out.append(dict(episode_id=spec['id'],split=split,raw=asset(root/'trace.jsonl'),result=read(root/'result.json'),planned_snapshots=len(records),manifest=asset(packroot/'manifest.json'),frame_continuity=all(rows[i+1]['frame']==rows[i]['frame']+1 for i in range(len(rows)-1))))
 write(f'reports/m7d_{split}_episode_preparation.json',dict(provenance=provenance(),kind='ActualRecordedEpisodePacketPreparation',records=out))
 print(split,[(x['episode_id'],x['planned_snapshots']) for x in out])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();run(a.split)
