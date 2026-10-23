"""Recompute historical seven known risk cases without old artifact mutation."""
from dataclasses import replace
import numpy as np
from tools.m7c_common import *
from tools.m7c_equivalence import candidate_from,request_from
from diagnostics.m7b_motion import ArcRoute,ObservedMotion,build_hypotheses,evaluate_dynamic_hypotheses

def main():
 cfg=read('configs/m7b/protocol.json');out=[]
 for item in read('reports/m7a_fixture_validity.json')['valid_future_failures']:
  split=item['scene_id'].split('_road')[0];file='development_v2' if split=='development' else split
  r=next(x for x in read('reports/m7a_planner_'+file+'.json')['records'] if x['scene_id']==item['scene_id']);req=request_from(r);c=candidate_from(next(x for x in r['candidates'] if x['name']==item['candidate']))
  hs=[];u=[]
  for b in req.scene.boxes:
   v=np.array([np.cos(b.yaw),np.sin(b.yaw)]);route=ArcRoute([(*b.xy,b.yaw),(*(np.array(b.xy)+100*v),b.yaw)])
   hh,uu=build_hypotheses(replace(req.scene,boxes=(b,)),[ObservedMotion(b.object_id,route,0.,'historical_declared_tangent')],cfg);hs+=hh;u+=uu
  d=evaluate_dynamic_hypotheses(c,req.scene,hs,u);out.append(dict(scene_id=req.scene_id,candidate=c.name,risk_detected=bool(d['collisions']),result=d))
 write('reports/m7c_historical_seven.json',dict(kind='ActualOfflineHistoricalRiskReplay',provenance=provenance(),records=out,detected=sum(r['risk_detected'] for r in out),total=len(out),scope='Original historical declared tangent corridor; not corrected road placement or new generalization'))
 print('detected',sum(r['risk_detected'] for r in out),'/',len(out))
if __name__=='__main__':main()
