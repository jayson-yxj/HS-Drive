"""Additional fixed-object diagnostics by distance/view/pose and occlusion configuration."""
import json
from pathlib import Path
from tools.m5l_common import write,asset,summary

def main():
 for split in ['val','regression','test']:
  p=f'runs/m5l_objects_{split}.json';raw=json.loads(Path(p).read_text());rr=raw['actors'];out=[]
  for seed in [101,102]:
   for axis,values in [('camera',sorted({r['camera']for r in rr})),('type',['vehicle','person','cone']),('distance',['near','10_20','20_35','far']),('pose',['flat','slope']),('configuration',['occlusion','other'])]:
    for value in values:
     def selected(r):
      if r['seed']!=seed:return False
      if axis in ['camera','type']:return r[axis]==value
      if axis=='distance':return {'near':r['distance_m']<10,'10_20':10<=r['distance_m']<20,'20_35':20<=r['distance_m']<35,'far':r['distance_m']>=35}[value]
      if axis=='pose':return (abs(r['ego_pitch_deg'])<=1)==(value=='flat')
      occluded=r['configuration']in ([16,17]if split=='test'else[24]);return occluded==(value=='occlusion')
     group=[r for r in rr if selected(r)];v=[r for r in group if r['valid']];out.append(dict(seed=seed,axis=axis,value=value,actor_views=len(group),detected=sum(r['detected']for r in group),valid=len(v),unmatched=sum(not r['detected']for r in group),abstain=sum(r['detected']and not r['valid']for r in group),position_count=len({r['group']for r in v}),metrics={m:{k:summary([r[m][k]for r in v])for k in ['xy_m','radial_error_m','bearing_error_deg']}for m in ['plane_center','plane_bottom','dense_expected_center','dense_mode_center']}))
  write(f'reports/m5l_object_strata_{split}.json',dict(kind='offline_frozen_object_outputs',records=out,raw=asset(p),limits='Pose pitchproxy not exactlocalroadplane. Occlusion labels configurationintent, not independentvisiblecontact truth. Nullmetrics zero-support unevaluated.'))
if __name__=='__main__':main()
