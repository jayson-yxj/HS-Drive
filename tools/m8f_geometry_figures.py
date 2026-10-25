"""Actual requested/final footprint and ray support diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m8f_common import *

def main():
 files=[];protocol=json.loads(Path('configs/m8f/protocol.json').read_text())
 for g in protocol['groups']:
  req=json.loads((ROOT/'capture'/g['group']/'evaluator/grounded_requests.json').read_text())['requests']
  fig=plt.figure(figsize=(14,7));ax=fig.add_subplot(121);az=fig.add_subplot(122,projection='3d')
  ego=g['ego']['xyz'];ax.scatter(ego[0],ego[1],c='black',marker='*',label='Ego')
  for r in req:
   q=r['preflight']['present'];color='green' if r['accepted_for_spawn'] else 'red'
   if q['requested_obb']:
    p=np.asarray(q['requested_obb']['corners_world_m']);b=p[[0,4,6,2,0]];ax.plot(b[:,0],b[:,1],c=color);ax.text(p[:,0].mean(),p[:,1].mean(),r['key'],fontsize=8)
    az.plot(b[:,0],b[:,1],b[:,2],c=color)
   for rr in q['rays']:
    for h in rr['hits']:
     x,y,z=h['xyz'];az.scatter(x,y,z,c='blue' if h['semantic'] in [1,2,24] else 'orange',s=5)
   if not q['requested_obb']:
    p=q['waypoint']['xyz'];ax.scatter(p[0],p[1],c=color);ax.text(p[0],p[1],r['key'])
  ax.set_aspect('equal');ax.set_xlabel('CARLA world X m');ax.set_ylabel('CARLA world Y m');ax.set_title('Footprint requests: green accepted, red rejected')
  az.set_title('Recorded ray hits and proposed OBB floor');az.set_xlabel('X');az.set_ylabel('Y');az.set_zlabel('Z')
  fig.suptitle(g['group']+' / geometry proxy, no physical contact certification')
  p=Path('reports/m8f_figures')/(g['group']+'_geometry.png');fig.savefig(p,dpi=140);plt.close(fig);files.append(asset(p))
 checks=json.loads((ROOT/'capture/m8f_development_01/evaluator/present_support.json').read_text())['checks']
 fig,ax=plt.subplots(figsize=(8,4))
 for phase in ['present','shifted']:
  checks=json.loads((ROOT/'capture/m8f_development_01/evaluator'/(phase+'_support.json')).read_text())['checks']
  for r in checks:ax.plot(r['final']['support_gaps_m'],'o-',label=phase)
 ax.axhline(.12,c='red',ls='--');ax.axhline(-.03,c='red',ls='--');ax.set_xlabel('Actual footprint support sample');ax.set_ylabel('OBB floor minus measured surface (m)');ax.legend();fig.tight_layout()
 p=Path('reports/m8f_figures/final_support_gaps.png');fig.savefig(p,dpi=140);plt.close(fig);files.append(asset(p))
 write('reports/m8f_geometry_visualization.json',dict(**provenance('OfflineActualRayAndOBBFigures'),files=files))
if __name__=='__main__':main()
