"""Measure capture diversity and motion without treating frame count as independence."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from simulator.recorder import atomic_json,file_hash


def audit(scene):
    root=Path(scene);m=json.loads((root/'manifest.json').read_text())
    frames=[json.loads(p.read_text()) for p in sorted((root/'frames').glob('*.json'))]
    if not frames:raise ValueError('Empty scene')
    poses=np.array([f['ego']['T_world_from_ego'] for f in frames]);xy=poses[:,:2,3]
    heading=np.unwrap(np.arctan2(poses[:,1,0],poses[:,0,0]))
    speed=np.array([f['ego']['speed_mps'] for f in frames]);times=np.array([f['timestamp_s'] for f in frames])
    distance=np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()
    cells=len(np.unique(np.floor(xy/5).astype(int),axis=0))
    cadence=bool(np.allclose(np.diff(times),.1,atol=1e-5,rtol=0))
    return {'scene':str(root),'map':m['map'].split('/')[-1],'status':m['status'],
        'frames':len(frames),'duration_s':float(times[-1]-times[0]),'distance_m':float(distance),
        'displacement_m':float(np.linalg.norm(xy[-1]-xy[0])),
        'heading_span_deg':float(np.ptp(heading)*180/np.pi),'moving_fraction_gt_1mps':float((speed>1).mean()),
        'speed_p50_mps':float(np.median(speed)),'visited_5m_cells':cells,'continuous_10hz':cadence,
        'motion_accepted':bool(m['status']=='complete' and len(frames)>=100 and distance>=20 and (speed>1).mean()>=.5 and cadence),
        'manifest_sha256':file_hash(root/'manifest.json')},xy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',default='configs/capture_expanded.yaml')
    p.add_argument('--output',required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.campaign).read_text())
    scenes=[];positions={}
    for r in cfg['runs']:
        report,xy=audit(Path(cfg['raw_root'])/r['run_id']);report.update(run_id=r['run_id'],split=r['split'],included=r.get('include_training',True),exclusion_reason=r.get('exclusion_reason'))
        scenes.append(report);positions[r['run_id']]=xy
    pairs=[]
    for i,first in enumerate(scenes):
        for second in scenes[i+1:]:
            if first['map']!=second['map']:continue
            x=positions[first['run_id']];y=positions[second['run_id']]
            d=np.linalg.norm(x[:,None]-y[None],axis=-1)
            pairs.append({'first':first['run_id'],'second':second['run_id'],'min_ego_distance_m':float(d.min()),
                'first_fraction_within_10m_of_second':float((d.min(1)<10).mean())})
    report={'schema_version':'hsd-motion-audit-v1','scenes':scenes,'same_map_route_overlap':pairs,
        'all_motion_accepted':all(s['motion_accepted'] for s in scenes if s['included']),
        'total_distance_m':sum(s['distance_m'] for s in scenes if s['included']),
        'excluded_scenes':[s['run_id'] for s in scenes if not s['included']],
        'scope':'No NPC traffic; TM drives are data collection, not learned closed-loop success; map/route grouping prevents split leakage'}
    draw_routes(scenes,positions,Path(a.output).with_suffix('.png'))
    atomic_json(a.output,report);print(json.dumps(report,indent=2))
    if not report['all_motion_accepted']:raise SystemExit(1)

def draw_routes(scenes,positions,path):
    import cv2
    panels=[];palette=[(40,190,40),(210,120,20),(30,160,220)]
    for name in sorted({s['map'] for s in scenes}):
        group=[s for s in scenes if s['map']==name];all_xy=np.concatenate([positions[s['run_id']] for s in group])
        lo=all_xy.min(0)-10;hi=all_xy.max(0)+10;scale=min(500/(hi[0]-lo[0]),340/(hi[1]-lo[1]))
        panel=np.full((480,600,3),245,np.uint8)
        cv2.putText(panel,name+' | HS world meters | ego trajectories only',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
        for i,s in enumerate(group):
            xy=(positions[s['run_id']]-lo)*scale;xy[:,1]=360-xy[:,1];pixels=np.rint(xy+[40,10]).astype(np.int32)
            color=palette[i%len(palette)] if s['included'] else (120,120,120)
            cv2.polylines(panel,[pixels],False,color,3);cv2.circle(panel,tuple(pixels[0]),6,color,2)
            legend=f"{s['run_id']} {s['split']} {s['distance_m']:.1f}m"+(' EXCLUDED' if not s['included'] else '')
            cv2.putText(panel,legend,(12,400+22*i),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.line(panel,(430,375),(430+round(10*scale),375),(20,20,20),2)
        cv2.putText(panel,'10m',(430,365),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        panels.append(panel)
    while len(panels)%2:panels.append(np.full_like(panels[0],245))
    cv2.imwrite(str(path),np.concatenate([np.concatenate(panels[i:i+2],1) for i in range(0,len(panels),2)]))


if __name__=='__main__':main()
