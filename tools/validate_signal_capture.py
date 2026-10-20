"""Audit recorded traffic supervision against committed six-camera frames."""
import argparse,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from simulator.traffic_rules import validate_traffic
from simulator.recorder import file_hash,atomic_json
from tools.validate_recording import validate
from geometry.reference import transform_points


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();scene=Path(a.scene);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    integrity=validate(scene);cal=json.loads((scene/'calibration.json').read_text());counts=Counter();rows=[];panels=[];shown=set()
    for path in sorted((scene/'frames').glob('*.json')):
        f=json.loads(path.read_text());rel=f['labels']['traffic_rules']
        if file_hash(scene/rel)!=f['sha256'][rel]:raise ValueError('Rule checksum')
        r=json.loads((scene/rel).read_text());validate_traffic(r,f['frame_id'],f['timestamp_s'])
        matches=[(s,p) for s in r['signals'] for p in s['stop_positions'] if p['applies_to_current_lane']]
        state=matches[0][0]['state'] if matches else 'NoMatch';counts[state]+=1
        counts['valid']+=r['ego_stop_valid'];counts['invalid']+=not r['ego_stop_valid']
        if r['ego_stop_valid']:counts['valid_'+state]+=1
        if state=='Yellow' and r['ego_stop_valid']:raise ValueError('Yellow policy incorrectly supervised')
        rows.append({'frame_id':f['frame_id'],'state':state,'valid':r['ego_stop_valid'],'stop_required':r['ego_stop_required'],'distance_m':r['ego_stop_distance_m']})
        if state not in shown and (r['ego_stop_valid'] or state=='Yellow'):
            shown.add(state);rgb=cv2.imread(str(scene/f['cameras']['front']['rgb']));c=cal['cameras']['front'];k=np.array(c['K'])
            if matches:
                ends=np.array(matches[0][1]['stop_segment_hs_ego_m']);opt=transform_points(ends,np.linalg.inv(np.array(c['T_hs_ego_from_opt'])))
                if np.all(opt[:,2]>0):
                    uv=opt@k.T;uv=uv[:,:2]/uv[:,2:];cv2.line(rgb,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),(255,180,20),2)
            rgb=cv2.resize(rgb,(640,384));rgb=cv2.copyMakeBorder(rgb,40,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(rgb,f"{state} | stop={r['ego_stop_required']} valid={r['ego_stop_valid']} frame={f['frame_id']}",(8,19),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            cv2.putText(rgb,'Cyan: trigger-derived stop segment; green signal != full permission',(8,35),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);panels.append(rgb)
    checks={'recording_integrity':integrity['recording_integrity_checks_passed'],
        'red_and_green_valid':counts['valid_Red']>0 and counts['valid_Green']>0,
        'yellow_present_masked':counts['Yellow']>0 and counts['valid_Yellow']==0,
        'transition_guard_exercised':counts['invalid']>0,'every_frame_has_rules':len(rows)==integrity['frames_verified']}
    atomic_json(out/'report.json',{'schema_version':'hsd-signal-audit-v1','passed':all(checks.values()),'checks':checks,'counts':dict(counts),'rows':rows,
        'integrity':integrity,'manifest_sha256':file_hash(scene/'manifest.json'),
        'scope':'Controlled stationary diagnostic; metadata/frame alignment and rule truth table. Not learned traffic perception or driving policy; painted-line alignment not asserted.'})
    if panels:cv2.imwrite(str(out/'signal_timeline.png'),np.concatenate(panels))
    print(json.dumps(checks,indent=2))
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
