"""Reliable teacher agreement/conflicts stratified by camera, scene, class, range."""
import json
from pathlib import Path
import numpy as np,torch,cv2
from tools.m5i_data import rows
from tools.m5f_common import pixel_label
from training.m5i_transfer import masks
from simulator.recorder import atomic_json,file_hash

def teacher_index():return {r['sample_id']:r for f in ['runs/m5h_teacher_trainval/index.json','runs/m5i_teacher_trainval/index.json']for r in json.loads(Path(f).read_text())['records']}
def main():
    dest=Path('reports/m5i_teacher_transfer_audit.json')
    if dest.exists():raise FileExistsError(dest)
    ti=teacher_index();records=[]
    for r in rows():
        d=pixel_label(r);t=ti[r['sample_id']];assert file_hash(t['path'])==t['sha256'];q=np.load(t['path']);score=q['score'].astype(np.float32)/256;trusted=q['trusted'].astype(np.float32)/256;soft,p,n=masks(torch.from_numpy(d['target']),torch.from_numpy(d['valid']),torch.from_numpy(d['classes']),torch.from_numpy(score),torch.from_numpy(trusted));p=p.numpy();n=n.numpy();pred=score.max(0)
        for c,cam in enumerate(r['camera_names']):
            boundary=cv2.dilate(d['target'][c].astype(np.uint8),np.ones((5,5),np.uint8))!=cv2.erode(d['target'][c].astype(np.uint8),np.ones((5,5),np.uint8))
            for lo,hi in [(0,10),(10,20),(20,35),(35,80)]:
                area=(d['radial'][c]>=lo)&(d['radial'][c]<hi);v=d['valid'][c]&area;target=d['target'][c];records.append(dict(sample_id=r['sample_id'],split=r['split'],group=r['group'],camera=cam,weather=r['weather'],distance=[lo,hi],positive_kd=int((p[c]&area).sum()),negative_kd=int((n[c]&area).sum()),gt_positive_teacher_miss=int((v&target&np.isin(d['classes'][c],[1,2])&(pred[c]<.05)).sum()),gt_negative_teacher_conflict=int((v&~target&(pred[c]>=.7)).sum()),boundary_known=int((v&boundary).sum()),boundary_unknown_teacher_high=int((area&~d['valid'][c]&boundary&(pred[c]>=.7)).sum()),class_support={str(k):int((v&target&(d['classes'][c]==k)).sum())for k in [1,2,3]}))
    atomic_json(dest,dict(kind='offline_actual_RGB_teacher_vs_CARLA_trainval',records=records,source_sha256=file_hash(__file__),contract='GTneveroverwritten. SoftKDpositiveonlysupportedagreements; negativesonlyGTreliable nontarget teacher<=.05. Unknown/conesexcluded. FeatureKDseparateGTmaskedsemanticmatching; no outputKD combined.',limits='Teacherzero maydetectorabsence, notcertainbackground; GTconfirmsnegative. Ignoredocclusionedgecannotcertifyteachererror. Boundarycountsdiagnosticonly.'))
if __name__=='__main__':main()
