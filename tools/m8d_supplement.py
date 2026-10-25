"""Full-cohort output changes and fixed historical identity loss, without retuning."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tools.m8d_common import *
from diagnostics.m8d_evaluation import paired_counts

def main():
    rows=[]
    for token in CFG['full_tokens']:
        a=json.loads((ROOT/'full'/(token+'_A.json')).read_text())
        b=json.loads((ROOT/'full'/(token+'_B.json')).read_text())
        delta=np.abs(np.asarray(a['boxes'])-b['boxes'])
        rows.append(dict(token=token,boxes_exact=bool(np.array_equal(a['boxes'],b['boxes'])),
            scores_exact=bool(np.array_equal(a['scores'],b['scores'])),
            ranked_box_abs=stats(delta),ranked_score_abs=stats(np.abs(np.array(a['scores'])-b['scores'])),
            ranked_label_changes=int(np.count_nonzero(np.array(a['labels'])!=b['labels'])),
            A_score01=sum(s>=.1 for s in a['scores']),B_score01=sum(s>=.1 for s in b['scores'])))
    identity=json.loads(Path('reports/m8d_paired_identity_support.json').read_text())
    common=identity['common_supported']; byid={v:{(r['token'],r['gt_index']):r for r in identity['all_rows'][v]} for v in ['A','B']}
    differences=[dict(token=k[0],gt_index=k[1],
        A_XY=byid['A'][tuple(k)]['metric_error']['XY_m'],
        B_XY=byid['B'][tuple(k)]['metric_error']['XY_m'],
        delta_XY=byid['B'][tuple(k)]['metric_error']['XY_m']-byid['A'][tuple(k)]['metric_error']['XY_m']) for k in common]
    frames=json.loads((ROOT/'evaluation.json').read_text())['frames']
    transitions4=[]
    for f in frames:
        a=f['variants']['A']['sensitivity4m'];b=f['variants']['B']['sensitivity4m']
        for gi in range(a['gt']):
            transitions4.append(dict(A=any(j==gi for i,j in a['pairs']),B=any(j==gi for i,j in b['pairs'])))
    latency=json.loads(Path('reports/m8d_latency_resource.json').read_text())
    raw=json.loads((ROOT/'full/run.json').read_text())['records']
    by={(r['token'],r['variant']):r for r in raw}
    timing_delta={k:stats([by[(t,'B')]['latency_ms'][k]-by[(t,'A')]['latency_ms'][k]
                         for t in CFG['full_tokens'][1:]]) for k in ['permutation_ms','preprocess_ms','model_forward_ms','e2e_before_artifact_write_ms']}
    lost=[r for r in identity['A_fixed_six'] if r['B']['grade']!='IdentitySupported']
    figures=[]
    for item in lost:
        token,gi=item['key'];packet=json.loads((BASELINE/'rgb_bundle'/token/'packet.json').read_text())
        views=item['A']['views']
        fig,axes=plt.subplots(len(views),2,figsize=(12,3.5*len(views)),squeeze=False)
        for i,view in enumerate(views):
            ci=view['camera_index'];cam=packet['cameras'][ci]
            for j,v in enumerate(['A','B']):
                ev=next(x for x in item[v]['views'] if x['camera_index']==ci)
                ax=axes[i,j];ax.imshow(Image.open(cam['path']))
                for box,color in [(ev['instance_bbox'],'lime'),(ev['projected_bbox'],'red')]:
                    if box:
                        box=np.asarray(box);ax.add_patch(plt.Rectangle(box[:2],*(box[2:]-box[:2]),fill=False,color=color,lw=1.5))
                ax.set_title('%s %s %s; mask recall %.4f'%(token,v,cam['name'],ev['mask_recall']))
                ax.axis('off')
        path=Path('reports/m8d_figures')/(token+'_identity_threshold.png')
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig);figures.append(asset(path))
    write('reports/m8d_paired_supplement.json',dict(**provenance('OfflineFullCohortPairedSupplement'),
        full192_output_change=rows,frames_with_changed_boxes=sum(not r['boxes_exact'] for r in rows),
        frames_with_changed_scores=sum(not r['scores_exact'] for r in rows),common_identity_deltas=differences,
        common_identity_improved=sum(r['delta_XY']<0 for r in differences),
        common_identity_worsened=sum(r['delta_XY']>0 for r in differences),
        sensitivity4m_transitions=paired_counts(transitions4),paired_timing_delta_ms=timing_delta,
        lost_identity_figures=figures,
        caution='Rankwise box change does not identify object motion. Lost identity s111 is a fixed mask-recall boundary crossing (0.8288 to0.7972), not proof the object physically disappeared. No threshold adjustment.',
        risk='Original Calibration NoGo retained read-only; no risk threshold search or new risk qualification.'))
if __name__=='__main__':main()
