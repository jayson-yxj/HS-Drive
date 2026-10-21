"""Report independent position units and empty-layout backgrounds without reselection."""
import json
from pathlib import Path
import numpy as np
from tools.m5f_common import pixel_label
from diagnostics.m5f_metrics import histogram,metrics
from simulator.recorder import atomic_json,file_hash


def aggregate(pairs):
    n=sum(p['n'] for p in pairs);nr=sum(p['removed_n'] for p in pairs)
    def avg(key,count='n'):
        den=sum(p[count] for p in pairs);return sum((p[key] or 0)*p[count] for p in pairs)/den if den else None
    return dict(n=n,removed_n=nr,mean_rise=avg('mean_rise'),mean_present=avg('mean_present'),mean_empty=avg('mean_empty'),mean_removed=avg('mean_removed','removed_n'),rise_direction=avg('direction_rise'),fall_direction=avg('direction_fall','removed_n'),both_correct=sum(p['both_correct'] for p in pairs)/n if n else None,removed_fpr=sum(p['removed_fp'] for p in pairs)/nr if nr else None)


def main():
    out=Path('reports/m5f_pair_scene_audit.json')
    if out.exists():raise FileExistsError(out)
    rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records'];study=json.loads(Path('reports/m5f_2d_presence_absence.json').read_text())['records'];records=[]
    for model in study:
        groups=[]
        for group in sorted({r['group'] for r in rows}):
            hist=np.zeros((2,256),np.int64);background_target=0;unknown=0
            for row in [r for r in rows if r['group']==group and r['configuration'] in [0,11]]:
                d=pixel_label(row);p=next(p for p in model['predictions'] if p['sample_id']==row['sample_id']);score=np.load(p['path'])['probability_u8']/256;m=d['valid']&~d['target'];hist+=histogram(score,d['target'],m);background_target+=int(d['target'].sum());unknown+=int((~d['valid']).sum())
            groups.append(dict(group=group,paired=aggregate([p for p in model['pairs'] if p['group']==group]),empty_layout_known_nontarget=metrics(hist,model['threshold']),background_target_pixels_not_negative=background_target,unknown_pixels_not_negative=unknown))
        records.append(dict(name=model['name'],seed=model['seed'],groups=groups,macro_mean_rise=float(np.mean([g['paired']['mean_rise'] for g in groups if g['paired']['n']])),positive_mean_rise_groups=sum((g['paired']['mean_rise']or 0)>0 for g in groups)))
    atomic_json(out,dict(kind='offline_position_level_frozen_2D_predictions',records=records,source_sha256=file_hash(__file__),limits='Emptylayout means introducedactorsremoved; parked/backgroundtargets areexcluded fromnegative, unknownexcluded. Originalempty metric includedonlygloballytargetfreeframes; thissupplement coversall3groups withoutmislabellingbackgroundcars. Threeindependentpositions, notpixelstatisticalsignificance.'))

if __name__=='__main__':main()
