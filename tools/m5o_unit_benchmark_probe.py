"""Small synthetic end-to-end evaluator probe, never reported as CARLA performance."""
import copy
import numpy as np
from diagnostics.m5o_risk import fit_calibration,apply_calibration,quality_gate,risk_report
from datasets.m5o_benchmark import METHODS
from tools.m5o_common import write,provenance


def main():
    candidates=[];actors=[];predictions={}
    for g in range(6):
        for i in range(20):
            key=f'{g}:{i}';pos=[5+i,0,1];err=.2+i*.03+g*.02
            candidates.append(dict(candidate_id=key,group=str(g),region=[10+i*4,20,20+i,30],pixel_area=300+i,q_mean=.91+i*.002))
            actors.append(dict(sample_id=key,camera='front',actor_id=i,group=str(g),candidate_id=key,detected=True,unique_association=True,box_center_ego=pos))
            predictions[key]=dict(positions={m:[pos[0]+err*(1 if m=='direct' else 2),0,1] for m in METHODS})
    cal,oof=fit_calibration(candidates,actors,predictions);apply_calibration(cal,candidates,predictions,oof);gate=quality_gate(actors,predictions,0);risk=risk_report(actors,predictions)
    assert gate['passed'];assert risk['n']==120;assert 0<=risk['interval_coverage']<=1
    assert not quality_gate(actors,predictions,1.1)['passed']
    write('reports/m5o_synthetic_pipeline.json',dict(kind='synthetic_unit_test_NOT_CARLA_performance',passed=True,gate=gate,risk=risk,provenance=provenance()))
if __name__=='__main__':main()
