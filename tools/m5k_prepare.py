"""Predeclare conditional experiment gates and freeze historical file identities."""
import json,subprocess
from pathlib import Path
from tools.m5k_common import write,asset
from simulator.recorder import file_hash

def main():
    base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert base.startswith('db03730')
    old=json.loads(Path('reports/m5j_baseline_freeze.json').read_text());last=json.loads(Path('reports/m5j_artifact_manifest.json').read_text());assets=dict(old['assets']);assets.update({p:r['sha256']for p,r in last['new_assets'].items()})
    tracked={p:file_hash(p)for p in subprocess.check_output(['git','ls-files'],text=True).splitlines()}
    bad=[p for p,h in assets.items()if file_hash(p)!=h];assert not bad,bad
    write('runs/m5k_logs/baseline_files.json',dict(base=base,tracked=tracked,assets=assets))
    write('reports/m5k_baseline_freeze.json',dict(base=base,tracked=len(tracked),historical_assets=len(assets),mismatches=bad,index=asset('runs/m5k_logs/baseline_files.json')))
    write('configs/m5k_protocol.json',dict(stage='M5-K',base=base,seeds=[101,102],threshold=.2,roi_threshold=.9,grid=[4,160,120],depth_bins=48,default_changes=False,
      alignment=dict(first_experiment=True,methods=['original_p8','native16_to_p8','replay101_p8','replay102_p8','oracle_p8','native16'],interpolation='bilinear align_corners=False border replicated; native rounded UV8+16i to P8 UV4+8i; normalize D',same_mask=True,clear_gain='both seeds P gain>=.01,R1 gain>=.05,FPR increase<=.001 and Actor detection not lower on historical and new Val'),
      support=dict(each_train_and_val=True,independent_positions=4,strict_actor_rays_20_35=100,classes_with_at_least_10_strict_rays=2,strict_mask='existing semantic/instance interior +finite3x3 opticalZ range<=.5m; GTinBEV; no relaxation'),
      residual=dict(conditional='Only if alignment insufficient AND both Train/Val support pass',arms=['frozen','no_mobile','mobile'],steps=600,seeds=[101,102],lr=.001,capacity='shared identical inputchannel; no_mobile foreground channel zero',loss='CE balanced near/far Actor and background + .25 negative log same-layer1m probability where candidates exist +.5 M4 near replay CE +.1 KL original near',selection='step200/400/600 Val loss; fixedthreshold.2; stop at budget'),
      qualification=dict(precision_gain=.01,recall1_gain=.05,max_fpr_increase=.001,actor_detection_loss=0,near_mae_max_ratio=1.05,global_abs_bias_increase=.25,band_abs_bias_increase=.5,near_actor_space_loss=0,both_seeds=True),
      sampling=dict(base='P8 UV4',full='S4 UV2',adaptive='P8 plus S4 where frozenMobile>=.9, all six cameras',uniform='equal extra count deterministic evenly spaced S4 ray indices',denominator='all native reliable target reference; includes ROI missed Actors',go='bothseeds adaptiveR1>uniformR1 and FPRincrease<=.001',no_s4_network=True),
      split=dict(dev_train=4,dev_val=4,sealed_test=4,min_prior_distance_m=45,capture_test_after_freeze=True),
      input_pipeline=dict(deadline_s=.25,no_vehicle_control=True),limits='sourceA non-target not Free; B endpointcoarsecell mismatch not physicalFP; Eunknown separate; bin_count != unique rays; no GT model input; no historical modifications; no Occupancy/Temporal/SparseQuery training'))
if __name__=='__main__':main()
