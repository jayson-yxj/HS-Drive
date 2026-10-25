"""Explicit stop decision, review, runtime provenance and independent statuses."""
import json
from pathlib import Path
from tools.m8f_common import *

def main():
 p=provenance('ActualCARLAStaticSmoke_OfflineAudits_SyntheticTests')
 counts={}
 for name,n in [('final_new_tests',17),('m8e_regression',18),('m8b_regression',18),('m8d_regression',16)]:
  log=LOG/(name+'.txt');s=log.read_text();assert 'Ran %d tests'%n in s and '\nOK\n' in s
  counts[name]=dict(count=n,log=asset(log))
 figures=[asset(x) for x in sorted(Path('reports/m8f_figures').glob('*.png'))]
 write('reports/m8f_annotation_review.json',dict(**p,review_type='AssistantVisualReview',human_reviewers=0,HumanDoubleVerified=False,
    reviewed_figures=figures,initial_choices=[
      dict(group='m8f_development_00',choice='No controlled targets spawned; retain all4 rejected requests',notes='Guardrails/trees dominate roadside. High canopy rejection is conservative, not proven body collision.'),
      dict(group='m8f_development_01',choice='Retain source/geometry support for one pedestrian, refuse independent training eligibility',notes='Small pedestrian on sidewalk appears Present/Shifted and absent controls. Exact foot contact not resolvable at65/30pixels.'),
      dict(figure='final_support_gaps.png',choice='GroundedByGeometryProxy only',notes='15 actual footprint samples near0.03m gap for both poses; physics disabled so static pose not proof of contact.')],
    no_manual_grade_override=True,conflicting_evidence=[
      'Historical apparent-floating cone at Town05 has actual elevated Roads hit8.1m in new query; neither confirms nor repairs historical placement.',
      'Trees above subject volume trigger strict CoveredOrWrongTopSurface; may be conservative false rejection.',
      'NoLaneAtFootprint can reflect OpenDRIVE/render boundary mismatch; not proven exact physical road violation.']))
 failures=[json.loads(x.read_text()) for x in LOG.glob('*.json') if json.loads(x.read_text()).get('returncode',0)!=0]
 write('reports/m8f_failure_ledger.json',dict(**p,failed_processes=failures,
    fixes=['NumPy2 removed ndarray.ptp; changed to np.ptp. Failed preflight artifact and executed source retained; no RGB produced by failed run.',
           'Common provenance expanded to include simulator/m8f source. Initial and final helper source versions retained.'],
    quality_failures=['Smoke00 has no accepted subject','Smoke01 has no accepted vehicle/cone/barrier','Seven planned targets rejected before spawn','No independent final region','Sparse single-camera far support','Static background classes often Unknown/Ignore'],
    scope_limits=['Historical index initially uses config/supervision/traces/M8E; further6190 raw M5 frames recovered after smoke. Cannot call initial preflight exhaustive.',
      'Development ego poses chosen for geometry debugging despite overlapping historical envelopes; explicitly not final/independent training.',
      'No real multiple-road-layer acceptance stress found in selected smoke; synthetic two-layer rejection tested. Actual water/road-height probes separately recorded.',
      '15point finite footprint samples and OBB overlap are geometric proxies, not continuous support/mesh collision.',
      'No multi-subject accepted scene observed; interaction support unverified.'],
    stopped_after_frames=8,new_training_steps=0))
 isolation=json.loads((LOG/'input_isolation.txt').read_text())
 write('reports/m8f_input_isolation.json',dict(**p,result=isolation,new_model_forward='NotRun',contract='RGB-only packet whitelist, Python file allowlist and unshare network namespace; not hostile native-code sandbox'))
 write('reports/m8f_acceptance.json',dict(**p,overall='Implemented_Partial_SmokeNoGo',
    decisions=dict(A_GroundPlacement='NoFallbackImplemented_ProxyValidatedOneSubject',
      B_NaturalScene='SmokeNoGo',C_RenderIdentity='SingleViewSupported_MultiViewNotDemonstrated',
      D_Background='RegionLimitedIdentityPartial',E_FarField='InsufficientSupport_1NaturalProxy_0IndependentEligible',
      F_Independence='HeldoutIndependenceNoGo',G_TrainingReadiness='TrainingReadinessNoGo',
      DataContract='8Frames_192SensorRecordsVerified',Perception='NotValidated',Safety='NotValidated',Occupancy='NotValidated',Runtime='HistoricalPartialUnchanged'),
    actual_execution=dict(CARLA='0.9.15',GPU='RTX4090_rendering_only',maps_with_RGB=['Town02'],geometry_only_probe_maps=['Town04','Town05'],
      location_groups=2,six_camera_frames=8,rgb_images=48,sensor_payloads=192,
      planned_targets=8,spawned_targets=1,rejected_before_spawn=7,actual_support_poses=2,
      planned_support_rays=240,final_support_rays=30,natural_render_far_targets=1,independent_eligible_far=0,
      validation_frames=0,calibration_frames=0,final_frames=0,new_PETR_forwards=0,training_steps=0,closed_loop_driving=0),
    tests=counts,total_tests=69,additional_probes=dict(input_packets=8,GT_field_injections_rejected=8,forbidden_reads_rejected=3),
    history=dict(M8E_original_targets=80,M8E_single=28,M8E_multi=18,M8E_unknown=34,M8E_far_render=21,unchanged=True),
    next_step='No automatic expansion. Diagnose lane boundary queries, unknown collision geometry and body-height-aware clearance in a separate bounded development proposal. Complete precollection raw historical footprint index before any new split.',
    default_model_or_control_changes=False,merge_main=False))
 print('69 tests; smoke stop, no training/final',flush=True)
if __name__=='__main__':main()
