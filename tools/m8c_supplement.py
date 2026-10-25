"""Additional stratification and review notes without changing frozen model/rules."""
import json
from collections import Counter
import numpy as np
from tools.m8c_common import *
def main():
    data=json.loads((ROOT/'association_ledger.json').read_text());rr=data['associations'];bins={}
    for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,1.0001)]:
        rows=[r for r in rr if r['views'] and lo<=max(v['instance_bbox_iou'] for v in r['views'])<hi]
        bins[str(lo)+'_'+str(hi)]=dict(count=len(rows),grade_counts=dict(Counter(r['grade'] for r in rows)),high_score_ge0_5=sum(r['score']>=.5 for r in rows),views=dict(Counter('multi' if r['visible_cameras']>1 else 'single' for r in rows)))
    strata={}
    for c in ['vehicle','person','cone']:
        for lo,hi in [(0,10),(10,20),(20,40)]:
            rows=[r for r in rr if r['type']==c and lo<=r['range_m']<hi]
            strata[c+'_'+str(lo)+'_'+str(hi)]=dict(known=len(rows),grades=dict(Counter(r['grade'] for r in rows)))
    framecounts=Counter(r['token'] for r in rr)
    write('reports/m8c_association_strata.json',dict(**provenance('OfflineFixedRuleStratification'),iou_bins=bins,class_distance=strata,high_score_pairs=[dict(token=r['token'],index=r['prediction_index'],score=r['score'],grade=r['grade'],views=r['views']) for r in rr if r['score'] is not None and r['score']>=.5],overlapping_actor_support=dict(multiactor_frames=sum(n>1 for n in framecounts.values()),known_targets_in_multiactor_frames=sum(n for n in framecounts.values() if n>1),ambiguous_count=sum(r['grade']=='AmbiguousAssociation' for r in rr),status='InsufficientSupport_NotZeroAssociationRisk'),no_pair=dict(all=sum(r['grade']=='NoSupportedAssociation' for r in rr),far=sum(r['grade']=='NoSupportedAssociation' and r['range_m']>=20 for r in rr)),limits='Mask competitor checks cover recorded known actors only; unrecorded static/dynamic objects can still create ambiguity. IoU computed versus tight nativeinstancebbox aftercrop, not oldOBBprojectionIoU.'))
    notes=[
      dict(sheet='identity_supported',observation='All six depict visible vehicle body in both views; s102 partially occluded by foregroundpole. Agreement supports identity association, but does not validate metriccenter. Do not upgrade beyond automatedgrade.'),
      dict(sheet='far_proxy',observation='Red boxes often offset vertically or cover only a part of car/person or broad background; s049 includes pottedplant and only smallvisiblemask. No image-only identity upgrade.'),
      dict(sheet='background_00',observation='Predictions on road, railing, wall, pole and plant; exact semantic/instance evidence retained. No hiddenobjectabsence certificate.'),
      dict(sheet='background_01',observation='s038q0/q1 visibly overlap a car despite failing natural3D match. Shows unmatcheddoesnotmeanabsence; no new3D errors.'),
      dict(sheet='background_02',observation='s040q13/q18 and s042q1 overlap visiblevehicles; several person hypotheses overlapvehicleparts or wall edges.'),
      dict(sheet='background_03',observation='Many road/pole/planter-associated hypotheses; barrier isunannotatedclass, cannot identifyphysicalboundaryfromboxalone.'),
      dict(sheet='background_04',observation='s046 person box offset above visibleperson, cars/background nearby; recognition vs location cannot be conflated.'),
      dict(sheet='background_05',observation='Vehicleparts and storefronts, repeated centers; no controlledphysicalFP proof.'),
      dict(sheet='background_06',observation='s075 boxes overlap vehicleback; s097 wideboxes on planter/storefront background.'),
      dict(sheet='background_07',observation='s098q17 includes visibleperson partly; otherboxes on facade/planter. No trustworthy3D identity from these rectangles.'),
      dict(sheet='background_08',observation='Window/facade/hedge and road surfaces dominate shown hypotheses; physicalabsence unverified.'),
      dict(sheet='background_09',observation='s146q0 nearcamera projection clipped by ego/body; s149 candidate shifted above car. Edgeclippingambiguity.'),
      dict(sheet='background_10',observation='s151q0/q1/q2 cover visiblevan, s155cone boxes shifted off cone; barrier predictions on roadside surfaces.'),
      dict(sheet='background_11',observation='s159van behind wall; s165 boxes include stationarybus/van-likeobjects absentfromrecorded insertedActor inventory. Backgroundtruthincomplete.'),
      dict(sheet='background_12',observation='Hedge/planter vs parkedvehicle ambiguity; image evidencenot sufficient to validate metricboxes.')]
    write('reports/m8c_visual_review.json',dict(**provenance('AssistantVisualInspection_NotCertifiedHumanAnnotation'),reviewer='AI assistant visual inspection of saved actual RGB sheets',sheets=[dict(**n,file=asset('reports/m8c_figures/'+n['sheet']+'.png')) for n in notes],background_candidates_viewed=154,identity_examples_viewed=6,far_proxy_examples_viewed=8,grade_changes=0,ConfirmedFalsePositive=0,unreviewed='No human double annotation; onlybestprojectionperbackgroundcandidate displayed, other viewstats available',decision='RetainUnknownForUnresolvedIdentity; requestfuturefullannotation review beforetraining'))
    # Check image-index encoding itself, not a full reordered-model ablation.
    import torch,sys
    sys.path.insert(0,str(RUNTIME/'sources/PETR'))
    import projects.mmdet3d_plugin
    from projects.mmdet3d_plugin.models.utils.positional_encoding import SinePositionalEncoding3D
    m=SinePositionalEncoding3D(num_feats=128,normalize=True);a=m(torch.zeros(1,6,16,44,dtype=torch.bool));permutation=[0,2,1,5,3,4]
    diffs=[]
    for oldindex,newindex in enumerate([0,2,1,4,5,3]):diffs.append(float((a[0,:,0,0][oldindex]-a[0,:,0,0][newindex]).abs().max()) if False else float((a[0,oldindex,:,0,0]-a[0,newindex,:,0,0]).abs().max()))
    write('reports/m8c_camera_index_encoding_test.json',dict(**provenance('SyntheticOfficialSineEncoding_NoFullNetworkAblation'),shape=list(a.shape),HS_to_official_index=[0,2,1,4,5,3],max_embedding_channel_difference_by_camera=diffs,expected_mapping_not_applied_to_GPU=True,source=asset(RUNTIME/'sources/PETR/projects/mmdet3d_plugin/models/utils/positional_encoding.py'),note='Only proves encoding inputdifference. Doesnot quantify causalperformance impact or establishactualtrainingPKLorder beyondofficialconverter.'))
    write('reports/m8c_dataset_support_decision.json',dict(**provenance('EvidenceBasedGateDecision'),status='TrainingReadinessNoGo',priority_order=['ObtaintraceableofficialnuScenescalibration+originaltraininginfoorder; reviewcameraindexcontract','Separatelyrequestonefactorcameraordersmoke/Valdiagnostic, no height orcropjointchanges','Completeannotation/identitysupport and rightside/far/occlusion coverage with independentnewpositions','Onlythenproposeboundedtrainingbudget'],observed=dict(frames=192,groups=38,known_objects=155,strict_identity_supported=6,strict_far_identity_supported=0,person_or_cone_strict_identity_supported=0,background_audit=154,new_blind_positions=0),gaps=['NoLIDAR_TOP/Camerarealcalibrationtable','NoempiricalnuScenesreferenceheight/FOV/targetdistribution','Cameraindexconventiondiscrepancyunablated','Noindependenthumanidentityannotation','Incompletebackgroundtaxonomy/objects','No farstrictmetriclocalizationreference','Rightfront12/rightrear4historysupport','Only1multiactorframe','CalibrationNoGo unchanged'],not_authorized=['training','defaultmodelreplacement','visualcontrol','secondreferenceheightsearch']))
if __name__=='__main__':main()
