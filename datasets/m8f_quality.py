"""Pure geometric proxy and conjunctive qualification; no simulator/model imports."""
import numpy as np

def polygon(center,extent,yaw):
 c,s=np.cos(yaw),np.sin(yaw)
 return np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array(extent) @ np.array([[c,s],[-s,c]])+center

def polygon_distance(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float)
 separated=False
 for p in [a,b]:
  for e in np.roll(p,-1,axis=0)-p:
   n=np.array([-e[1],e[0]]);x=a@n;y=b@n
   if x.max()<y.min() or y.max()<x.min():separated=True
 if not separated:return 0.
 def point_edge(p,q,r):
  d=r-q;t=np.clip(np.dot(p-q,d)/max(np.dot(d,d),1e-12),0,1)
  return float(np.linalg.norm(p-(q+t*d)))
 return min(point_edge(p,q,r) for x,y in [(a,b),(b,a)] for p in x for q,r in zip(y,np.roll(y,-1,axis=0)))

def choose_surface(hits,expected_z,allowed,contract):
 reasons=[]
 if not hits:return dict(status='GroundSupportUnknown',reasons=['NoRayHit'],selected=None)
 relevant=[h for h in hits if abs(h['xyz'][2]-expected_z)<=contract['layer_window_m']]
 if any(h['semantic']==23 for h in relevant):reasons.append('WaterSurface')
 allowed_hits=[h for h in hits if h['semantic'] in allowed]
 levels=[]
 for h in sorted(allowed_hits,key=lambda x:x['xyz'][2]):
  if not levels or abs(h['xyz'][2]-levels[-1])>contract['distinct_layer_m']:levels.append(h['xyz'][2])
 if len(levels)>1:reasons.append('MultipleGroundLayers')
 close=[h for h in allowed_hits if abs(h['xyz'][2]-expected_z)<=contract['waypoint_z_tolerance_m']]
 if not close:reasons.append('NoCompatibleSurfaceAtDeclaredRoadLayer')
 selected=min(close,key=lambda h:abs(h['xyz'][2]-expected_z)) if close else None
 if selected and any(h['semantic'] not in allowed and h['xyz'][2]>selected['xyz'][2]+.1 for h in hits):reasons.append('CoveredOrWrongTopSurface')
 return dict(status='Rejected' if reasons else 'GroundedByGeometryProxy',reasons=reasons,selected=selected,ground_levels=levels,selection_reason='Unique permitted semantic layer within frozen waypoint-Z tolerance; no fallback')

def eligibility(grades):
 required={'RenderIdentityGrade':{'RenderLinkedSingleView','RenderLinkedMultiView'},'GroundSurfaceGrade':{'GroundedByGeometryProxy'},'NaturalPlacementGrade':{'NaturalByGeometryProxy'},'AnnotationGrade':{'SourceContractComplete'},'SplitIndependenceGrade':{'SpatiallySeparated'},'InputSourceGrade':{'CompleteSynchronized'}}
 failures=[k for k,v in required.items() if grades.get(k) not in v]
 return dict(status='EligibleForProposalOnly' if not failures else 'NotEligible',failed_fields=failures,training_authorized=False)
