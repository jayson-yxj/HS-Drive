"""Add independent eligibility fields to every observed reference, including static unknowns."""
import json
from pathlib import Path
from collections import Counter
from datasets.m8f_quality import eligibility
from tools.m8f_common import *

def main():
 out=ROOT/'certified_references';out.mkdir(exist_ok=False);files=[];counts=Counter()
 for path in sorted((ROOT/'audit').glob('*.json')):
  a=json.loads(path.read_text());rows=[]
  for f in a['frames']:
   side=json.loads(Path(f['sidecar']['path']).read_text());support=json.loads(Path(side['ground_support']['path']).read_text())['checks'];byid={x['entity_id']:x for x in support}
   for obj in f['objects']:
    c=byid.get(obj['entity_id']) if obj['source']=='Actor' else None
    ok=c is not None and c['final']['status']=='GroundedByGeometryProxy'
    grades=dict(RenderIdentityGrade=obj['identity_grade'],GroundSurfaceGrade='GroundedByGeometryProxy' if ok else 'GroundSupportUnknown',NaturalPlacementGrade='NaturalByGeometryProxy' if ok else 'Unknown',AnnotationGrade='SourceContractComplete' if obj['frame_consistent'] and obj['petr_class'] not in ['Unknown','Ignore'] else 'UnknownOrIgnoredTaxonomy',SplitIndependenceGrade='HistoricalCoverageIncomplete',InputSourceGrade='CompleteSynchronized')
    row=dict(**obj,**grades,TrainingEligibility=eligibility(grades));rows.append(row);counts[row['TrainingEligibility']['status']]+=1
  p=out/(a['group']+'.json');write(p,dict(**provenance('OfflinePerReferenceIndependentCertificates'),objects=rows,source_audit=asset(path)));files.append(asset(p))
 write('reports/m8f_reference_certificates.json',dict(**provenance('AllObservedReferenceCertificateIndex'),files=files,counts=dict(counts),unknown_static_support_not_promoted=True))
 print(dict(counts))
if __name__=='__main__':main()
