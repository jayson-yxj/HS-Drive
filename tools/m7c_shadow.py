"""Byte/schema regression only; no repeated model forward."""
from pathlib import Path
from tools.m7c_common import *
from datasets.m7a_shadow import validate_packet
from datasets.m7a_contracts import Unknown,reject_control_export

def main():
 records=[]
 for p in sorted(Path('reports/m7a_shadow_packets').glob('*.json')):
  d=read(p);validate_packet(d,d['timestamp_s'],verify_files=True);records.append(asset(p))
 blocked=False
 try:reject_control_export(Unknown('insufficient_evidence'))
 except TypeError:blocked=True
 assert blocked and not Unknown('not_observed').safety_free
 write('reports/m7c_shadow_regression.json',dict(kind='ActualServerHistoricalPacketContract',provenance=provenance(),count=len(records),packets=records,model='CandidateNoGo',GT='EvaluatorOnly_NoNewInference',Unknown='NotFree',control_export_rejected=True,strict20Hz='RuntimePartial',Bench2Drive='PreparationOnly_NoNewExecution',PETR='NotRun',new_GPU_forwards=0,training_steps=0))
 print('shadow',len(records),'packets verified')
if __name__=='__main__':main()
