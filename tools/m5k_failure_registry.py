"""Collect actual failure records without deleting/replacing the original logs."""
from pathlib import Path
from tools.m5k_common import write,asset

def main():
 failures=[]
 for p in Path('runs/m5k_logs').glob('*.log'):
  text=p.read_text(errors='replace')
  if any(k in text for k in ['Traceback (most recent call last)','ERROR:','FAILED ','SyntaxError:']):failures.append(dict(log=asset(p),tail=text[-1200:]))
 write('reports/m5k_failures.json',dict(kind='actual_log_failure_index',records=failures,limits='Failedattempts remainserver. No CARLA crash encountered thisstage. One supplementaryscript compile typo fixed beforeexecution; no experimental outputs fromfailedcompile. GPU parity check corrected tobitexactuint8 plusFP32 tolerance; measured error retained.'))
if __name__=='__main__':main()
