"""Freeze and evaluate after actual shadow completion, keeping failedattempts."""
import time
from pathlib import Path
from tools.m5l_development import run

def main():
 deadline=time.monotonic()+1800
 while not Path('reports/m5l_online_shadow.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Shadow incomplete')
  time.sleep(5)
 run(['tools.m5l_freeze'],'freeze.log')
 run(['tools.m5l_test_pipeline'],'test_pipeline.log')
 run(['tools.m5l_finalize'],'finalize.log')
 run(['tools.m5l_object_strata'],'object_strata.log')
 run(['tools.m5l_figures'],'figures.log')
if __name__=='__main__':main()
