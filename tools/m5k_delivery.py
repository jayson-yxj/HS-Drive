"""Finish postfreeze descriptive reporting only after sealed run completes."""
import time
from pathlib import Path
from tools.m5k_dev_pipeline import run

def main():
 deadline=time.monotonic()+3600
 while not Path('reports/m5k_sampling_test.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Sealed run incomplete')
  time.sleep(5)
 run(['tools.m5k_details','--split','test'],'details_test.log')
 run(['tools.m5k_supplement'],'supplement.log')
 run(['tools.m5k_figures'],'figures.log')
 run(['tools.m5k_finalize'],'finalize.log')
if __name__=='__main__':main()
