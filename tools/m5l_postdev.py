"""Sequential shadow then scientific freeze and new sealed capture."""
import time
from pathlib import Path
from tools.m5l_development import run

def main():
 deadline=time.monotonic()+3600
 while not Path('reports/m5l_adaptive_regression.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Development incomplete')
  time.sleep(5)
 run(['tools.m5l_online_shadow'],'online_shadow.log')
 run(['tools.m5l_freeze'],'freeze.log')
 run(['tools.m5l_test_pipeline'],'test_pipeline.log')
 run(['tools.m5l_finalize'],'finalize.log')
if __name__=='__main__':main()
