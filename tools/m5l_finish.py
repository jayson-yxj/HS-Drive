"""Wait for sealed scientific report, append diagnostics and verify historical assets."""
import time
from pathlib import Path
from tools.m5l_development import run

def main():
 deadline=time.monotonic()+3600
 while not Path('reports/m5l_visualizations.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Sealed pipeline unfinished')
  time.sleep(5)
 run(['tools.m5l_post_audit'],'post_audit.log')
 run(['tools.m5l_delivery_audit'],'delivery_audit.log')
 run(['tools.m5l_integrity'],'integrity.log')
if __name__=='__main__':main()
