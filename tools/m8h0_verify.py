"""Read-only evidence verification; no CARLA/actor/model imports."""
import json
from pathlib import Path
from tools.m8e_verify import verify
from tools.m8a_common import sha
from tools.m8h0_common import HISTORY

def main():
 for n in ['m8g_release_manifest','m8h0_release_manifest']:
  print(n,verify(Path('reports')/(n+'.json')),'sealed assets verified',flush=True)
 d=json.loads(HISTORY.read_text())
 for a in d['historical_sources']:assert sha(a['path'])==a['sha256'],a['path']
 print(len(d['historical_sources']),'historical source SHA verified; no simulation queries',flush=True)
if __name__=='__main__':main()
