"""Read-only M8-H1 and historical seal verification; no simulator/model imports."""
from pathlib import Path
from tools.m8e_verify import verify

def main():
 for name in ['m8g_release_manifest','m8h0_release_manifest','m8h1_release_manifest']:
  print(name,verify(Path('reports')/(name+'.json')),'SHA verified',flush=True)
if __name__=='__main__':main()
