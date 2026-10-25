"""Repeatable read-only M8-F and historical SHA verification."""
from pathlib import Path
from tools.m8e_verify import verify

def main():
 for name in ['m8b_release_manifest','m8b_evaluation_reference_audit','m8c_release_manifest','m8d_release_manifest','m8e_release_manifest','m8f_release_manifest']:
  print(name,verify(Path('reports')/(name+'.json')),'SHA verified',flush=True)
if __name__=='__main__':main()
