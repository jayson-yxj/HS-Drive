"""Frozen historical seven rerun with append-only output redirection."""
from tools import m7b_history_profile as original
from tools.m7e_common import write,provenance
class Completed(Exception):pass

def output(path,data):
    write('reports/m7e_historical_seven.json',{**data,'provenance':provenance()});raise Completed()

def main():
    original.write=output
    try:original.main()
    except Completed:pass
if __name__=='__main__':main()
