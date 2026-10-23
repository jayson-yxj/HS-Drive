"""Reexecute historical seven only; prevent inherited report/profile writes."""
from tools import m7b_history_profile as original
from tools.m7d_common import write,provenance
class Completed(Exception):pass

def output(path,data):
    write('reports/m7d_historical_seven.json',{**data,'provenance':provenance()})
    raise Completed()

def main():
    original.write=output
    try:original.main()
    except Completed:pass
if __name__=='__main__':main()
