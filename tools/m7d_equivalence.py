"""Reexecute all 205/1662 frozen cases without modifying historical runner/files."""
import sys
from tools import m7c_equivalence as old
from tools.m7d_common import write, provenance


def main():
    old.write=lambda path,data:write('reports/m7d_historical_equivalence.json',{**data,'provenance':provenance(),'historical_runner':'tools.m7c_equivalence unchanged','scope':'All frozen candidate outputs; snapshot polygon optimization separately paired on real episodes'})
    sys.argv=['m7c_equivalence'];old.main()
if __name__=='__main__':main()
