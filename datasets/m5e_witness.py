"""Gold remains exactly M5-D. Candidate geometry never enters this function."""
from datasets.m5d_witness import certified_witness


def gold_witness(*args,**kwargs):
    return certified_witness(*args,**kwargs)
