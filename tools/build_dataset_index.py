"""Validate raw scenes and atomically build a versioned M3 dataset index."""
import argparse
from datasets.index import build_index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',default='configs/dataset_v1.yaml')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    index=build_index(args.registry,args.output)
    print(f"Committed {len(index['samples'])} samples in {len(index['scenes'])} scenes: {args.output}")


if __name__=='__main__':main()
