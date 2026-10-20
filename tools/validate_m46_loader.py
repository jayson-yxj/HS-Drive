"""Exercise real six-camera IO and ignored-label gradients without a network."""

import argparse
from pathlib import Path

import torch

from datasets.occupancy_training import OccupancyDataset
from simulator.recorder import atomic_json
from training.occupancy_loss import observed_occupancy_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        default="/root/gpufree-data/hs-drive-data/processed/m46_training_dense_index",
    )
    parser.add_argument("--output", default="reports/m46_loader_release.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    results = {}
    for split in ("train", "val"):
        data = OccupancyDataset(args.index, split)
        if not len(data):
            raise ValueError("Empty split")
        counts = torch.zeros(3, dtype=torch.int64)
        for item in data:
            state = item.targets["occupancy_state"][None]
            valid = item.targets["occupancy_valid"][None]
            logits = torch.zeros((1, 2, *state.shape[1:]), requires_grad=True)
            loss = observed_occupancy_loss(logits, state, valid)
            loss.backward()
            assert not logits.grad.permute(0, 2, 3, 4, 1)[~valid].any()
            assert logits.grad.permute(0, 2, 3, 4, 1)[valid].abs().sum() > 0
            counts += torch.bincount(state.flatten().long(), minlength=3)
        inference = OccupancyDataset(args.index, split, supervision=False)[0]
        assert inference.targets is None
        results[split] = dict(
            frames=len(data), state_counts=counts.tolist(), unknown_gradient_zero=True
        )
    atomic_json(
        args.output,
        dict(
            passed=True,
            index=args.index,
            results=results,
            network_trained=False,
            optimizer_created=False,
            scope="Actual RGB/label IO and masked 2-class CE gradient on throwaway zero logits; no network.",
        ),
    )


if __name__ == "__main__":
    main()
