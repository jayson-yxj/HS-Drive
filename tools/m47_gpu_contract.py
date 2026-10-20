"""GPU supervision-contract test only; quality-gated optimizer is not started."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from datasets.occupancy_training import OccupancyDataset
from datasets.inputs import INPUT_FIELDS
from training.occupancy_loss import observed_occupancy_loss
from training.runner import initialize,load_checkpoint
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="reports/m47_gpu_contract_verified.json")
    args=parser.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    prepare_cuda()
    torch.manual_seed(47);torch.set_num_threads(2)
    ds=OccupancyDataset("/root/gpufree-data/hs-drive-data/processed/m46_training_dense_index","train",supervision=False)
    item=ds[0];assert set(item.inputs)==INPUT_FIELDS and item.targets is None
    man=json.loads(Path("runs/m4_expanded_joint/manifest.json").read_text())
    model=initialize(man["config"],pretrained=False).cuda().eval()
    load_checkpoint(Path("runs/m4_expanded_joint/best.pt"),model)
    inputs={k:v[None].cuda() for k,v in item.inputs.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():features=model(**inputs)["bev_current"].detach()
    sample=item.metadata["sample_id"]
    scene,fid=sample.split("/")
    label=Path("/root/gpufree-data/hs-drive-data/processed/m47_quarantined")/scene/(fid+".npz")
    with np.load(label) as z:labels={k:torch.from_numpy(z[k]).cuda() for k in z.files}
    # Throwaway heads, one backward, no optimizer and no model update.
    head=torch.nn.Conv2d(128,8,1).cuda()
    logits=head(features).reshape(1,2,4,160,120);logits.retain_grad()
    loss=observed_occupancy_loss(logits,labels["occupancy_state"][None],labels["occupancy_valid"][None])
    loss.backward()
    valid=labels["occupancy_valid"][None]
    ignored_zero=not bool(logits.grad.permute(0,2,3,4,1)[~valid].any())
    semantic={}
    for kind,mask in [("ground",labels["surface_semantic_valid"]),("obstacle",labels["obstacle_valid"])]:
        pred=torch.zeros(4,160,120,device="cuda",requires_grad=True)
        truth=(labels["surface_semantic"]==1) if kind=="ground" else labels["obstacle_mask"]
        sl=F.binary_cross_entropy_with_logits(pred[mask],truth[mask].float());sl.backward()
        semantic[kind]=dict(valid=int(mask.sum()),positives=int(truth[mask].sum()),loss=float(sl.detach()),ignored_gradient_zero=not bool(pred.grad[~mask].any()))
    atomic_json(target,dict(kind="actual_gpu_rgb_forward_and_masked_backward",sample=sample,
        loss=float(loss.detach()),finite_gradients=bool(torch.isfinite(head.weight.grad).all()),unknown_gradient_zero=ignored_zero,
        semantic=semantic,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        checkpoint_sha256=file_hash("runs/m4_expanded_joint/best.pt"),label_sha256=file_hash(label),
        optimizer_created=False,overfit_executed=False,source_sha256=file_hash(__file__),
        reason="Independent reserved-scene spatial quality remains Partial. Conditional overfit phase deferred; backward is numerical/contract test, not learnability proof.",
        position_prior_comparison="Not executed because optimization phase was not started. No superiority claim."))


if __name__=="__main__":main()
