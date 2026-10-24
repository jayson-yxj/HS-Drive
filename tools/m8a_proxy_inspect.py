"""CPU weights-only checkpoint inspection; launched ONLY inside bubblewrap."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


EXPECTED_CLASSES = ["car","truck","construction_vehicle","bus","trailer","barrier",
                    "motorcycle","bicycle","pedestrian","traffic_cone"]


def shape_contract(shapes, classes):
    expected = {
        "img_backbone.conv1.weight": [64,3,7,7],
        "pts_bbox_head.input_proj.weight": [256,2048,1,1],
        "pts_bbox_head.reference_points.weight": [900,3],
        "pts_bbox_head.position_encoder.0.weight": [1024,192,1,1],
        "pts_bbox_head.position_encoder.2.weight": [256,1024,1,1],
        "pts_bbox_head.query_embedding.0.weight": [256,384],
        "pts_bbox_head.query_embedding.2.weight": [256,256],
    }
    for layer in range(6):
        expected[f"pts_bbox_head.cls_branches.{layer}.6.weight"] = [10,256]
        expected[f"pts_bbox_head.cls_branches.{layer}.6.bias"] = [10]
        expected[f"pts_bbox_head.reg_branches.{layer}.4.weight"] = [10,256]
        expected[f"pts_bbox_head.reg_branches.{layer}.4.bias"] = [10]
    mismatches = [{"key": k, "expected": v, "actual": shapes.get(k)}
                  for k,v in expected.items() if shapes.get(k) != v]
    decoder = sorted({int(m.group(1)) for k in shapes
                      if (m:=re.search(r"pts_bbox_head\.transformer\.decoder\.layers\.(\d+)\.",k))})
    dcn_keys = [k for k in shapes if k.startswith("img_backbone.") and ".conv_offset." in k]
    return dict(expected_shapes=expected, mismatches=mismatches,
        decoder_layers_found=decoder, dcn_offset_tensor_keys=dcn_keys,
        classes_match=list(classes or []) == EXPECTED_CLASSES,
        required_detection_groups_present=all(any(k.startswith(p) for k in shapes)
            for p in ["img_backbone.layer4.","pts_bbox_head.cls_branches.",
                      "pts_bbox_head.reg_branches.","pts_bbox_head.transformer."]),
        passed=not mismatches and decoder==list(range(6)) and bool(dcn_keys)
               and list(classes or [])==EXPECTED_CLASSES)


def main():
    # No user/site customizations, no GPU devices, no credentials or project mount.
    assert os.environ.get("M8A_ISOLATED") == "1"
    assert not Path("/root/.ssh").exists()
    assert not Path("/root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0").exists()
    assert not list(Path("/dev").glob("nvidia*"))
    network_blocked = False
    with socket.socket() as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(("1.1.1.1",443))
        except OSError as exc:
            network_blocked = True
            network_error = type(exc).__name__
    assert network_blocked, "Isolation must block networking"
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    obj = torch.load("/input/epoch_24.pth", map_location="cpu", weights_only=True)
    assert isinstance(obj,dict) and isinstance(obj.get("state_dict"),dict)
    weights = obj["state_dict"]
    shapes = {}
    dtypes = {}
    invalid = []
    for k,v in weights.items():
        assert isinstance(k,str) and torch.is_tensor(v), "Unexpected state_dict member"
        shapes[k] = list(v.shape)
        dtypes[k] = str(v.dtype)
        if v.is_floating_point() and not torch.isfinite(v).all().item():
            invalid.append(k)
    meta = obj.get("meta",{})
    config = meta.get("config","")
    config_sha = hashlib.sha256(config.encode()).hexdigest() if isinstance(config,str) else None
    contract = shape_contract(shapes,meta.get("CLASSES"))
    optimizer = obj.get("optimizer")
    # Preserve structure and counts, not entire optimizer or arbitrary metadata text.
    report = dict(kind="ActualCPUWeightsOnlyDeserializationInNetworkFilesystemIsolation",
        status="BasicCheckpointContractPassed" if contract["passed"] and not invalid else "CheckpointContractNoGo",
        torch_version=torch.__version__, cpu_only=True, weights_only=True,
        unrestricted_pickle_fallback=False, torch_cuda_initialized=torch.cuda.is_initialized(),
        top_level_keys=sorted(obj), tensor_count=len(weights),
        tensor_elements=sum(v.numel() for v in weights.values()),
        state_dict_shapes=shapes, state_dict_dtypes=dtypes, nonfinite_tensors=invalid,
        metadata=dict(keys=sorted(meta),epoch=meta.get("epoch"),iteration=meta.get("iter"),
                      classes=meta.get("CLASSES"),config_text_sha256=config_sha,
                      config_text_bytes=len(config.encode()) if isinstance(config,str) else None),
        components={
            "DetectionCheckpoint":{"present":True,"backbone_tensor_count":sum(k.startswith("img_backbone.") for k in weights),
                "head_tensor_count":sum(k.startswith("pts_bbox_head.") for k in weights)},
            "BackboneInitializationWeights":{"standalone_substitute_used":False,
                "note":"Backbone tensors embedded in the full trained detector"},
            "OptimizerState":{"present":optimizer is not None,
                "keys":sorted(optimizer) if isinstance(optimizer,dict) else None,
                "state_entries":len(optimizer.get("state",{})) if isinstance(optimizer,dict) else None},
            "Metadata":{"present":bool(meta),"executed_as_code":False}},
        contract=contract, elapsed_s=time.monotonic()-started,
        isolation=dict(network_blocked=network_blocked,network_error=network_error,
            credentials_visible=False,project_visible=False,gpu_devices_visible=False,
            environment_keys=sorted(os.environ),runtime_read_only=True),
        limitations=["Basic shape/structure consistency is not strict full model load_state_dict.",
                    "No PETR model build, CUDA operator, or six-camera inference performed.",
                    "No independent published checkpoint hash available."])
    Path("/output/inspection.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","tensor_count","components","metadata","elapsed_s")}))


if __name__=="__main__":
    main()
