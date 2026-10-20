"""V2 sidecar adapter. Inference delegates to unchanged RGB-only loader."""
import json
from pathlib import Path
import numpy as np
import torch
from datasets.occupancy_training import OccupancyDataset,checked
from datasets.training import DataItem
from datasets.m48_evidence import validate_evidence

class EvidenceDataset(OccupancyDataset):
    def __init__(self,index,split,labels_report=None,*,supervision=True):
        super().__init__(index,split,supervision=False)
        self.evidence_supervision=supervision
        self.labels={}
        if supervision:
            report=json.loads(Path(labels_report).read_text())
            self.labels={f"{r['scene']}/{r['frame_id']:08d}":r for r in report['records']}
    def __getitem__(self,i):
        item=super().__getitem__(i)
        if not self.evidence_supervision:return item
        r=self.labels[item.metadata['sample_id']]
        with np.load(checked(r['label'],r['label_sha256']),allow_pickle=False) as z:d={k:z[k] for k in z.files}
        validate_evidence(d)
        return DataItem(item.inputs,{k:torch.from_numpy(v) for k,v in d.items()},item.metadata)
