"""Official local-version model provenance and original ResNet18 lineage."""
import json,inspect,hashlib,urllib.request
from pathlib import Path
import torch,torchvision
from torchvision.models import MobileNet_V3_Large_Weights,mobilenet_v3_large
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_pretrained_provenance.json')
    if out.exists():raise FileExistsError(out)
    w=MobileNet_V3_Large_Weights.IMAGENET1K_V2;folder=Path('/root/gpufree-data/hs-drive-cache/weights/m5i');folder.mkdir(exist_ok=True);p=folder/w.url.split('/')[-1]
    if not p.exists():torch.hub.download_url_to_file(w.url,str(p),hash_prefix=p.stem.split('-')[-1])
    assert file_hash(p).startswith(p.stem.split('-')[-1]);model=mobilenet_v3_large(weights=None);model.load_state_dict(torch.load(p,weights_only=True));man=json.loads(Path('runs/m4_expanded_joint/manifest.json').read_text());old=Path(man['config']['pretrained_weights']);assert file_hash(old)==man['pretrained_sha256'];assert file_hash('runs/m4_expanded_joint/best.pt')==man['best_sha256']
    atomic_json(out,dict(kind='actual_installed_torchvision_and_weights_verification',torchvision=torchvision.__version__,candidate=dict(name='mobilenet_v3_large',weights='IMAGENET1K_V2',url=w.url,path=str(p),sha256=file_hash(p),parameters=sum(x.numel()for x in model.parameters()),feature_parameters=sum(x.numel()for x in model.features.parameters()),source_sha256=file_hash(inspect.getfile(mobilenet_v3_large)),official_top1=w.meta['_metrics']['ImageNet-1K']['acc@1']),original_lineage=dict(initialization='official ImageNet ResNet18 IMAGENET1K_V1 then M4 CARLA optimization; NOT random initialization',pretrained_path=str(old),pretrained_sha256=file_hash(old),m4_checkpoint_sha256=file_hash('runs/m4_expanded_joint/best.pt'),manifest_sha256=file_hash('runs/m4_expanded_joint/manifest.json'),config_sha256=file_hash('configs/m4_expanded_joint.yaml')),license=dict(code='BSD-3-Clause',code_url='https://github.com/pytorch/vision/blob/v0.23.0/LICENSE',weights_terms='Pretrained weights mayhave dataset-derived terms; researchuse only,no redistribution,no claimunrestrictedcommerciallicense',terms_url='https://github.com/pytorch/vision/blob/v0.23.0/docs/source/models.rst',model_url='https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.mobilenet_v3_large.html'),source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
