"""One frozen Torch2.0/cu118 + MMCV1 route, old detection framework preserved."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
from tools.m8a_p1_common import ROOT,LOG,CFG,run,environment,provenance,write,asset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--step",choices=["bootstrap","torch","sources"],required=True)
    args=parser.parse_args()
    python=str(ROOT/"env/bin/python")
    if args.step=="bootstrap":
        ROOT.mkdir(exist_ok=False)
        for sub in ["home","wheels","sources","wheel-cache","artifacts"]: (ROOT/sub).mkdir()
        write(LOG/"asset_freeze.json",dict(**provenance("ReadOnlyAssetReverification"),checkpoint=asset(CFG["checkpoint"])))
        run("create_environment",["/root/gpufree-data/hs-drive-experiments/m6a-petr/env/bin/python","-m","venv",str(ROOT/"env")])
        run("bootstrap", [python,"-m","pip","install","--index-url","https://pypi.org/simple",
             "--timeout","30","--retries","1","pip==24.3.1","setuptools==59.5.0","wheel==0.38.4","ninja==1.11.1.1"],network=True)
    elif args.step=="torch":
        run("torch_wheels",[python,"-m","pip","download","--dest",str(ROOT/"wheels"),
            "--index-url","https://download.pytorch.org/whl/cu118","--timeout","60","--retries","1",
            "torch==2.0.0+cu118","torchvision==0.15.1+cu118","numpy==1.19.5"],network=True,timeout=1500)
        run("torch_install",[python,"-m","pip","install","--no-index","--find-links",str(ROOT/"wheels"),
                            "torch==2.0.0+cu118","torchvision==0.15.1+cu118","numpy==1.19.5"],timeout=600)
    else:
        old=Path("/root/gpufree-data/hs-drive-experiments/m6a-petr/source")
        run("petr_clone",["git","clone","--no-hardlinks",str(old),str(ROOT/"sources/PETR")])
        run("petr_checkout",["git","checkout",CFG["official_commit"]],cwd=ROOT/"sources/PETR")
        run("mmdet3d_clone",["git","clone","--depth","1","--branch","v0.17.1",
            "https://github.com/open-mmlab/mmdetection3d.git",str(ROOT/"sources/mmdetection3d")],network=True,timeout=300)
        (ROOT/"sources/PETR/mmdetection3d").symlink_to(ROOT/"sources/mmdetection3d",target_is_directory=True)
        run("legacy_framework_wheels",[python,"-m","pip","download","--no-deps","--dest",str(ROOT/"wheels"),
            "--index-url","https://pypi.org/simple","--timeout","30","--retries","1",
            CFG["single_route"]["mmcv_wheel"],"mmdet==2.24.1","mmsegmentation==0.20.2"],network=True,timeout=600)
        run("legacy_framework_install",[python,"-m","pip","install","--no-deps","--no-index",
            "--find-links",str(ROOT/"wheels"),"mmcv-full==1.7.2","mmdet==2.24.1","mmsegmentation==0.20.2"])


if __name__=="__main__":
    LOG.mkdir(exist_ok=True)
    main()
