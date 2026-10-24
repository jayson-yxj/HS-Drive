"""Auditable, build/version-only patches; no detector math changes."""
from pathlib import Path
import difflib
from tools.m8a_p1_common import ROOT,LOG,write,asset,provenance


def main():
    records=[];diffs=[]
    def patch(path,old,new,reason):
        before=path.read_text();assert old in before,(path,old)
        old_sha=asset(path)['sha256'];after=before.replace(old,new)
        diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(path),tofile=str(path)))
        path.write_text(after);records.append(dict(path=str(path),before_sha256=old_sha,after=asset(path),reason=reason))
    src=ROOT/'sources/mmdetection3d'
    site=ROOT/'env/lib/python3.8/site-packages'
    for path in [src/'mmdet3d/__init__.py',site/'mmdet/__init__.py',site/'mmseg/__init__.py']:
        text=path.read_text()
        import re
        pattern=r"(?:mmcv_maximum_version|MMCV_MAX) = '[^']+'"
        matches=re.findall(pattern,text);assert len(matches)==1,(path,matches)
        patch(path,matches[0],matches[0].split('=')[0]+"= '1.7.2'",'Permit frozen MMCV1.7.2 version; no fake version or assertion removal; real CUDA and model tests required')
    for path in sorted((src/'mmdet3d/ops').rglob('*.cpp')):
        if '#include <THC/THC.h>' in path.read_text():
            # All uses were unused include + unused extern; stream already uses ATen CUDAContext.
            patch(path,'#include <THC/THC.h>\n','', 'Remove obsolete unused THC header; ATen CUDAContext and kernels unchanged')
            patch(path,'extern THCState *state;\n','', 'Remove unused extern; not referenced by implementation')
    patch(src/'requirements/runtime.txt','numba==0.48.0','numba==0.53.1','Frozen Python3.8 compatible numba runtime; no PETR computation change')
    (LOG/'compatibility.patch').write_text(''.join(diffs))
    write(LOG/'compatibility_patches.json',dict(**provenance('ExplicitIsolatedCompatibilityPatches'),records=records,patch=asset(LOG/'compatibility.patch')))

if __name__=='__main__':main()
