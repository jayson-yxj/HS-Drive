"""Make a private official-source copy with only optional Flash imports disabled."""
import difflib,hashlib,json,shutil
from pathlib import Path
c=json.loads(Path('configs/m8j11_protocol.json').read_text());r=Path(c['runtime_root']);src=r/'official-source';dst=r/'runtime-source'
assert not dst.exists();shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git'))
(dst/'mmdetection3d').symlink_to(r/'deps/mmdetection3d-47285b3f1e9dba358e98fcd12e523cfd0769c876',target_is_directory=True)
changes={
 'projects/mmdet3d_plugin/models/utils/petr_transformer.py':('from .attention import FlashMHA','try:\n    from .attention import FlashMHA\nexcept ImportError:\n    FlashMHA = None  # Official optional backend; FP32 uses PETRMultiheadAttention.'),
 'projects/mmdet3d_plugin/models/backbones/__init__.py':('from .eva_vit import EVAViT','# Optional unrelated EVA/Flash backbone is not imported for fixed ResNet50.')}
diff=[]
for rel,(before,after) in changes.items():
 p=dst/rel;old=p.read_text();assert old.count(before)==1;new=old.replace(before,after);p.write_text(new)
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='official/'+rel,tofile='isolated/'+rel))
Path('reports/m8j11_sources/optional_flash_imports.patch').write_text(''.join(diff))
print(dst)
