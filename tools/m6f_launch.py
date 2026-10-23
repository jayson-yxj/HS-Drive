"""Freeze actual source bytes at import, not a later on-disk approximation."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import marshal
from pathlib import Path
import subprocess
import sys

LOADED = {__name__: dict(path=__file__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), loaded_code_sha256=hashlib.sha256(marshal.dumps(sys._getframe().f_code)).hexdigest())}
ROOT = Path.cwd().resolve()


class CapturedSource(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        raw = Path(self.path).read_bytes()
        code = self.source_to_code(raw, self.path)
        LOADED[fullname] = dict(path=self.path, source_sha256=hashlib.sha256(raw).hexdigest(),
                               loaded_code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest())
        return code


class CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            p = Path(spec.origin).resolve()
            if ROOT in p.parents and '.venv' not in str(p):
                spec.loader = CapturedSource(fullname, str(p))
        return spec


def inventory():
    external = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file() and name not in LOADED:
            p = Path(filename)
            external[name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  assurance='external_disk_at_boundary_not_memory_image')
    return dict(project_imports=LOADED.copy(), external_modules=external,
                python=sys.version, executable=sys.executable,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())


def verify_loaded():
    for name, item in LOADED.items():
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['source_sha256']:
            raise RuntimeError('Loaded source changed during batch: '+name)


def main():
    if subprocess.check_output(['git','diff','--name-only']).strip():
        raise RuntimeError('Commit tracked edits before a batch')
    sys.modules['tools.m6f_launch'] = sys.modules[__name__]
    sys.meta_path.insert(0, CaptureFinder())
    from tools.m6f_probe import main as probe
    probe()


if __name__=='__main__':
    main()
