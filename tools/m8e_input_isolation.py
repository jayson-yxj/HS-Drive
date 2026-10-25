"""RGB input contract probe, without loading or executing a model.

Run in a network namespace. Python audit hooks detect accidental file access;
they are not a security boundary against hostile native code.
"""
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from datasets.m8b_petr_adapter import PETRCarlaOfflineAdapter


def main():
    capture = Path(sys.argv[1]).resolve()
    packets = sorted(capture.glob('*/rgb/*_packet.json'))
    # Discover only the RGB directory; never enumerate evaluator contents.
    permitted = {p.resolve() for p in packets}
    permitted.update(p.resolve() for p in capture.glob('*/rgb/*.png'))
    blocked = []
    project = Path.cwd()

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        protected = (capture in path.parents or project/'reports' in path.parents
                     or project/'runs' in path.parents
                     or str(path).startswith('/root/gpufree-data/hs-drive-data')
                     or path.name in {'evaluation_index.json', 'evaluation_ledger.json', 'association_ledger.json'})
        if protected and path not in permitted:
            blocked.append(str(path))
            raise PermissionError('Evaluator path forbidden in RGB probe')

    sys.addaudithook(guard)
    canaries = [packets[0].parent.parent/'evaluator'/'present_sidecar.json',
                project/'reports'/'m8e_actor_environment_gt.json',
                Path('/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot/evaluation_index.json')]
    for path in canaries:
        try:
            path.read_bytes()
        except PermissionError:
            continue
        raise AssertionError('Isolation canary not blocked')
    adapter = PETRCarlaOfflineAdapter(capture)
    images = 0
    for path in packets:
        packet = json.loads(path.read_text())
        adapter.validate(packet)
        for camera in packet['cameras']:
            blob = Path(camera['path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == camera['sha256']
            assert np.asarray(Image.open(io.BytesIO(blob))).shape == (384, 640, 3)
            images += 1
        injected = dict(packet, actor_id=1)
        try:
            adapter.validate(injected)
        except ValueError:
            continue
        raise AssertionError('GT field injection accepted')
    print(json.dumps(dict(status='Passed', packets=len(packets), images=images,
                         blocked_canaries=blocked, injection_rejections=len(packets),
                         model_forward_calls=0, model_loaded=False,
                         scope='Actual input/file-contract probe, not a model forward or hostile-code sandbox')))


if __name__ == '__main__':
    main()
