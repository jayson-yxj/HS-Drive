"""M5-I adds unknown response without turning unseen pixels into negatives."""
import numpy as np
from diagnostics.m5h_metrics import enhanced as baseline


def enhanced(rows, labels, probs, threshold=None, classes=(1, 2, 3), locked_low=None):
    result = baseline(rows, labels, probs, threshold, classes, locked_low)
    hist = np.zeros(256, np.int64)
    by_camera = {}
    for row, label, probability in zip(rows, labels, probs):
        for camera, name in enumerate(row['camera_names']):
            mask = ~label['valid'][camera]
            q = np.minimum(probability[camera][mask] * 256, 255).astype(np.uint8)
            h = np.bincount(q, minlength=256)
            hist += h
            by_camera[name] = by_camera.get(name, np.zeros(256, np.int64)) + h
    def stats(h):
        n = int(h.sum())
        return dict(n=n, histogram=h.tolist(), mean=float(h @ (np.arange(256)/256)/n) if n else None,
                    high_fraction=float(h[int(result['threshold']*256):].sum()/n) if n else None)
    result['unknown_response'] = dict(pooled=stats(hist), by_camera={k:stats(v) for k,v in by_camera.items()},
                                      interpretation='Unknown scores are diagnostic; no negative labels, precision, or safety-free claim.')
    return result
