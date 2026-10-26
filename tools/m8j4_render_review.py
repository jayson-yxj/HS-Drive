"""Reproduce private reviewed (post-quarantine) mosaics, without a model."""
import json

import cv2
import numpy as np

from tools.m8j4_sidecar import OUT, render


def main():
    manifest = json.loads((OUT / 'review_manifest.json').read_text())
    output = OUT / 'reviewed_visuals'
    output.mkdir(exist_ok=False)
    for ordinal in manifest['all_review_frames']:
        frame = json.loads((OUT / 'reviewed_sidecar' / (ordinal + '.json')).read_text())
        render(frame, output)
    for label, ordinals in [('uniform', manifest['uniform']), ('anomalies', manifest['anomalies'])]:
        for i in range(0, len(ordinals), 4):
            images = [cv2.imread(str(output / (s + '_rgb.jpg'))) for s in ordinals[i:i + 4]]
            cv2.imwrite(str(output / f'{label}_{i // 4:02}.jpg'), np.concatenate(images, axis=0))


if __name__ == '__main__':
    main()
