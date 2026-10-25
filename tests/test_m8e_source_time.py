"""Evidence qualification must fail on stale/missing source snapshots."""
import copy
import json
from pathlib import Path
import unittest
from datasets.m8e_reference import require_source_frame


class SourceTime(unittest.TestCase):
    def setUp(self):
        sensor = dict(frame_id=10, timestamp_s=.5)
        self.side = dict(frame_id=10, timestamp_s=.5,
            objects=[dict(sensor,snapshot_present=True)],
            sensors={str(i):{k:dict(sensor) for k in ['rgb','depth','instance','semantic']} for i in range(6)})

    def test_reject_actor_fallback_and_sensor_staleness(self):
        require_source_frame(self.side)
        for mutate in [lambda s:s['objects'][0].update(snapshot_present=False),
                       lambda s:s['sensors']['0']['depth'].update(frame_id=9),
                       lambda s:s['objects'][0].update(timestamp_s=.4)]:
            s=copy.deepcopy(self.side);mutate(s)
            with self.assertRaises(ValueError):require_source_frame(s)

    def test_reject_nonfinite_or_missing_modality(self):
        for mutate in [lambda s:s['sensors']['0']['instance'].update(timestamp_s=float('nan')),
                       lambda s:s['sensors']['0'].pop('semantic')]:
            s=copy.deepcopy(self.side);mutate(s)
            with self.assertRaises(ValueError):require_source_frame(s)

    def test_all_actual_source_sidecars(self):
        r=json.loads(Path('reports/m8e_actor_environment_gt.json').read_text())
        count=0
        for m in r['manifests']:
            for f in json.loads(Path(m['path']).read_text())['frames']:
                require_source_frame(json.loads(Path(f['path']).read_text()));count+=1
        self.assertEqual(count,32)


if __name__=='__main__':unittest.main()
