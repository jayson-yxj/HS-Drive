"""Offline tests only: metadata whitelist, credential redaction, one-batch marker."""
import tempfile
import unittest
from pathlib import Path
from tools.m8j1_metadata_batch import validate, public_location, write, CLIP, PIN


class MetadataBoundaryTests(unittest.TestCase):
    def test_archive_get_and_range_method_rejected(self):
        url=f'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/{PIN}/{CLIP}'
        for method in ('GET','POST','RANGE'):
            with self.subTest(method=method),self.assertRaises(AssertionError):validate(method,url)
        validate('HEAD',url)

    def test_only_target_revision_and_official_hosts(self):
        urls=[f'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/{CLIP}',
              f'https://unverified.example/datasets/rethinklab/Bench2Drive/resolve/{PIN}/{CLIP}',
              f'http://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/{PIN}/{CLIP}']
        for url in urls:
            with self.subTest(url=url),self.assertRaises(AssertionError):validate('HEAD',url)

    def test_public_metadata_allowed(self):
        validate('GET','https://huggingface.co/api/datasets/rethinklab/Bench2Drive?blobs=true')
        validate('GET','https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/LICENSE?ref=fixed')

    def test_exclusive_marker_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'marker.json';write(p,{'batch':1})
            with self.assertRaises(FileExistsError):write(p,{'batch':2})
            self.assertIn('1',p.read_text())

    def test_redact_signed_location(self):
        v=public_location('https://user:secret@official.example/object?Signature=secret&Token=hidden')
        self.assertEqual(v,'https://official.example/object?REDACTED')


if __name__=='__main__':unittest.main()
