"""Offline boundary tests for the metadata-only review; no HTTP calls."""
import unittest
from tools.m8j0_probe import CLIP, validate, public_url


class ReviewBoundaryTest(unittest.TestCase):
    def test_target_payload_get_is_rejected(self):
        with self.assertRaises(AssertionError):
            validate('GET', 'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/' + CLIP)

    def test_target_head_is_allowed(self):
        validate('HEAD', 'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/' + CLIP)

    def test_other_asset_head_is_rejected(self):
        with self.assertRaises(AssertionError):
            validate('HEAD', 'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/other.tar.gz')

    def test_arbitrary_domain_and_source_archive_rejected(self):
        for url in ('https://unverified.example/README.md',
                    'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/tarball/main',
                    'https://huggingface.co/datasets/other/repo/raw/main/README.md'):
            with self.subTest(url=url), self.assertRaises(AssertionError): validate('GET', url)

    def test_metadata_and_readme_allowed(self):
        validate('GET', 'https://huggingface.co/api/datasets/rethinklab/Bench2Drive?blobs=true')
        validate('GET', 'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/LICENSE?ref=fixed')
        validate('GET', 'https://huggingface.co/datasets/rethinklab/Bench2Drive/raw/fixed/README.md')

    def test_redirect_requires_verified_origin(self):
        target = 'https://cas-bridge.xethub.hf.co/x/y?Signature=secret'
        with self.assertRaises(AssertionError): validate('HEAD', target)
        validate('HEAD', target, 'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/fixed/' + CLIP)

    def test_non_https_encoded_path_and_credentials_rejected(self):
        for url in ('http://huggingface.co/api/datasets/rethinklab/Bench2Drive',
                    'https://user:secret@huggingface.co/api/datasets/rethinklab/Bench2Drive',
                    'https://huggingface.co/datasets/rethinklab/Bench2Drive/raw/main/%52EADME.md'):
            with self.subTest(url=url), self.assertRaises(AssertionError): validate('GET', url)

    def test_signed_query_redaction(self):
        value = public_url('https://user:secret@cas-bridge.xethub.hf.co/path?Signature=abc&Token=def')
        self.assertNotIn('secret', value); self.assertNotIn('abc', value); self.assertNotIn('def', value)
        self.assertEqual(value.count('REDACTED'), 2)


if __name__ == '__main__': unittest.main()
