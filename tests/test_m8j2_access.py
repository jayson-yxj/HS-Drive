"""Synthetic request-boundary checks; no remote calls."""
import unittest
from tools.m8j2_access_probe import validate,public_url,redact,CLIP


class AccessBoundaryTests(unittest.TestCase):
    def test_payload_get_rejected(self):
        url='https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/fixed/'+CLIP
        with self.assertRaises(AssertionError):validate('GET',url)
        validate('HEAD',url)

    def test_other_clip_rejected(self):
        with self.assertRaises(AssertionError):validate('HEAD','https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/fixed/other.tar.gz')

    def test_storage_head_requires_issued_redirect(self):
        url='https://cas-bridge.xethub.hf.co/object?Signature=secret'
        with self.assertRaises(AssertionError):validate('HEAD',url)
        validate('HEAD',url,True)
        with self.assertRaises(AssertionError):validate('GET',url,True)

    def test_untrusted_hosts_http_and_credentials_rejected(self):
        for url in ('http://huggingface.co/api/datasets/rethinklab/Bench2Drive',
                    'https://other.example/object','https://user:secret@huggingface.co/api/datasets/rethinklab/Bench2Drive'):
            with self.subTest(url=url),self.assertRaises(AssertionError):validate('GET',url)

    def test_metadata_allowed(self):
        validate('GET','https://huggingface.co/api/datasets/rethinklab/Bench2Drive?blobs=true')
        validate('GET','https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/LICENSE?ref=fixed')

    def test_signed_storage_query_removed(self):
        self.assertEqual(public_url('https://user:secret@cas-bridge.xethub.hf.co/a?Signature=abc&Token=secret'),
                         'https://cas-bridge.xethub.hf.co/a?REDACTED')

    def test_error_address_followed_by_punctuation_redacted(self):
        self.assertEqual(redact('rate limit exceeded for 192.0.2.10. (retry later)'),
                         'rate limit exceeded for <ip_redacted>. (retry later)')


if __name__=='__main__':unittest.main()
