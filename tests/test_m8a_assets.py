"""Synthetic tests: acquisition evidence must never masquerade as a detector."""
from tools.m8a_assets import access_barrier, asset_gate, is_checkpoint_candidate
from tools.m8a_common import write
import pytest


def test_html_named_checkpoint_is_not_weights(tmp_path):
    p = tmp_path/"epoch_24.pth"
    p.write_bytes(b"<html>login required</html>" * 100)
    assert not is_checkpoint_candidate(p, minimum=10)


def test_archive_is_only_candidate_and_does_not_grant_gate(tmp_path):
    p = tmp_path/"epoch_24.pth"
    p.write_bytes(b"PK\x03\x04" + b"\0"*100)
    assert is_checkpoint_candidate(p, minimum=10)
    assert not asset_gate({"official_source": True, "complete_bytes": True,
                           "sha256_recorded": True})


@pytest.mark.parametrize("absent", ["official_source", "usage_scope_confirmed",
    "complete_bytes", "sha256_recorded", "readable_isolated",
    "detection_head_present", "key_shapes_match"])
def test_every_asset_requirement_is_mandatory(absent):
    checks = dict.fromkeys(["official_source", "usage_scope_confirmed", "complete_bytes",
        "sha256_recorded", "readable_isolated", "detection_head_present", "key_shapes_match"], True)
    assert asset_gate(checks)
    checks[absent] = False
    assert not asset_gate(checks)


def test_access_barrier_is_not_retry_instruction():
    assert access_barrier("Download quota exceeded")
    assert access_barrier("Request access")
    assert not access_barrier("Network unreachable")


def test_old_evidence_cannot_be_overwritten(tmp_path):
    p = tmp_path/"evidence.json"
    write(p, {"result":"failed"})
    original = p.read_bytes()
    with pytest.raises(FileExistsError):
        write(p, {"result":"passed"})
    assert p.read_bytes() == original
