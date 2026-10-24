"""Credential handling tests; no external requests."""
from tools.m8a_proxy_audit import redact,clean_env


def test_proxy_and_exit_redaction():
    value=redact("https://name:private@192.0.2.5:8111/?token=secret 198.51.100.3\nAuthorization: Bearer testsecret")
    for forbidden in ("private","secret","192.0.2.5","198.51.100.3","name:","testsecret"):
        assert forbidden not in value


def test_proxy_comparison_env_does_not_mutate_parent(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY","http://localhost:1234")
    monkeypatch.setenv("all_proxy","socks5h://localhost:4321")
    result=clean_env()
    assert "HTTPS_PROXY" not in result and "all_proxy" not in result
    import os
    assert os.environ["HTTPS_PROXY"]=="http://localhost:1234"


def test_backbone_only_and_wrong_head_shapes_fail():
    from tools.m8a_proxy_inspect import shape_contract, EXPECTED_CLASSES
    assert not shape_contract({"img_backbone.conv1.weight":[64,3,7,7]},EXPECTED_CLASSES)["passed"]
    required=shape_contract({},EXPECTED_CLASSES)["expected_shapes"]
    shapes=dict(required)
    for i in range(6):
        shapes[f"pts_bbox_head.transformer.decoder.layers.{i}.norms.0.weight"]=[256]
    shapes["img_backbone.layer3.0.conv2.conv_offset.weight"]=[27,256,3,3]
    shapes["img_backbone.layer4.0.conv1.weight"]=[512,1024,1,1]
    assert shape_contract(shapes,EXPECTED_CLASSES)["passed"]
    shapes["pts_bbox_head.reference_points.weight"]=[300,3]
    assert not shape_contract(shapes,EXPECTED_CLASSES)["passed"]
