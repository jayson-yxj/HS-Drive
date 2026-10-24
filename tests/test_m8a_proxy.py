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
