import hashlib
from pathlib import Path
import pytest
from tools.m6f_launch import CapturedSource, LOADED, verify_loaded


def test_import_hash_is_the_executed_source_not_later_disk(tmp_path):
    p=tmp_path/'m6f_fixture.py';original=b'VALUE = 47\n';p.write_bytes(original)
    name='m6f_test_fixture';loader=CapturedSource(name,str(p));code=loader.get_code(name);scope={};exec(code,scope)
    assert scope['VALUE']==47
    assert LOADED[name]['source_sha256']==hashlib.sha256(original).hexdigest()
    p.write_text('VALUE = 48\n')
    assert scope['VALUE']==47
    try:
        with pytest.raises(RuntimeError,match='Loaded source changed'):verify_loaded()
    finally:LOADED.pop(name)
