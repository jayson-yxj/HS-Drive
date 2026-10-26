import tarfile
import pytest
from tools.m8j3_archive_audit import validate_member, MAX_FILE


@pytest.mark.parametrize('name', ['/etc/x', '../x', 'x/../../y', 'C:/x', 'x\\y', 'x\ny'])
def test_reject_paths(name):
    with pytest.raises(ValueError):
        validate_member(tarfile.TarInfo(name), set())


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_reject_nonregular(kind):
    m = tarfile.TarInfo('ok'); m.type = kind
    with pytest.raises(ValueError):
        validate_member(m, set())


def test_duplicate_and_size():
    seen = set()
    assert validate_member(tarfile.TarInfo('./safe/a.jpg'), seen) == 'safe/a.jpg'
    with pytest.raises(ValueError):
        validate_member(tarfile.TarInfo('safe/a.jpg'), seen)
    m = tarfile.TarInfo('large'); m.size = MAX_FILE + 1
    with pytest.raises(ValueError):
        validate_member(m, seen)
