"""Archive path safety independent of the downloaded private archive."""
import pytest

from tools.m8j6_ingest import member_path


@pytest.mark.parametrize('name', ['/etc/passwd', '../escape', 'a/../../escape', 'a\\..\\escape'])
def test_reject_unsafe_archive_paths(name):
    with pytest.raises(ValueError):
        member_path(name)


@pytest.mark.parametrize('name', ['v1.0-mini/sample.json', './samples/CAM_FRONT/a.jpg', 'maps/a.png'])
def test_valid_relative_paths(name):
    assert not member_path(name).is_absolute()
