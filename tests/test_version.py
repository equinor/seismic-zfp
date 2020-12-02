import pytest
from seismic_zfp.version import SeismicZfpVersion


def test_create_from_string():
    version = SeismicZfpVersion("100.23.9")
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is False


def test_create_from_encoding():
    version = SeismicZfpVersion(209762323)
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is False


def test_create_from_encoding_rev():
    version = SeismicZfpVersion(209762322)
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is True


def test_create_from_tuple():
    version = SeismicZfpVersion((100, 23, 9, ".dev"))
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is True


def test_compare():
    version1 = SeismicZfpVersion(33550336)
    version2 = SeismicZfpVersion(8128)
    assert version1 > version2


def test_order():
    version_development = SeismicZfpVersion("100.23.9.dev")
    version_release = SeismicZfpVersion("100.23.9")
    assert version_release > version_development
