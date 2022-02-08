import pytest
from seismic_zfp.version import SeismicZfpVersion


def test_create_from_string():
    version = SeismicZfpVersion("100.23.9")
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is False


def test_create_from_string_changes():
    version = SeismicZfpVersion("100.23.9rc2")
    assert version.major == 100
    assert version.minor == 23
    assert version.patch == 9
    assert version.changes_exist is True


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


def test_to_tuple_dev():
    version = SeismicZfpVersion("100.23.9rc2")
    assert version.to_tuple() == (100, 23, 9, ".dev")


def test_to_tuple():
    version = SeismicZfpVersion("100.23.9")
    assert version.to_tuple() == (100, 23, 9)


def test_repr():
    version = SeismicZfpVersion(209762322)
    assert version.__repr__() == "Version(100.23.9.dev)"


def test_compare():
    version1 = SeismicZfpVersion(33550336)
    version2 = SeismicZfpVersion(8128)
    version3 = SeismicZfpVersion("1.2.3")
    version4 = SeismicZfpVersion(2101255)
    version5 = SeismicZfpVersion((1, 2, 3))
    assert version1 > version2
    assert version3 < version1
    assert version3 == version4
    assert version4 == version5
    assert version3 == version5


def test_order():
    version_development = SeismicZfpVersion("100.23.9.dev")
    version_release = SeismicZfpVersion("100.23.9")
    assert version_release > version_development
