import pytest
from click.testing import CliRunner
from seismic_zfp.cli import cli
import os
import warnings

try:
    with warnings.catch_warnings():
        # pyzgy will warn us that sdglue is not available. This is expected, and safe for our purposes.
        warnings.filterwarnings("ignore", message="seismic store access is not available: No module named 'sdglue'")
        import pyzgy
except ImportError:
    pyzgy = None

def test_sgy2sgz():
    runner = CliRunner()
    result = runner.invoke(cli, ["sgy2sgz", "--help"])
    assert result.exit_code == 0


def test_sgz2sgy_convert():
    input_file = os.path.join("test_data", "small_4bit.sgz")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small.sgy"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["sgz2sgy", input_file_absolute, output_file])
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


def test_sgy2sgz_convert_default():
    input_file = os.path.join("test_data", "small.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_4bit_converted.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["sgy2sgz", input_file_absolute, output_file])
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


def test_sgy2sgz_convert_bits_per_voxel():
    input_file = os.path.join("test_data", "small.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_2bit_converted.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["sgy2sgz", input_file_absolute, output_file, "--bits-per-voxel", "2"]
        )
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


def test_sgy2sgz_convert_all_params():
    input_file = os.path.join("test_data", "small.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_2bit_converted_64_64_-1.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "sgy2sgz",
                input_file_absolute,
                output_file,
                "--bits-per-voxel",
                "2",
                "--blockshape",
                "64",
                "64",
                "-1",
                "--reduce-iops",
                "true",
                "--min-il",
                "0",
                "--max-il",
                "4",
                "--min-xl",
                "0",
                "--max-xl",
                "3",
            ],
        )
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_zgy2sgz():
    runner = CliRunner()
    result = runner.invoke(cli, ["zgy2sgz", "--help"])
    assert result.exit_code == 0


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_zgy2sgz_convert_default():
    input_file = os.path.join("test_data", "zgy", "small-8bit.zgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_4bit_converted_zgy.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["zgy2sgz", input_file_absolute, output_file])
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_zgy2sgz_convert_bits_per_voxel():
    input_file = os.path.join("test_data", "zgy", "small-16bit.zgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_2bit_converted_zgy.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["zgy2sgz", input_file_absolute, output_file, "--bits-per-voxel", "2"]
        )
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_zgy2sgz_convert_all_params():
    input_file = os.path.join("test_data", "zgy", "small-32bit.zgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_2bit_converted_64_64_-1_zgy.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "zgy2sgz",
                input_file_absolute,
                output_file,
                "--bits-per-voxel",
                "2",
            ],
        )
        assert os.path.exists(output_file)
        assert os.stat(output_file).st_size > 0
    assert result.exit_code == 0
