import pytest
from click.testing import CliRunner
from seismic_zfp.cli import cli
from seismic_zfp.conversion import SegyConverter, ZgyConverter
from seismic_zfp.read import SgzReader
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


def test_sgy2sgz_convert_2d_default():
    input_file = os.path.join("test_data", "small-2d.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small-2d_4bit_converted.sgz"
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


def test_sgy2sgz_get_output_size():
    input_file = os.path.join("test_data", "small.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_4bit_converted.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        with SegyConverter(input_file_absolute) as converter:
            expected_size = converter.get_output_size(bits_per_voxel=4)

        result = runner.invoke(
            cli,
            [
                "sgy2sgz",
                input_file_absolute,
                output_file,
                "--get-output-size",
            ],
        )
        assert result.exit_code == 0
        assert result.output.strip() == str(expected_size)
        assert not os.path.exists(output_file)


def test_sgy2sgz_convert_all_params():
    input_file = os.path.join("test_data", "small.sgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_2bit_converted_64_64_-1.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        with SegyConverter(input_file_absolute, min_il=0, max_il=4, min_xl=0, max_xl=3) as converter:
            expected_size = converter.get_output_size(bits_per_voxel=2, blockshape=(64, 64, -1))
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
        assert result.exit_code == 0, result.output
        assert os.stat(output_file).st_size == expected_size
        with SgzReader(output_file) as reader:
            assert (reader.n_ilines, reader.n_xlines) == (4, 3)


def test_sgy2sgz_convert_4d_default():
    input_file_absolute = os.path.abspath(os.path.join("test_data", "small-4d.sgy"))
    output_file = "small-4d_4bit_converted.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        with SegyConverter(input_file_absolute) as converter:
            expected_size = converter.get_output_size(bits_per_voxel=4)
        result = runner.invoke(cli, ["sgy2sgz", input_file_absolute, output_file])
        assert result.exit_code == 0, result.output
        assert os.stat(output_file).st_size == expected_size


def test_sgy2sgz_convert_4d_all_params():
    input_file_absolute = os.path.abspath(os.path.join("test_data", "small-4d.sgy"))
    output_file = "small-4d_2bit_cropped.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        with SegyConverter(input_file_absolute, min_il=1, max_il=4, min_offset=1, max_offset=4) as converter:
            expected_size = converter.get_output_size(bits_per_voxel=2, blockshape=(8, 8, 4, 64))
        result = runner.invoke(
            cli,
            ["sgy2sgz", input_file_absolute, output_file,
             "--bits-per-voxel", "2",
             "--blockshape-4d", "8", "8", "4", "64",
             "--min-il", "1", "--max-il", "4",
             "--min-offset", "1", "--max-offset", "4"],
        )
        assert result.exit_code == 0, result.output
        assert os.stat(output_file).st_size == expected_size

        result = runner.invoke(
            cli,
            ["sgy2sgz", input_file_absolute, "--get-output-size",
             "--bits-per-voxel", "2", "--blockshape-4d", "8", "8", "4", "64",
             "--min-il", "1", "--max-il", "4", "--min-offset", "1", "--max-offset", "4"],
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip() == str(expected_size)


def test_sgy2sgz_blockshape_option_mismatches():
    sgy_4d = os.path.abspath(os.path.join("test_data", "small-4d.sgy"))
    sgy_3d = os.path.abspath(os.path.join("test_data", "small.sgy"))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["sgy2sgz", sgy_4d, "out.sgz", "--blockshape", "4", "4", "-1"])
        assert result.exit_code != 0
        assert "--blockshape-4d" in result.output

        result = runner.invoke(cli, ["sgy2sgz", sgy_3d, "out.sgz", "--blockshape-4d", "4", "4", "4", "-1"])
        assert result.exit_code != 0
        assert "only applicable to prestack" in result.output

        result = runner.invoke(cli, ["sgy2sgz", sgy_4d, "out.sgz",
                                     "--blockshape", "4", "4", "-1", "--blockshape-4d", "4", "4", "4", "-1"])
        assert result.exit_code != 0
        assert "only one of" in result.output

        result = runner.invoke(cli, ["sgy2sgz", sgy_3d, "out.sgz", "--min-offset", "1"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert not os.path.exists("out.sgz")


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
def test_zgy2sgz_get_output_size():
    input_file = os.path.join("test_data", "zgy", "small-8bit.zgy")
    input_file_absolute = os.path.abspath(input_file)
    output_file = "small_4bit_converted_zgy.sgz"
    runner = CliRunner()
    with runner.isolated_filesystem():
        with ZgyConverter(input_file_absolute) as converter:
            expected_size = converter.get_output_size(bits_per_voxel=4)

        result = runner.invoke(
            cli,
            [
                "zgy2sgz",
                input_file_absolute,
                output_file,
                "--get-output-size",
            ],
        )
        assert result.exit_code == 0
        assert result.output.strip() == str(expected_size)
        assert not os.path.exists(output_file)


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
