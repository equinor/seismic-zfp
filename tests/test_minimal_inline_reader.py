import os
import seismic_zfp
from seismic_zfp import conversion_utils
from seismic_zfp.conversion import SegyConverter
import segyio
import mock
import pytest

SGY_FILE = 'test_data/small.sgy'


def test_inline_reading():
    with segyio.open(SGY_FILE) as sgyfile:
        assert conversion_utils.MinimalInlineReader(sgyfile).self_test()


def test_minimal_inline_reader_defaults(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'test_minimal_il_reader.sgz')
    with mock.patch('seismic_zfp.conversion_utils.MinimalInlineReader.self_test') as mocked_self_test:
        mocked_self_test.return_value = False
        with pytest.warns(UserWarning):
            with SegyConverter(SGY_FILE) as converter:
                converter.run(out_sgz, reduce_iops=True)


def test_minimal_inline_reader_wrong_format(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'test_minimal_il_reader_wrong_format.sgz')
    with mock.patch('seismic_zfp.conversion_utils.MinimalInlineReader.get_format_code') as mocked_self_test_format_code:
        mocked_self_test_format_code.return_value = 2
        with pytest.raises(RuntimeError):
            with SegyConverter(SGY_FILE) as converter:
                converter.run(out_sgz, reduce_iops=True)
