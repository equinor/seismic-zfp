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
