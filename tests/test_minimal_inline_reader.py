from seismic_zfp import conversion_utils
import segyio

SGY_FILE = 'test_data/small.sgy'


def test_inline_reading():
    with segyio.open(SGY_FILE) as sgyfile:
        assert conversion_utils.MinimalInlineReader(sgyfile).self_test()
