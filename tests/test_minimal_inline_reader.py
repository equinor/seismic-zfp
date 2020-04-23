from seismic_zfp import conversion_utils

SGY_FILE = 'test_data/small.sgy'


def test_inline_reading():
    assert conversion_utils.MinimalInlineReader(SGY_FILE).self_test()
