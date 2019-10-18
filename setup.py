import setuptools


def read(filename):
    with open(filename) as f:
        return f.read()


long_description = read('README.md')

setuptools.setup(name='seismic-zfp',
                 author='equinor',
                 description='compress and decompress seismic data',
                 long_description=long_description,

                 use_scm_version=True,

                 packages=['seismic_zfp']
                 )
