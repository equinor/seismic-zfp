import setuptools


def read(filename):
    with open(filename) as f:
        return f.read()


long_description = read('README.md')

setuptools.setup(name='seismic-zfp',
                 author='equinor',
                 description='compress and decompress seismic data',
                 url='https://github.com/equinor/seismic-zfp',
                 license='LGPL-3.0',

                 use_scm_version=True,
                 install_requires=['segyio', 'pyzfp', 'psutil', 'pillow', 'matplotlib'],
                 setup_requires=['setuptools', 'setuptools_scm'],

                 packages=['seismic_zfp']
                 )
