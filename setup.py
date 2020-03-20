import setuptools


def read(filename):
    with open(filename) as f:
        return f.read()


long_description = read('README.md')

setuptools.setup(name='seismic-zfp',
                 author='equinor',
                 description='Compress and decompress seismic data',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/equinor/seismic-zfp',
                 license='LGPL-3.0',

                 use_scm_version=True,
                 install_requires=['functools32;python_version<"3"',
                                   'segyio', 'pyzfp', 'psutil', 'pillow', 'matplotlib', 'Cython'],
                 setup_requires=['setuptools', 'setuptools_scm'],

                 packages=['seismic_zfp']
                 )
