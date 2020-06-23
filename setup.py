import setuptools
import re


def get_long_description():
    with open('README.md') as f:
        raw_readme = f.read()
    base_repo = 'https://github.com/equinor/seismic-zfp/tree/'
    with open('.git/refs/heads/master') as f:
        commit = f.read().rstrip()
    substituted_readme = re.sub('\\]\\((?!https)', '](' + base_repo + commit + '/', raw_readme)
    return substituted_readme


setuptools.setup(name='seismic-zfp',
                 author='equinor',
                 description='Compress and decompress seismic data',
                 long_description=get_long_description(),
                 long_description_content_type='text/markdown',
                 url='https://github.com/equinor/seismic-zfp',
                 license='LGPL-3.0',

                 use_scm_version=True,
                 install_requires=['functools32;python_version<"3"',
                                   'numpy>=1.16', 'segyio', 'zfpy', 'psutil', 'pillow', 'matplotlib', 'Cython'],
                 setup_requires=['setuptools', 'setuptools_scm'],

                 packages=['seismic_zfp']
                 )
