import setuptools
import re


def get_long_description():
    with open('README.md') as f:
        raw_readme = f.read()
    base_repo = 'https://github.com/equinor/seismic-zfp/tree/'
    try:
        # Travis job provides this, but don't clutter Github Actions for code coverage
        with open('.git/refs/heads/master') as f:
            commit = f.read().rstrip()
        return re.sub('\\]\\((?!https)', '](' + base_repo + commit + '/', raw_readme)
    except FileNotFoundError:
        return raw_readme


setuptools.setup(name='seismic-zfp',
                 author='equinor',
                 description='Compress and decompress seismic data',
                 long_description=get_long_description(),
                 long_description_content_type='text/markdown',
                 url='https://github.com/equinor/seismic-zfp',
                 license='LGPL-3.0',

                 use_scm_version=True,
                 install_requires=['numpy>=1.20', 'segyio', 'zfpy', 'psutil', 'click'],
                 extras_require={
                     'zgy': ['pyzgy'],
                     'vds': ['pyvds'],
                     'xr': ['xarray>=0.20.2'],
                     'azure': ['azure-storage-blob'],
                 },
                 setup_requires=['setuptools', 'setuptools_scm'],
                 entry_points={
                     'xarray.backends': ['sgz_engine=seismic_zfp.sgz_xarray:SeismicZfpBackendEntrypoint'],
                     'console_scripts' : ['seismic-zfp=seismic_zfp.cli:cli']
                 },
                 packages=['seismic_zfp']
                 )
