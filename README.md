# seismic-zfp #

[![LGPLv3 License](https://img.shields.io/badge/License-LGPL%20v3-green.svg)](https://opensource.org/licenses/)
[![Travis](https://travis-ci.org/equinor/seismic-zfp.svg?branch=master)](https://travis-ci.org/equinor/seismic-zfp)
[![codecov](https://codecov.io/gh/equinor/seismic-zfp/branch/master/graph/badge.svg)](https://codecov.io/gh/equinor/seismic-zfp)
[![PyPi Version](https://img.shields.io/pypi/v/seismic-zfp.svg)](https://pypi.org/project/seismic-zfp/)

Python library to convert SEG-Y files to compressed cubes and retrieve arbitrary sub-volumes from these, fast.

## Motivation ##

Reading whole SEG-Y volumes to retrieve, for example, a single time-slice is wasteful.

Copying whole SEG-Y files uncompressed over networks is also wasteful.

This library addresses both issues by implementing the [seismic-zfp (.SGZ) format](docs/file-specification.md).
This format is based on [ZFP compression](https://computing.llnl.gov/projects/floating-point-compression)
from [Peter Lindstrom's paper](https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays)
using the official Python bindings, distributed as zfpy.


ZFP compression enables smoothly varying d-dimensional data in 4<sup>d</sup> subvolumes 
to be compressed at a fixed bitrate. The 32-bit floating point values in 4x4x4 units
of a 3D post-stack SEG-Y file are well suited to this scheme. 

Decomposing an appropriately padded 3D seismic volume into groups of these units which 
exactly fill one 4KB disk block, compressing these groups, and writing them sequentially 
to disk yields a file with the following properties:
- Compression ratio of 2<sup>n</sup>:1 compression, 
typically a bitrate of 4 works well, implying a ratio of 8:1
- The location of any seismic sample is known
- Arbitrary subvolumes can be read with *minimal* redundant I/O, for example:
  - Padding IL/XL dimensions with 4, and the z-dimension depending on bitrate
  - Padding IL/XL dimensions with 64 and the z-dimension with 4 (16:1 compression)
#### Using IL/XL optimized layout ###
- Groups of 4 inlines or crosslines can be read with **no** redundant I/O
- A single inline can be read and with **no** additional I/O compared to the SEG-Y 
best-case scenario (provided at least 4:1 compression ratio)
- A z-slice can be read by accessing n_traces/16 disk blocks, 
compared to n_traces disk blocks for SEG-Y
#### Using z-slice optimized layout ####
- A z-slice can be read by accessing **just** n_traces/4096 disk blocks, 
compared to n_traces disk blocks for SEG-Y

The [seismic-zfp (.SGZ) format](docs/file-specification.md) also allows for preservation of information in 
SEG-Y file and trace headers, with compression code identifying constant 
and varying trace header values and storing these appropriately.

For further explanation of the design and implementation of seismic-zfp, please refer to [publications](#publications).

#### NOTE: Previously the extension .sz was used for seismic-zfp, but has been replaced with .sgz to avoid confusion around the compression algorithm used.

## Get seismic-zfp
- Wheels from [PyPI](https://pypi.org/project/seismic-zfp/) with zgy and vds support: `pip install seismic-zfp[zgy,vds]`
- Wheels from [PyPI](https://pypi.org/project/seismic-zfp/) without zgy or vds support: `pip install seismic-zfp`
- Source from [Github](https://github.com/equinor/seismic-zfp): `git clone https://github.com/equinor/seismic-zfp.git`

*Note that seismic-zfp depends on the Python package [ZFPY](https://pypi.org/project/zfpy/), which is a binary distribution on PyPI built for Linux and Windows.*

*The optional dependency [pyvds](https://github.com/equinor/pyvds) - requires openvds package from Bluware which is **not** open-source*

*The optional dependency of zgy2sgz has been replaced with [pyzgy](https://github.com/equinor/pyzgy) - a pure-Python alternative.*

## Examples ##

Full example code is provided [here](examples), but the following reference is useful:

#### Create SGZ files from SEG-Y, ZGY or VDS ####

```python
from seismic_zfp.conversion import SegyConverter, ZgyConverter, VdsConverter

with SegyConverter("in.sgy") as converter:
    # Create a "standard" SGZ file with 8:1 compression, using in-memory method
    converter.run("out_standard.sgz", bits_per_voxel=4)
    # Create a "z-slice optimized" SGZ file
    converter.run("out_adv.sgz", bits_per_voxel=2, blockshape=(64, 64, 4))
                  
with ZgyConverter("in_8-int.zgy") as converter:
    # 8-bit integer ZGY and 1-bit SGZ have similar quality
    converter.run("out_8bit.sgz", bits_per_voxel=1)

with VdsConverter("in.vds") as converter:
    # VDS compression is superior, but doesn't permit non-cubic bricks...
    converter.run("out.sgz", bits_per_voxel=4, blockshape=(8, 8, 128))
```

#### Convert SGZ files to SEG-Y ####

```python
# Convert SGZ to SEG-Y
with SgzConverter("out_standard.sgz") as converter:
    converter.convert_to_segy("recovered.sgy")
```

#### Read an SGZ file ####
```python
from seismic_zfp.read import SgzReader
with SgzReader("in.sgz") as reader:
    inline_slice = reader.read_inline(LINE_IDX)
    crossline_slice = reader.read_crossline(LINE_IDX)
    z_slice = reader.read_zslice(LINE_IDX)
    sub_vol = reader.read_subvolume(min_il=min_il, max_il=max_il, 
                                    min_xl=min_xl, max_xl=max_xl, 
                                    min_z=min_z, max_z=max_z)
```

#### Use segyio-like interface to read SGZ files ####
```python
import seismic_zfp
with seismic_zfp.open("in.sgz")) as sgzfile:
    il_slice = sgzfile.iline[sgzfile.ilines[LINE_NUMBER]]
    xl_slices = [xl for xl in sgzfile.xline]
    zslices = sgzfile.depth_slice[:5317]
    trace = sgzfile.trace[TRACE_IDX]
    trace_header = sgzfile.header[TRACE_IDX]
    binary_file_header = sgzfile.bin
    text_file_header = sgzfile.text[0]
```
Including equivalents to segyio.tools
```python
with seismic_zfp.open("in.sgz") as sgz_file:
    dt_sgz = seismic_zfp.tools.dt(sgz_file)

cube_sgz = seismic_zfp.tools.cube("in.sgz")
```

Plus some extended utility functionality:
```python
with seismic_zfp.open("in.sgz")) as sgzfile:
    subvolume = sgzfile.subvolume[IL_NO_START:IL_NO_STOP:IL_NO_STEP, 
                                  XL_NO_START:XL_NO_STOP:XL_NO_STEP,
                                  SAMP_NO_START:SAMP_NO_STOP:SAMP_NO_STEP]
    header_arr = sgzfile.get_tracefield_values(segyio.tracefield.TraceField.NStackedTraces)
```

## Command Line Interface
A simple command line interface for converting from SEGY or ZGY to SGZ is also bundled with seismic-zfp. This is available using the command `seismic-zfp`.

#### Converting from SEG-Y to SGZ and back
```bash
seismic-zfp sgy2sgz <INPUT FILE PATH> <OUTPUT FILE PATH> --bits-per-voxel 4
seismic-zfp sgz2sgy <INPUT FILE PATH> <OUTPUT FILE PATH>
```
To see all options run the command `seismic-zfp sgy2sgz --help`

#### Converting from ZGY to SGZ
```bash
seismic-zfp zgy2sgz <INPUT FILE PATH> <OUTPUT FILE PATH> --bits-per-voxel 4
```
To see all options run the command `seismic-zfp zgy2sgz --help`

## Contributing
Contributions welcomed, whether you are reporting or fixing a bug, implementing or requesting a feature. Either make a github issue or fork the project and make a pull request. Please extend the unit tests with relevant passing/failing tests, run these as: `python -m pytest`

## Publications
Wade, David (2020): _Seismic-ZFP: Fast and Efficient Compression and Decompression of Seismic Data_, First EAGE Digitalization Conference and Exhibition, Nov 2020, Volume 2020, p.1-5
- [Full-Text PDF](docs/EAGE_extended-abstract.pdf) (without EAGE formatting)
- [EarthDoc Link](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202032080) (N.B. paywall protected)
