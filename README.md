# seismic-zfp #
Python library to convert SEG-Y files to compressed cubes and retrieve arbitrary sub-volumes from these, fast.

## Motivation ##

Reading whole SEG-Y volumes to retrieve, for example, a single time-slice is wasteful.

Copying whole SEG-Y files uncompressed over networks is also wasteful.

This library addresses both issues by implementing the seismic-zfp (.SZ) format.
This format is based on [ZFP compression](https://computing.llnl.gov/projects/floating-point-compression)
from [Peter Lindstrom's paper](https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays),
using [the Python wrapper](https://github.com/navjotk/pyzfp) developed by Navjot Kukreja.


ZFP compression enables smoothly varying d-dimensional data in 4<sup>d</sup> subvolumes 
to be compressed at a fixed bitrate. The 32-bit floating point values in 4x4x4 units
of a 3D post-stack SEG-Y file are well suited to this scheme. 

Decomposing an appropriately padded 3D seismic volume into groups of these units which 
exactly fill one 4Kb disk block, compressing these groups, and writing them sequentially 
to disk yields a file with the following properties:
- Compression ratio of 2<sup>n</sup>:1 compression, 
typically a bitrate of 4 works well, implying a ratio of 8:1
- The location of any seismic sample is known
- A group of 4 inlines can be read and with **no** additional I/O compared to SEGY 
(provided at least 4:1 compression ratio)
- A group of 4 crosslines can be read with **no** redundant I/O
- A z-slice can be read by accessing **just** n_traces/16 disk blocks, compared to n_traces for SEG-Y
- Arbitrary subvolumes can be read with *minimal* redundant I/O 
(padding IL/XL dimensions with 4, and the z-dimension depending on bitrate)

## Examples ##

Full example code is provided, but the following reference is useful:

#### Create an SZ file from SEGY ####

```python
from seismic_zfp.convert import convert_segy
convert_segy("in.segy", "out.sz", bits_per_voxel=8)
```

#### Read an SZ file ####
```python
from seismic_zfp.read import SzReader
reader = SzReader("in.sz")
inline_slice = reader.read_inline(LINE_NO)
crossline_slice = reader.read_crossline(LINE_NO)
z_slice = reader.read_zslice(LINE_NO)
sub_vol = reader.read_subvolume(min_il=min_il, max_il=max_il, 
                                min_xl=min_xl, max_xl=max_xl, 
                                min_z=min_z, max_z=max_z)
```