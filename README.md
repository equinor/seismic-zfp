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

The seismic-zfp format also allows for preservation of information in 
SEG-Y file and trace headers, with compression code identifying constant 
and varying trace header values and storing these appropriately.

## Examples ##

Full example code is provided, but the following reference is useful:

#### Create a "standard" SZ file from SEGY ####

```python
from seismic_zfp.convert import convert_segy
convert_segy("in.segy", "out.sz", bits_per_voxel=4)
```

#### Create a "z-slice optimized" SZ file from SEGY ####

```python
from seismic_zfp.convert import convert_segy
convert_segy("in.segy", "out.sz", bits_per_voxel=2, 
             blockshape=(64, 64, 4), method="Stream")
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

#### Convert an SZ file to SEGY ####
```python
from seismic_zfp.read import SzReader
reader = SzReader("in.sz")
reader.write_segy_file("out.segy")
```

