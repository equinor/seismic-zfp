# seismic-zfp examples

These are a set of small Python programs which demonstrate usage of seismic-zfp, which mostly fall into one of two categories:

### Reading SGZ files
   - Accessing inlines/crosslines/zslices
   - Reading individual traces
   - Reading subvolumes
   - Reading file/trace headers
   - Reading gathers from 4D (prestack) files
   
   Usage usually follows a pattern like this:
   
   ```shell
   python example.py [FILE_ROOT] [ITEM_NO]
   ```

   The 4D examples read gathers, common-offset sections and sub-volumes from prestack files,
   several of them drawing gathers side by side as a prestack viewer would:

   ```shell
   python read-gather-4d.py [FILE_ROOT] [INLINE] [CROSSLINE]
   python read-common-offset-4d.py [FILE_ROOT] [LINE_NO] [OFFSET]
   python read-inline-4d.py [FILE_ROOT] [LINE_NO] [FIRST_XL_ORDINAL] [N_GATHERS]
   python read-subvolume-4d.py [FILE_ROOT] [IL_SLICE] [XL_SLICE] [OFFSET_SLICE] [SAMPLE_SLICE]
   python read-fold-map-4d.py [SGZ_FILE] [LINE_NO]
   ```

   The fold map needs no seismic data at all: it is counted from the trace header arrays, in which
   positions of an irregular file with no trace are zero.

   SGZ files can also be opened with `xarray.open_dataset`, giving lazy label-based selection over
   (il, xl, z) or, for 4D files, (il, xl, offset, z):

   ```shell
   python read-as-xarray.py [FILE_ROOT] [LINE_NO]
   python read-as-xarray-4d.py [FILE_ROOT] [INLINE] [CROSSLINE] [OFFSET]
   ```


   
### Writing SGZ files
   - Conversion from SEG-Y and ZGY file formats
   - Creating new SGZ files from numpy arrays
   
   Usage usually follows a pattern like this:
   
  ```shell
  python example.py [IN_FILE] [OUT_FILE] [OUTPUT_BITS_PER_VOXEL]
  ```



