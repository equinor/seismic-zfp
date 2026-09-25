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

   The 4D examples draw several gathers side by side, as a prestack viewer would:

   ```shell
   python read-inline-4d.py [FILE_ROOT] [LINE_NO] [FIRST_XL_ORDINAL] [N_GATHERS]
   python read-subvolume-4d.py [FILE_ROOT] [IL_SLICE] [XL_SLICE] [OFFSET_SLICE] [SAMPLE_SLICE]
   ```


   
### Writing SGZ files
   - Conversion from SEG-Y and ZGY file formats
   - Creating new SGZ files from numpy arrays
   
   Usage usually follows a pattern like this:
   
  ```shell
  python example.py [IN_FILE] [OUT_FILE] [OUTPUT_BITS_PER_VOXEL]
  ```



