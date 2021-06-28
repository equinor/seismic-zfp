# seismic-zfp examples

These are a set of small Python programs which demonstrate usage of seismic-zfp, which mostly fall into one of two categories:

### Reading SGZ files
   - Accessing inlines/crosslines/zslices
   - Reading individual traces
   - Reading subvolumes
   - Reading file/trace headers
   
   Usage usually follows a pattern like this:
   
   ```shell
   python example.py [FILE_ROOT] [ITEM_NO]
   ```


   
### Writing SGZ files
   - Conversion from SEG-Y and ZGY file formats
   - Creating new SGZ files from numpy arrays
   
   Usage usually follows a pattern like this:
   
  ```shell
  python example.py [IN_FILE] [OUT_FILE] [OUTPUT_BITS_PER_VOXEL]
  ```



