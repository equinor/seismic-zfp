## SeismicZFP File Specification

A SeismicZFP (.sgz) file consists of 3 parts:
1. A header, containing:
   * Information needed to read the file
   * Values of _invariant_ SEG-Y trace header fields
   * SEG-Y Textual and Binary file headers
2. The compressed seismic data itself
3. A footer, containing values of _variant_ trace header fields from [SEG-Y](https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf)

The length of these component parts can be calculated as follows:


| | Size (bytes) | 
|-|-|
|**SeismicZFP Header** | 8192 (see below for details)
|**ZFP-compressed fp data** | padded(nIL x nXL x nS) x bitrate / 8
|**SeismicZFP Footer** | nHeaders x nIL x nXL x 4

The padding of the IL/XL/Samples dimensions is to the logical size of those dimensions stored in a single disk-block, which is dependent on the bit-rate. For example, at 2 bits-per-voxel the logical size of the data in a 4K disk block may be 4x4x1024, or 64x64x4, etc.

---

### SeismicZFP Header (v0.1.7 onwards)

Byte encoding is little-endian.

| Bytes  | Type | Contents |
|---|---:|---|
|0-3   |uint32 |Number of 4K blocks of header
|4-7   |uint32 |Samples per trace
|8-11  |uint32 |Number of crosslines
|12-15 |uint32 |Number of inlines
|16-19 |uint32 |Minimum sample time/depth
|20-23 |uint32 |Minimum crossline number
|24-27 |uint32 |Minimum inline number
|28-31 |uint32 |Sample interval (μs/m)
|32-35 |uint32 |Crossline interval
|36-39 |uint32 |Inline interval
|40-43 |int32 |Bits-per-voxel (negative signifying reciprocal)
|44-47 |uint32 |Blockshape: IL-direction
|48-51 |uint32 |Blockshape: XL-direction
|52-55 |uint32 |Blockshape: Trace-direction
|56-59 |uint32 |Number of 4K disk blocks for data
|60-63 |uint32 |Number of bytes for each header array
|64-67 |uint32 |Number of header arrays
|68-71 |uint32 |Number of traces (unstructured files)
|72-75 |uint32 |Encoded version number
|76-79 |uint32 |Encoded source format 0=SEG-Y, 10=ZGY, 20=numpy
|80-83 |uint32 |Encoded header-detection method ***
|84-979 |---  | --- Unused ---
|980-2047 |** |Default trace header values
|2048-4095 |---  | --- Unused ---
|4096-7295 |[SEG-Y](https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf)  | SEG-Y First textual header
|7296-7695 |[SEG-Y](https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf)  | SEG-Y Binary header
|7676-8191 |---  | --- Unused ---

** *Invariant* trace header values are stored in an 89x3 array of 4-byte entries, corresponding to [the 89 fields in SEG-Y trace headers](https://github.com/equinor/segyio/blob/master/python/segyio/tracefield.py). For each row the enties are:
1. Trace header start-byte
2. Constant value
3. Duplicated trace header start-byte

Storing whether trace header fields are duplicates of previous ones reduces the space needed to store the SGZ footer.

*** SeismicZFP supports multiple methods of determining which SEG-Y trace headers are present and/or duplicated. The file maintains a record of which was used (heuristic only until v0.1.10):

0=heuristic, 10=thorough, 20=exhaustive, 30=strip