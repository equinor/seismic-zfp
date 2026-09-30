# seismic-zfp tools

Standalone utilities which are useful around seismic-zfp but are not part of the library.

### make-regular-prestack-segy.py
Derives a regular prestack SEG-Y (every offset present at every IL/XL position) from an
irregular one by zero-padding missing traces, optionally trimming sparsely filled axis ends.

```shell
python make-regular-prestack-segy.py IN.sgy OUT.sgy [--min-fill F] [--dry-run]
```

seismic-zfp compresses irregular prestack SEG-Y directly; a regular file is needed to use
`reduce_iops`, to crop on conversion, or to compare against tools which expect a full lattice.
