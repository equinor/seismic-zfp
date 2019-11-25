import os
from read import SzReader
import segyio

base_path = '/data-share/s/seismic-zfp'

reader = SzReader(os.path.join(base_path, '0_headers.sz'))

print(reader.segy_traceheader_template)

with segyio.open('/data-share/s/seismic-zfp/0.segy') as segyfile:
    print(segyfile.ilines)
    print(segyfile.xlines)
    print(segyfile.samples)
    print(segyfile.format)
