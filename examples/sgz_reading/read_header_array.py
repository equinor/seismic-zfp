from seismic_zfp.read import SgzReader
import segyio
import os
import sys
from matplotlib import pyplot as plt

base_path = sys.argv[1]

# Here we demonstrate retrieving header values for all traces in the file
# ... rather more quickly than would be possible with SEG-Y

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    reader.read_variant_headers()
    header_values = reader.variant_headers[segyio.TraceField.NStackedTraces]
    plt.imsave(os.path.join(base_path, '0_NStackedTraces.png'),
               header_values.reshape(reader.n_ilines, reader.n_xlines))
