import os
import sys
import numpy as np
from matplotlib import pyplot as plt

import seismic_zfp
import segyio

base_path = sys.argv[1]

# Here we demonstrate retrieving header values for all traces in the file
# ... rather more quickly and simply than would be possible with SEG-Y


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    sgz_headers = sgzfile.get_tracefield_values(segyio.tracefield.TraceField.NStackedTraces)
    plt.imsave(os.path.join(base_path, '0_NStackedTraces_sgz.png'), sgz_headers)

with segyio.open(os.path.join(base_path, '0.sgy')) as sgyfile:
    tracefield_values = np.array([h[segyio.tracefield.TraceField.NStackedTraces] for h in sgyfile.header[:]])
    sgy_headers = tracefield_values.reshape((len(sgyfile.ilines), len(sgyfile.xlines)))
    plt.imsave(os.path.join(base_path, '0_NStackedTraces_sgy.png'), sgy_headers)
