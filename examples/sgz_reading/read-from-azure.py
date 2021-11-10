import seismic_zfp
import time

sas = None
remote_file = (sas, "testseismic", "Volve.sgz")

f = seismic_zfp.open(remote_file)

print("Trace shape", f.trace[1337].shape)

t0 = time.time()
data = f.iline[f.ilines[0]]
print("Inline took", time.time()-t0)


t0 = time.time()
data = f.xline[f.xlines[0]]
print("Crossline took", time.time()-t0)
