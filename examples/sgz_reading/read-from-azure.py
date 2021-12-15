import seismic_zfp
import time

sas_tokens = {'premiumblock': None, 'premiumpage' : None, 'hotblock': None}

for storage_type, sas_token in sas_tokens.items():
    print(storage_type)
    remote_file = (sas_token, "testseismic", "Volve.sgz")

    f = seismic_zfp.open(remote_file)

    print("Trace shape\n", f.trace[1337].shape)

    print("File text header\n", f.text[0])
    print("File binary header\n", f.bin)

    print("Trace header #42\n", f.header[42])

    t0 = time.time()
    data = f.iline[f.ilines[0]]
    print("Inline took", time.time()-t0)

    t0 = time.time()
    data = f.xline[f.xlines[0]]
    print("Crossline took", time.time()-t0)

    t0 = time.time()
    data = f.depth_slice[200]
    print("Zslice took", time.time()-t0)
