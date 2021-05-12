import pathlib
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

dir_ = Path('Processed/CeramicSamples/RequestedSamples/Sample2')
dir_shifted = Path('Processed/CeramicSamples/RequestedSamples/Sample2_shifted')
pathlib.Path(dir_shifted).mkdir(parents=True, exist_ok=True)

datafiles = [os.path.join(root, name)
     for root, dirs, files in os.walk(dir_)
        for name in files
            if name.endswith((".txt"))]

ref = np.loadtxt(datafiles[0])

for file in datafiles:
    if 'deg' in file:
        sample = np.loadtxt(file)
    else:
        continue

    # index delta
    shift = np.where(ref[:,0]==min(sample[:,0]))[0][0]
    # zero pad and add time axis section
    zero_pad = np.zeros((shift, 2))
    zero_pad[:,0] = ref[:shift,0]

    # add zeropad and cut end by shift amount of points
    sample_shifted = np.concatenate((zero_pad, sample[:len(sample)-shift]))

    np.savetxt(dir_shifted / Path(str(file.split('/')[-1])), sample_shifted)

plt.plot(ref[:,0], ref[:,1])
plt.plot(sample_shifted[:,0], sample_shifted[:,1])
plt.show()
