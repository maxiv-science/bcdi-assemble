"""
Visualizes a single burst from a given scan as a video.
"""

import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#plt.ion()

scan, pos = 21, 23
begin, end = 0, 800

import sys, os
PATH = sys.argv[1]
filename = os.path.join(PATH, 'data2/%06u.h5' % scan)
mask_file = os.path.join(PATH, 'data2/merlin_mask_200430_8keV.h5')

# load and mask data
with h5py.File(mask_file, 'r') as fp:
    mask = fp['mask'][:]
with h5py.File(filename, 'r') as fp:
    burst = fp['entry/measurement/merlin/frames'][pos*1000:(pos+1)*1000].astype(np.int32) # to allow negative masked values
    burst = burst[begin:end]
masked = np.where(mask == 0)
burst[:, masked[0], masked[1]] = -1

# find the brightest frame and its center
brightest = np.unravel_index(np.argmax(burst), burst.shape)[0]
center = np.unravel_index(np.argmax(burst[brightest]), burst[brightest].shape)

# crop the whole burst around this pixel
shape = 150
before_i = -int(min(center[0] - shape/2, 0))
before_j = -int(min(center[1] - shape/2, 0))
after_i = int(max(center[0] + shape/2 - burst.shape[-2], 0))
after_j = int(max(center[1] + shape/2 - burst.shape[-1], 0))
burst = np.pad(burst, 
               pad_width=((0, 0), (before_i, after_i), (before_j, after_j)),
               mode='constant', constant_values=0)
burst = burst[:,
              before_i+center[0]-shape//2:before_i+center[0]+shape//2,
              before_j+center[1]-shape//2:before_j+center[1]+shape//2]
np.savez('picked.npz', data=burst)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.log10(burst[brightest]))
ax[1].plot(np.max(burst, axis=(1,2)))

plt.show()
