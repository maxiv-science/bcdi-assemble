"""
Picks pre-defined frames from a burst, interpolates the mask and crops. Saves
data together with the location of the phi rotation center in the frame's
coordinates.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

scan, pos = 111, 207
begin, end = 0, -1

import sys, os
PATH = sys.argv[1]
merlin_base = os.path.join(PATH, 'data1/scan_%04d_merlin_0000.hdf5')
mask_file = os.path.join(PATH, 'data1/merlin_mask_190222_14keV.h5')

# load and mask data
with h5py.File(mask_file, 'r') as fp:
    mask = fp['mask'][:]
with h5py.File(merlin_base%scan, 'r') as fp:
    burst = fp['entry_%04d/measurement/Merlin/data' % pos][begin:end].astype(np.int32) # to allow negative masked values
masked = np.where(mask == 0)
burst[:, masked[0], masked[1]] = -1

# find the brightest frame and its center
brightest = np.unravel_index(np.argmax(burst), burst.shape)[0]
center = np.unravel_index(np.argmax(burst[brightest]), burst[brightest].shape)

# crop the whole burst around this pixel
shape = 150
before_i = -min(center[0] - shape/2, 0)
before_j = -min(center[1] - shape/2, 0)
after_i = max(center[0] + shape/2 - burst.shape[-2], 0)
after_j = max(center[1] + shape/2 - burst.shape[-1], 0)
burst = np.pad(burst, 
               pad_width=((0, 0), (before_i, after_i), (before_j, after_j)),
               mode='constant', constant_values=0)
burst = burst[:,
              before_i+center[0]-shape//2:before_i+center[0]+shape//2,
              before_j+center[1]-shape//2:before_j+center[1]+shape//2]

# work out where the center of phi rotation will be in the new frame's
# axis coordinates.
phi_center = np.array((2600, 340))  # original frame
phi_center = phi_center - np.array(center) # relative to peak center
phi_center = phi_center + shape / 2  # relative to new frame's origin

np.savez('picked.npz', data=burst, center=phi_center)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.log10(burst[brightest]))
ax[1].plot(np.max(burst, axis=(1,2)))

plt.show()

