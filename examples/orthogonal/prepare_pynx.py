"""
Does not consider the q-space pixel sizes, just shifts the COM
to the center and pads to make the third dimension as long as
the other two.
"""

import h5py
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

data = np.load('assembled.npz')['W_ortho']

# we can at least roll the peak to the center
com = np.sum(np.indices(data.shape) * data, axis=(1,2,3)) / np.sum(data)
shifts = (np.array(data.shape)//2 - np.round(com)).astype(np.int)
data = np.roll(data, shifts, axis=(0,1,2))

print('maximum data pixel was %u'%data.max())
np.savez('prepared.npz', data=(data*10).astype(int))
