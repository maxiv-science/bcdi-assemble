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

# cut out the worst crap
data = data[5:-5]

# then pad for an equal pixel number
add = data.shape[-1] - data.shape[0] 
before = (add + 1) // 2
after = add // 2
data = np.pad(data, ((before, after), (0,0), (0,0)), mode='constant')

print('maximum data pixel was %u'%data.max())
np.savez('prepared.npz', data=(data*10).astype(int))
