"""
Does not consider the q-space pixel sizes, just shifts the COM
to the center and pads to make the third dimension as long as
the other two.
"""

import h5py
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

for subset in (0, 1, 10):
	data = np.load('assembled_%u.npz'%subset)['W'][2:28]

	# pad to equal side, it gets weird otherwise
	pad = data.shape[-1] - data.shape[0]
	before = (pad + 1) // 2
	after = pad // 2
	data = np.pad(data, ((before, after), (0, 0), (0, 0)), mode='constant', constant_values=0)

	# we can at least roll the peak to the COM - no we can't, this causes a phase ramp
	com = np.sum(np.indices(data.shape) * data, axis=(1,2,3)) / np.sum(data)
	maxpos = np.unravel_index(np.argmax(data), data.shape)
	shifts = (np.array(data.shape)//2 - np.round(maxpos)).astype(np.int)
	data = np.roll(data, shifts, axis=(0,1,2))
	print('com at %s, maximum at %s'%(com, maxpos))
	print('shifting by %s'%shifts)

	print('maximum data pixel was %u'%data.max())
	np.savez('prepared_%u.npz'%subset, data=np.round((data*100)).astype(int))
