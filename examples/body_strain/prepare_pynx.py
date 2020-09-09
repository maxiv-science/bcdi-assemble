import h5py
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import os

assfiles = [f for f in os.listdir() if 'assembled_' in f and f.endswith('.npz')]

for filename in assfiles:
	data = np.load(filename)['W'][5:-5]
	strain = filename.split('.npz')[0].split('assembled_')[1]

	# pad to equal side, it gets weird otherwise
	pad = data.shape[-1] - data.shape[0]
	before = (pad + 1) // 2
	after = pad // 2
	data = np.pad(data, ((before, after), (0, 0), (0, 0)), mode='constant', constant_values=0)

	# we can at least roll the peak to the center
	com = np.sum(np.indices(data.shape) * data, axis=(1,2,3)) / np.sum(data)
	shifts = (np.array(data.shape)//2 - np.round(com)).astype(np.int)
	data = np.roll(data, shifts, axis=(0,1,2))

	print('maximum data pixel was %u'%data.max())
	np.savez('prepared_%s.npz'%strain.replace('.', ''), data=(data*10).astype(int))
