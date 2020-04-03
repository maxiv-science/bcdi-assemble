import h5py
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import os

assfiles = [f for f in os.listdir() if 'assembled_' in f and f.endswith('.npz')]

for filename in assfiles:
	print(filename)
	data = np.load(filename)['W_ortho']
	strain = filename.split('.npz')[0].split('assembled_')[1]

	# we can at least roll the peak to the center
	com = np.sum(np.indices(data.shape) * data, axis=(1,2,3)) / np.sum(data)
	shifts = (np.array(data.shape)//2 - np.round(com)).astype(np.int)
	data = np.roll(data, shifts, axis=(0,1,2))

	data = data[8:-8]

	# cropped_resampled:
	# first resample so that the Bragg peak is roughly isotropic,
	# then pad the rest to equal size.
	brightest = np.array(np.unravel_index(data.argmax(), data.shape))
	inds = np.indices(data.shape) - brightest[:,None,None,None]
	bi, bj, bk = brightest
	extents = [
	    np.sum(np.abs(data[:, bj, bk] * (np.arange(data.shape[0]) - data.shape[0]/2))),
	    np.sum(np.abs(data[bi, :, bk] * (np.arange(data.shape[1]) - data.shape[1]/2))),
	    np.sum(np.abs(data[bi, bj, :] * (np.arange(data.shape[2]) - data.shape[2]/2)))
	    ]
	ratio = extents[0] / (extents[1] + extents[2]) * 2
	ii, jj, kk = np.indices((data.shape[-1],)*3)
	ii = ii * ratio
	ii = ii - np.mean(ii) + data.shape[0]//2
	new_data = map_coordinates(data, (ii, jj, kk), mode='constant', cval=0.0)
	print('writing cropped and resampled data %s'%(new_data.shape,))

	print('maximum data pixel was %u'%new_data.max())
	np.savez('prepared_%s.npz'%strain.replace('.', ''), data=(new_data*10).astype(int))
