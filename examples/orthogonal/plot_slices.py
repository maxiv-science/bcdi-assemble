import h5py
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

phaselim = .2

N = 3 # the number of images on either side of the max
n = 2 # the step through the third dimension of the data
shape=32

with h5py.File('modes.h5', 'r') as fp:
    im = fp['entry_1/data_1/data'][0]
center = np.argmax(np.sum(np.abs(im), axis=(1,2)))
com = np.sum(np.indices(im.shape) * np.abs(im)[None,:,:,:], axis=(1,2,3)) / np.sum(np.abs(im))
com = np.round(com).astype(int)

im = im[:, com[1]-shape//2:com[1]+shape//2, com[2]-shape//2:com[2]+shape//2]
inds = [-i*n for i in range(1,N+1)][::-1] + [0,] + [i*n for i in range(1,N+1)]
im = im[np.array(inds) + int(round(com[0]))]
fig, ax = plt.subplots(ncols=len(im), nrows=2, figsize=(10,2), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)
mod = np.abs(im)
im *= np.exp(-1j * np.angle(im[im.shape[0]//2, im.shape[1]//2]))
phase = np.angle(im)
phase[mod < mod.max()/10] = np.nan
for i in range(len(im)):
    vmax = np.abs(im).max()
    ax[0, i].imshow(mod[i], vmin=vmax/10, vmax=vmax)
    pi = ax[1, i].imshow(phase[i], vmin=-phaselim, vmax=phaselim, cmap='jet')
fig.colorbar(pi, ax=ax[1,-1])
ax[0,0].set_ylabel('abs')
ax[1,0].set_ylabel('phase')

for a_ in ax.flatten():
    plt.setp(a_, 'xticklabels', [], 'yticklabels', [])
