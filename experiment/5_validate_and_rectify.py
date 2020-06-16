"""
Fourier shell correlation of reconstructed subsets.

Also rectifies and resamples the full reconstruction on a regular grid
with aspec ratio 1:1:1, for easier visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from diffassemble.utils import rectify_sample
#plt.ion()

# shifting makes a difference so think about aligning
# the phases have to be aligned to get the q=0 normalization right
# pre-envoloping the particle needed perhaps

folder = './'

def load(fn, N, cutoff=.1):
    with h5py.File(fn, 'r') as fp:
        p = fp['entry_1/data_1/data'][0]
    p[np.abs(p) < np.abs(p).max() * cutoff] = 0
    p = np.pad(p, N//2, mode='constant')
    com = np.sum(np.indices(p.shape) * np.abs(p), axis=(1,2,3)) / np.sum(np.abs(p))
    com = np.round(com).astype(int)
    p = p[com[0]-N//2:com[0]+N//2,
            com[1]-N//2:com[1]+N//2,
            com[2]-N//2:com[2]+N//2,]
    p[:] = p * np.exp(-1j * np.angle(p[N//2, N//2, N//2]))
    return p

# load the q space scales and work out the new q range after cropping etc
N = 100
Q3, Q1, Q2 = np.load(folder+'assembled_10.npz')['Q'] # full range of original assembly
nq3, nq1, nq2 = np.load(folder+'assembled_10.npz')['W'].shape # original shape
dq3, dq1, dq2 = Q3/nq3, Q1/nq1, Q2/nq2 # original q space pixel sizes
N_recons = np.load(folder+'prepared_10.npz')['data'].shape
Q3, Q1, Q2 = np.array((dq3, dq1, dq2)) * N_recons # full q range used in the reconstruction
# resolution: res * qmax = 2 pi
#   - if qmax is half the q range (origin to edge) then res is the full period resolution
#   - if qmax is the full q range (edge to edge) then res is the pixel size
dr3, dr1, dr2 = (2 * np.pi / q for q in (Q3, Q1, Q2)) # half-period res (pixel size)
p0 = load(folder+'modes_0.h5', N)
p1 = load(folder+'modes_1.h5', N)

# threshold the particles
p0[np.abs(p0) < np.abs(p0).max()*.25] = 0
p1[np.abs(p1) < np.abs(p1).max()*.25] = 0

# plot the input data and FT:s
fig, ax = plt.subplots(ncols=2, nrows=2)
ext = np.array((-dr2*N/2, dr2*N/2, -dr1*N/2, dr1*N/2)) * 1e9
ax[0, 0].imshow(np.abs(p0[N//2]), extent=ext)
ax[0, 1].imshow(np.abs(p1[N//2]), extent=ext)
plt.setp(ax[0, 0], xlim=[-50,50], ylim=[-50,50])
plt.setp(ax[0, 1], xlim=[-50,50], ylim=[-50,50])
f0 = np.fft.fftshift(np.fft.fftn(p0))
f1 = np.fft.fftshift(np.fft.fftn(p1))
ax[1, 0].imshow(np.log10(np.abs(f0[N//2])))
ax[1, 1].imshow(np.log10(np.abs(f1[N//2])))
fig.suptitle('input images and FT:s')

# plot along all three axes to understand the aspect ratio
fig, ax = plt.subplots(ncols=3)
# from the front
ext = np.array((-dr2*N/2, dr2*N/2, -dr1*N/2, dr1*N/2)) * 1e9
ax[0].imshow(np.abs(p0).sum(axis=0), extent=ext)
plt.setp(ax[0], xlim=[-50,50], ylim=[-50,50], title='front view',
         xlabel='r2', ylabel='r1')
# from the top
ext = np.array((-dr2*N/2, dr2*N/2, -dr3*N/2, dr3*N/2)) * 1e9
im = np.flip(np.abs(p0).sum(axis=1), axis=0)
ax[1].imshow(im, extent=ext)
plt.setp(ax[1], xlim=[-50,50], ylim=[-50,50], title='top view',
         xlabel='r2', ylabel='r3')
# from the side
ext = np.array((-dr3*N/2, dr3*N/2, -dr1*N/2, dr1*N/2)) * 1e9
im = np.transpose(np.abs(p0).sum(axis=2))
ax[2].imshow(im, extent=ext)
plt.setp(ax[2], xlim=[-50,50], ylim=[-50,50], title='side view',
         xlabel='r3', ylabel='r1')
fig.suptitle('2d projections')

# calculate and plot the FSC
# so Q1, Q2, Q3 is still the full q-range corresponding to dr3, dr1, dr2
# but we have cropped the sample to N so need a new q space pixel size
dq = np.array((Q3, Q1, Q2)) / N
# 3d q components:
q3d = (np.indices(p0.shape) - N//2) * dq.reshape((3,1,1,1))
q = np.sqrt(np.sum(q3d**2, axis=0)) # q is now |q|
qstep = 3*dq1 # the size of a q bin - arbitrary
qbins3d = (q // qstep).astype(int)
qbins2d = qbins3d[N//2]
fsc = []
for i in np.unique(qbins3d):
    shell = np.where(qbins3d==i)
    val = np.sum(f0[shell] * f1[shell].conj())
    val /= np.sqrt(np.sum(np.abs(f0[shell])**2))
    val /= np.sqrt(np.sum(np.abs(f1[shell])**2))
    fsc.append(val)
#    ring = np.where(qbins2d==i)
plt.figure()
a1 = plt.gca()
x = np.unique(qbins3d) * qstep * 1e-9
a1.add_patch(plt.Rectangle(xy=(0, .143), width=x.max(), height=(.5-.143), fc=(.8,)*3))
#imax = int(len(x) * .75)
#a1.plot(x[:imax], fsc[:imax], label='FSC (3d)')
a1.plot(x, fsc, label='FSC (3d)')
a1.set_xlabel('q = $2\pi/\Delta r$  [nm-1]')
a1.set_ylabel('Fourier shell correlation')
#a1.set_xlim(0, 1.)
a1.set_ylim(0, 1.1)
a2 = plt.gca().twiny()
a2.set_xlabel('Full-period resolution $\Delta r$  [nm]')
resolutions = (2, 5, 8, 10, 15, 12, 20, 30, 50)
a2.set_xticks([2*np.pi/dr for dr in resolutions])
a2.set_xticklabels(resolutions)
a2.set_xlim(a1.get_xlim())
np.savez('validation.npz', q=x, fsc=fsc)
#a1.legend()

# rectify the full reconstruction, save and plot
p = load(folder+'modes_10.h5', N)
#p = np.flip(p, axis=0)
p, psize = rectify_sample(p, (dr3, dr1, dr2), 15.0, find_order=False) # x z y
np.savez('rectified.npz', data=p, psize=psize)

# plot along all three axes to understand the aspect ratio
fig, ax = plt.subplots(ncols=3)
ext = np.array((-psize/2*N, psize/2*N, -psize/2*N, psize/2*N)) * 1e9 # valid after the operations below
# from the front
ax[0].imshow(np.abs(p).sum(axis=0), extent=ext)
plt.setp(ax[0], xlim=[-50,50], ylim=[-50,50], title='front view',
         xlabel='y', ylabel='z')
# from the top
im = np.flip(np.abs(p).sum(axis=1), axis=0)
ax[1].imshow(im, extent=ext)
plt.setp(ax[1], xlim=[-50,50], ylim=[-50,50], title='top view',
         xlabel='y', ylabel='x')
# from the side
im = np.transpose(im)
ax[2].imshow(im, extent=ext)
plt.setp(ax[2], xlim=[-50,50], ylim=[-50,50], title='side view',
         xlabel='x', ylabel='z')
fig.suptitle('resampled on orthogonal grids')

plt.show()