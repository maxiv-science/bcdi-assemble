import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from lib.reconstruction import M, C
from lib.reconstruction import generate_envelope, generate_initial, first_axis_com

### the number of theta bins of the model
Nj = 10

### load and plot generated data
filename = 'data/ten_simulations_4.npz'
data = np.load(filename)['frames']
o = np.load(filename)['offsets']
ax = plt.gca()
ax.plot(o, 'r', label='theta')
ax.plot([0, len(o)-1], [o.min(), o.max()], 'k--')
r = -np.load(filename)['rolls']
ax0twin = plt.twinx(ax)
ax0twin.plot(r, 'b', label='phi')
ax0twin.legend(loc='upper right')
ax.legend(loc='upper left')
ax.set_title('rocking and phi positions of the simulation')
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.abs(data[:,64,:]), vmax=np.abs(data[:,64,:]).max()/10)
ax[1].imshow(np.abs(data[len(data)//2,:,:]), vmax=np.abs(data[len(data)//2]).max()/10)
ax[0].set_title('frames seen from above')
ax[1].set_title('a central frame')

### build the the autocorrelation envelope and the initial model
envelope = generate_envelope(Nj, data.shape[-1], support=.5)
W = generate_initial(data, Nj)

### iteration time!
fig, ax = plt.subplots(ncols=5, figsize=(12,3))
errors = []
fudge = 1e-4
for i in range(200):

    W, Pjk = M(W, data, beta=fudge, force_continuity=True)
    W, error = C(W, envelope)
    errors.append(error)

    # expand the resolution now and then
    if i and (Nj<20) and (i % 20) == 0:
        fudge *= np.sqrt(2)
        W = np.pad(W, ((1,1),(0,0),(0,0)))
        Pjk = np.pad(Pjk, ((1,1),(0,0)))
        Nj = W.shape[0]
        envelope = generate_envelope(Nj, data.shape[-1], support=.5)
        print('increased Nj to %u'%Nj)

    [a.clear() for a in ax]
    ax[0].imshow(np.abs(W[:,64,:]), vmax=np.abs(W[:,64,:]).max()/10)
    ax[1].imshow(np.abs(W[Nj//2,:,:]), vmax=np.abs(W[Nj//2]).max()/10)
    ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/10)
    ax[3].plot(first_axis_com(Pjk))
    ax[4].plot(errors)
    ax[0].set_title('model from above')
    ax[1].set_title('central model slice')
    ax[2].set_title('|Pjk|')
    ax[3].set_title('Most likely angular\nbin per frame')
    ax[4].set_title('Error')
    plt.pause(.01)

### plot the result and compare to the simulated truth
fix, ax = plt.subplots(nrows=2, sharex=True, figsize=(4,4))
ax[0].plot(o)
if np.argmax(Pjk[:,-1]) > np.argmax(Pjk[:,0]):
    Pjk = np.flipud(Pjk)
ax[1].imshow(Pjk, aspect='auto')
ax[0].set_title('simulated theta')
ax[1].set_title('reconstructed |Pjk|')
