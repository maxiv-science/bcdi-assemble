import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from lib.reconstruction import M, C
from lib.reconstruction import generate_initial

### the number of theta bins of the model
N = 15

### load and plot generated data
filename = 'data/test_data.npz'
data = np.load(filename)['frames'] * 1e-6
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

### build and plot the initial model from the autocorrelation envelope
W, envelope = generate_initial(N, data.shape[-1], support=.5)

### iteration time!
priors = np.linspace(0, N, len(data))
fig, ax = plt.subplots(ncols=5, figsize=(12,3))
errors = []
for i in range(50):

    W, priors, Pjk = M(W, priors, data)
    W, error = C(W, envelope)
    errors.append(error)

    [a.clear() for a in ax]
    ax[0].imshow(np.abs(W[:,64,:]), vmax=np.abs(W[:,64,:]).max()/10)
    ax[1].imshow(np.abs(W[N//2,:,:]), vmax=np.abs(W[N//2]).max()/10)
    ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/10)
    ax[3].plot(priors)
    ax[4].plot(errors)
    ax[0].set_title('model from above')
    ax[1].set_title('central model slice')
    ax[2].set_title('|Pjk|')
    ax[3].set_title('Most likely angular\nbin per frame')
    ax[4].set_title('Error')
    plt.pause(.01)

### plot the result and compare to the simulated truth
fix, ax = plt.subplots(nrows=3, sharex=True, figsize=(4,6))
ax[0].plot(o)
ax[1].plot(o)
ax[1].set_ylim([-.2,.3])
ax[2].imshow(np.flipud(np.abs(Pjk)), aspect='auto')
ax[0].set_title('simulated theta')
ax[1].set_title('simulated theta - closeup')
ax[2].set_title('reconstructed |Pjk|')