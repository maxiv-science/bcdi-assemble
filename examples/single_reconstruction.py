import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from diffassemble import assemble

# input and parameters
SIMFILE = '../data/ten_simulations_9.npz'

# first plot the simulated truth
dct = np.load(SIMFILE)
data = dct['frames']
o = dct['offsets']
ax = plt.gca()
ax.plot(o, 'r', label='theta')
ax.plot([0, len(o)-1], [o.min(), o.max()], 'k--')
r = -dct['rolls']
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

# then do the assembly, plotting on each iteration
fig, ax = plt.subplots(ncols=5, figsize=(12,3))
plt.pause(.1)
for output in assemble(data, Nl=5, n_iter=6):
    W, Pjlk, errors = output
    Nj = Pjlk.shape[0]
    [a.clear() for a in ax]
    ax[0].imshow(np.abs(W[:,64,:]), vmax=np.abs(W[:,64,:]).max()/10)
    ax[1].imshow(np.abs(W[Nj//2,:,:]), vmax=np.abs(W[Nj//2]).max()/10)
    Pjk = np.sum(Pjlk, axis=1)
    ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/10)
    Plk = np.sum(Pjlk, axis=0)
    ax[3].imshow(np.abs(Plk))#, vmax=np.abs(Plk).max()/10)
    ax[4].plot(errors)
    ax[0].set_title('model from above')
    ax[1].set_title('central model slice')
    ax[2].set_title('|Pjk|')
    ax[3].set_title('|Plk|')
    ax[4].set_title('Error')
    plt.draw()
    plt.pause(.01)
np.savez(SIMFILE + '_assembled.npz', data=W)

# then plot the result together with the simulation
fix, ax = plt.subplots(nrows=3, sharex=True, figsize=(4,4))
try:
    ax[0].plot(o, 'r')
    ax0twin = plt.twinx(ax[0])
    ax0twin.plot(r, 'b')
    ax[0].set_title('simulated theta and phi')
except NameError:
    pass
Pjk = np.sum(Pjlk, axis=1)
Plk = np.sum(Pjlk, axis=0)
if np.argmax(Pjk[:,-1]) > np.argmax(Pjk[:,0]):
    Pjk = np.flipud(Pjk)
    Plk = np.flipud(Plk)
ax[1].imshow(Pjk, aspect='auto')
ax[2].imshow(Plk, aspect='auto')
ax[1].set_title('reconstructed |Pjk|')
ax[2].set_title('reconstructed |Plk|')
