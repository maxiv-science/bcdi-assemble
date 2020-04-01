import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# load assembly and add the pre-alignment rolls
dct = np.load('assembled.npz')
Pjlk = dct['Pjlk']
Pjlk = np.pad(Pjlk, ((0,0), (10,10), (0,0)))
W = dct['W']
prerolls = dct['rolls']
for k in range(Pjlk.shape[-1]):
    Pjlk[:, :, k] = np.roll(Pjlk[:, :, k], prerolls[k], axis=1)

# load the truths
dct = np.load('data.npz')
phi = dct['rolls']
theta = dct['offsets']

# make some plots
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].plot(theta)
ax[0, 1].plot(phi)
ax[1, 0].imshow(np.flip(Pjlk.sum(axis=1), axis=0), aspect='auto')
ax[1, 1].imshow(Pjlk.sum(axis=0), aspect='auto')
