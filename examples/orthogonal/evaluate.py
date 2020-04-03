"""
Has to be run in ipython with %run.
"""

import numpy as np

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView, CutPlane
from silx.gui import qt

import matplotlib.pyplot as plt
plt.ion()

# load assembly and add the pre-alignment rolls
dct = np.load('assembled.npz')
Pjlk = dct['Pjlk']
Pjlk = np.pad(Pjlk, ((0,0), (10,10), (0,0)), mode='constant')
W = dct['W_ortho']
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
ax[1, 0].imshow(np.flip(Pjlk.sum(axis=1), axis=0), aspect='auto', vmax=.5)
ax[1, 1].imshow(Pjlk.sum(axis=0), aspect='auto', vmax=.5)

# show the intensity
app = qt.QApplication([])
window = ScalarFieldView()
window.setData(W[6:-6])
window.setScale(1, 1, 1) # voxel sizes
window.addIsosurface(W.max()/500, '#FF0000AA')

cut = CutPlane(window)
cut.setVisible(True)
cut.setPoint((20,20,20), constraint=False)
cut.setNormal((1, 0, 0))
cut.moveToCenter()
print(cut.isValid())

window.show()
app.exec_()
