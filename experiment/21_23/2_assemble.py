"""
Exemplifies how to run an assembly with real units in the q3 axis,
which then allows resampling the data on an orthogonal grid.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from diffassemble.utils import C, M
from diffassemble.utils import generate_initial, generate_envelope, pre_align_rolls
from diffassemble.utils import ProgressPlot, rectify

for subset in (10, 1, 0):

    # input and parameters
    data_ = np.load('picked.npz')['data']
    Nj, Nl, ml = 20, 4, 1
    Nj_max = 32
    fudge = 2e-3
    increase_every = 5
    fudge_max = 1
    CENTER = [4000,256]
    if subset == 10:
        print('now doing the whole dataset')
        data = data_
    else:
        inds = list(range(subset, len(data_), 2))
        print('now doing subset %u: %s'%(subset, inds,))
        data = data_[inds]

    # physics
    theta = 15
    psize = 55e-6
    distance = .320
    hc = 4.136e-15 * 3.000e8
    E = 10000.
    Q12 = psize * 2 * np.pi / distance / (hc / E) * data.shape[-1]
    Q3 = Q12 / 150 * 30 # first minimum is around 30 pixels across, so Nj=30 should give a 1:1 aspect ratio with the first minimum on the q3 edges
    dq3 = Q3 / Nj
    Dmax = 60e-9

    # do the assembly, plotting on each iteration
    data, rolls = pre_align_rolls(data, roll_center=CENTER)
    envelope = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
    W = generate_initial(data, Nj)
    p = ProgressPlot()
    errors = []
    for i in range(60):
        print(i)
        W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
                            nproc=24,
                            roll_center=CENTER)
        [print(k, '%.3f'%v) for k, v in timing.items()]
        W, error = C(W, envelope)
        errors.append(error)
        p.update(np.log10(W), Pjlk, errors, vmax=1)

        # expand the resolution now and then
        if i and (Nj<Nj_max) and (i % increase_every) == 0:
            W = np.pad(W, ((2,2),(0,0),(0,0)))
            Nj = W.shape[0]
            Q3 = dq3 * Nj
            envelope = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
            print('increased Nj to %u and Q3 to %.2e'%(Nj, Q3))
        if i and (fudge < fudge_max) and (i % increase_every) == 0:
            fudge *= 2**(1/2)
            print('increased fudge to %e'%(fudge))

    # assuming that we now know the q-range, we can interpolate to qx, qy, qz
    # the outermost q3 slices contain everything that couldn't be assigned well.
    oldshape = W.shape[0]
    Q3 = W.shape[0] / oldshape * Q3
    W_ortho, Qnew = rectify(W, (Q3, Q12, Q12), theta)

    np.savez('assembled_%u.npz'%subset, W=W, W_ortho=W_ortho, Pjlk=Pjlk, rolls=rolls, Q_ortho=Qnew, Q=(Q3, Q12, Q12))
