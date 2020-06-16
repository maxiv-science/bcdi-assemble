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

for subset in (0, 10, 1,):
    # input and parameters
    data_ = np.load('picked.npz')['data']
    Nj_max = 32
    increase_Nj_every = 5
    increase_fudge_every = 5
    increase_fudge_by = 2**(1/2)
    fudge_max = 4e-3
    CENTER = [2600,340]
    if subset == 10:
        print('now doing the whole dataset')
        data = data_
    else:
        inds = list(range(subset, len(data_), 2))
        print('now doing subset %u: %s'%(subset, inds,))
        data = data_[inds]

    # physics
    a = 4.065e-10
    E = 10000.
    d = a / np.sqrt(3) # (111)
    hc = 4.136e-15 * 3.000e8
    theta = np.arcsin(hc / (2 * d * E)) / np.pi * 180
    psize = 55e-6
    distance = .25
    # detector plane: dq = |k| * dtheta(pixel-pixel)
    Q12 = psize * 2 * np.pi / distance / (hc / E) * data.shape[-1]

    Nj, Nl, ml = 16, 4, 1
    fudge = 5e-4
    Q3 = Q12 / 150 * 15 # first minimum is 25 pixels wide
    Dmax = 60e-9
    print('%e %e'%(Q3, Q12))
    dq3 = Q3 / Nj

    # do the assembly, plotting on each iteration
    data, rolls = pre_align_rolls(data, roll_center=CENTER)
    envelope = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
    envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
    W = generate_initial(data, Nj)
    p = ProgressPlot()
    errors = []
    for i in range(60):
        print(i)
        W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
                            force_continuity=3, nproc=24,
                            roll_center=CENTER)
        [print(k, '%.3f'%v) for k, v in timing.items()]
        if i < 0:
            W, error = C(W, envelope*envelope2)
        else:
            W, error = C(W, envelope)#*envelope2)
        errors.append(error)
        p.update(np.log10(W), Pjlk, errors, vmax=1)

        # expand the resolution now and then
        if i and (Nj<Nj_max) and (i % increase_Nj_every) == 0:
            W = np.pad(W, ((2,2),(0,0),(0,0)))
            Nj = W.shape[0]
            Q3 = dq3 * Nj
            envelope = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
            envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
            print('increased Nj to %u'%Nj)

        if i and (fudge < fudge_max) and (i % increase_fudge_every) == 0:
            fudge *= increase_fudge_by
            print('increased fudge to %e'%fudge)

    # assuming that we now know the q-range, we can interpolate to qx, qy, qz
    # the outermost q3 slices contain everything that couldn't be assigned well.
    oldshape = W.shape[0]
    Q3 = W.shape[0] / oldshape * Q3
    W_ortho, Qnew = rectify(W, (Q3, Q12, Q12), theta)

    np.savez('assembled_%u.npz'%subset, W=W, W_ortho=W_ortho, Pjlk=Pjlk, rolls=rolls, Q=Qnew)
