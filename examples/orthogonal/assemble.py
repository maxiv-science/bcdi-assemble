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

# input and parameters
data = np.load('data.npz')['frames']
Nj, Nl, ml = 25, 10, 1
Nj_max = 50
fudge = 5e-5
increase_Nj_every = 5
increase_fudge_every = 5
increase_fudge_by = 2**(1/2)
fudge_max = 3e-4

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
Q3 = Q12 / 128 * 25 # first minimum is 25 pixels wide
dq3 = Q3 / Nj
Dmax = 60e-9
print('%e %e'%(Q3, Q12))

# do the assembly, plotting on each iteration
data, rolls = pre_align_rolls(data, roll_center=[200,64])
envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
W = generate_initial(data, Nj)
p = ProgressPlot()
errors = []
for i in range(50):
    print(i)
    W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
                        force_continuity=True, nproc=24,
                        roll_center=[200,64])
    [print(k, '%.3f'%v) for k, v in timing.items()]
    W, error = C(W, envelope1*envelope2)
    errors.append(error)
    p.update(W, Pjlk, errors)

    # expand the resolution now and then
    if i and (Nj<Nj_max) and (i % increase_Nj_every) == 0:
        W = np.pad(W, ((2,2),(0,0),(0,0)))
        Nj = W.shape[0]
        Q3 = dq3 * Nj
        envelope1 = generate_envelope(Nj, data.shape[-1], Q=(Q3, Q12, Q12), Dmax=(Dmax, 1, 1), theta=theta)
        envelope2 = generate_envelope(Nj, data.shape[-1], support=(1, .25, .25))
        print('increased Nj to %u'%Nj)

    if i and (fudge < fudge_max) and (i % increase_fudge_every) == 0:
        fudge *= increase_fudge_by
        print('increased fudge to %e'%fudge)

# assuming that we now know the q-range, we can interpolate to qx, qy, qz
W_ortho, Qnew = rectify(W, (Q12, Q12, Q3), theta)

np.savez('assembled.npz', W=W, W_ortho=W_ortho, Pjlk=Pjlk, rolls=rolls, Q=Qnew)
