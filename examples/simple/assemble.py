"""
Exemplifies how to run a simple assembly with unitless constraints.
"""

import numpy as np
from bcdiass.utils import *
from bcdiass.utils import ProgressPlot

# input and parameters
data = np.load('data.npz')['frames']
Nj, Nl = 25, 10
Nj_max = 50
fudge = 5e-5

# do the assembly, plotting on each iteration
envelope = generate_envelope(Nj, data.shape[-1], support=.2)
W = generate_initial(data, Nj)
p = ProgressPlot()
errors = []
data, rolls = pre_align_rolls(data, roll_center=None)
for i in range(100):
    print(i)
    W, Pjlk, timing = M(W, data, Nl=Nl, beta=fudge)
    W, error = C(W, envelope)
    errors.append(error)
    p.update(W, Pjlk, errors)

    # expand the resolution now and then
    if i and (Nj<Nj_max) and (i % 5) == 0:
        W = np.pad(W, ((2,2),(0,0),(0,0)))
        Nj = W.shape[0]
        envelope = generate_envelope(Nj, data.shape[-1], support=.25)
        fudge *= 2**(1/2)
        print('increased Nj to %u and fudge to %.2e'%(Nj, fudge))

np.savez('assembled.npz', W=W, Pjlk=Pjlk, rolls=rolls)
