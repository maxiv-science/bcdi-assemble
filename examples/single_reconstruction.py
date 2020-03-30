import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from diffassemble.utils import C, M
from diffassemble.utils import generate_initial, generate_envelope, pre_align_rolls
from diffassemble.utils import ProgressPlot

# input and parameters
SIMFILE = '../data/ten_simulations_9.npz'
Nj, Nl, ml = 25, 10, 1
Nj_max = 50
fudge = 5e-5
increase_Nj_every = 5
increase_fudge_every = 5
increase_fudge_by = 2**(1/2)
support = .25

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

# then plot the stack of frames as received - ouch!
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.abs(data[:,64,:]), vmax=np.abs(data[:,64,:]).max()/10)
ax[1].imshow(np.abs(data[len(data)//2,:,:]), vmax=np.abs(data[len(data)//2]).max()/10)
ax[0].set_title('frames seen from above')
ax[1].set_title('a central frame')

# then do the assembly, plotting on each iteration
data, rolls = pre_align_rolls(data, roll_center=None)
envelope = generate_envelope(Nj, data.shape[-1], support=support)
W = generate_initial(data, Nj)
p = ProgressPlot()
errors = []
for i in range(60):
    print(i)
    W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge,
                        force_continuity=True, nproc=4,
                        roll_center=None)
    [print(k, '%.3f'%v) for k, v in timing.items()]
    W, error = C(W, envelope)
    errors.append(error)
    p.update(W, Pjlk, errors)

    # expand the resolution now and then
    if i and (Nj<Nj_max) and (i % increase_Nj_every) == 0:
        W = np.pad(W, ((2,2),(0,0),(0,0)))
        Nj = W.shape[0]
        envelope = generate_envelope(Nj, data.shape[-1], support=support)
        print('increased Nj to %u'%Nj)

    if i and (i % increase_fudge_every) == 0:
        fudge *= increase_fudge_by
        print('increased fudge to %e'%fudge)

np.savez(SIMFILE + '_assembled.npz', data=W)
