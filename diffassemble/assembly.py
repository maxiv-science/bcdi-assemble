import numpy as np
import h5py

from .utils import C, M, build_model, generate_initial, generate_envelope

def assemble(data, Nj=20, Nl=20, ml=1, n_iter=100,
             Nj_max=50, increase_Nj_every=5, fudge=5e-5, fudge_max=1,
             increase_fudge_every=10, increase_fudge_by=2**(1/2),
             pre_align_phi=True, support=.25, nproc=4):
    """
    Generator which performs the diffraction volume assembly and all its
    parameter logistics, and yields on every iteration so you can plot
    or so.
    """

    ### first hack the data to align the centers of mass of each frame
    rolls = np.zeros(len(data), dtype=np.int)
    if pre_align_phi:
        ii, jj = np.indices(data[0].shape)
        for k in range(len(data)):
            com = np.sum(jj * data[k]) / np.sum(data[k])
            shift = int(np.round(data.shape[-1]//2 - com))
            data[k] = np.roll(data[k], shift, axis=-1)
            rolls[k] = shift

    ### build the the autocorrelation envelope and the initial model
    envelope = generate_envelope(Nj, data.shape[-1], support=support, type='sphere')
    W = generate_initial(data, Nj)

    ### iteration time!
    errors = []
    with h5py.File('assembly_progress.h5', 'w') as ofp:
        for i in range(n_iter):
            print(i)

            W, Pjlk, timing = M(W, data, Nl=Nl, ml=ml, beta=fudge, force_continuity=True, nproc=nproc)
            [print(k, '%.3f'%v) for k, v in timing.items()]
            W, error = C(W, envelope)
            errors.append(error)
            ofp.create_group('entry%02u'%i)
            ofp['entry%02u/W'%i] = W
            ofp['entry%02u/Pjlk'%i] = Pjlk

            # expand the resolution now and then
            if i and (Nj<Nj_max) and (i % increase_Nj_every) == 0:
                W = np.pad(W, ((2,2),(0,0),(0,0)))
                Nj = W.shape[0]
                envelope = generate_envelope(Nj, data.shape[-1], support=support, type='sphere')
                print('increased Nj to %u'%Nj)

            if i and (fudge<fudge_max) and (i % increase_fudge_every) == 0:
                fudge *= increase_fudge_by
                print('increased fudge to %e'%fudge)

            yield W, Pjlk, errors
