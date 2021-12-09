import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import skimage.transform

import multiprocessing
from functools import partial

def first_axis_com(a):
    """
    Calculates the center of mass of an array along each column.
    """
    if a.ndim == 1:
        a = a[:, None]
    inds = np.arange(a.shape[0], dtype=int).reshape((-1, 1))
    return np.sum(a*inds, axis=0) / np.sum(a, axis=0)

def log_gauss(mu, x, sigma):
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    return np.log(norm) - 0.5 * ((x - mu) / sigma)**2

def roll(im, pixels, roll_center=None):
    """
    Roll the image, either by just rolling the array, or by rotating
    around a center.
    """
    if roll_center is None:
        rolled = np.roll(im, pixels, axis=-1)
    else:
        # mask out masked pixels to avoid interpolation weirdness
        dtype = im.dtype
        im = im.astype(float)
        im[np.where(im < 0)] = np.nan

        # approximate number of pixels per degree
        dist = np.sqrt(np.sum((np.array(roll_center) - np.array(im.shape)/2)**2))
        angle = pixels / dist / np.pi * 180
        # the roll angle is always positive and skimage rotates counter-clockwise,
        # but we want a positive roll to correspond to pixels to the right, so
        # change sign if center is below.
        if roll_center[0] > im.shape[0] / 2:
            angle *= -1
        rolled = skimage.transform.rotate(im, angle=angle,
                    center=roll_center[::-1], # skimage rotate takes (col, row)
                    mode='reflect', order=0)

        # restore the normal mask
        rolled[np.where(np.isnan(rolled))] = -1

        # cast back from float
        rolled = rolled.astype(dtype)

    return rolled

def inner_Mbuild(j, data, ml, roll_center, Pjlk):
    Npix = data.shape[-1]
    Nj, Nl, Nk = Pjlk.shape
    Wj = np.zeros((Npix, Npix), dtype=np.float64)
    norm = np.zeros_like(Wj)
    for l in range(Nl):
        for k in range(Nk):
            rolled = roll(data[k], ml*(l-Nl//2), roll_center)
            mask = (rolled >= 0)
            norm[:] += mask * Pjlk[j, l, k]
            Wj[:] = Wj + mask * Pjlk[j, l, k] * rolled
    Wj[:] = Wj / (norm + 1e-20)
    return Wj

def build_model(data, Pjlk, ml=1, roll_center=None, nproc=4):
    """
    Returns the maximum likelihood model W[j] based on the data[k] and
    probability matrix P[j,k]. This is Loh et al PRE 2009 eqn 11.
    """
    pool = multiprocessing.Pool(nproc)
    inner_ = partial(inner_Mbuild, data=data, ml=ml,
                     roll_center=roll_center, Pjlk=Pjlk)
    W = np.array(pool.map(inner_, range(Pjlk.shape[0])))
    pool.terminate()
    return W

def inner_Mcalc(j, W, data, ml, Nl, Nk, roll_center):
    """
    Broken out inner loops of the Rjlk calculation, for paralllelization.
    """
    logRjlk = np.empty((Nl, Nk), dtype=np.float64)
    for k in range(Nk):
        for l in range(Nl):
            rolled = roll(data[k], ml*(l-Nl//2), roll_center)
            mask = (rolled >= 0)
            logRjlk[l, k] = np.sum(mask * (rolled * np.log(W[j] + 1e-20) - W[j]))
    return logRjlk

def M(W, data, Nl=1, ml=1, beta=1.0, force_continuity=6, nproc=4,
      roll_center=None, find_direction=True):
    """
    Performs the M update rule, Loh et al PRE 2009 eqns 8-11, with the
    addition of the fudge factor from Ayyer et al J Appl Cryst 2016.

    force_continuity is the standard deviation (in j pixels) of the
    gaussian applied to enforce continuity of the rocking angle. Set
    to False or zero to lift this constraint.
    """
    t0 = time.time()
    Nj = W.shape[0]
    Nk = data.shape[0]

    # first, calculate the probabilities Pjlk based on the current model
    t1 = time.time()
    pool = multiprocessing.Pool(nproc)
    inner_ = partial(inner_Mcalc, W=W, data=data, ml=ml, Nl=Nl, Nk=Nk,
                     roll_center=roll_center)
    logRjlk = np.array(pool.map(inner_, range(Nj)))
    pool.terminate()
    logPjlk = beta * logRjlk

    # optionally force Pjk to describe something continuous
    t2 = time.time()
    if force_continuity == True:
        force_continuity == 6 # avoid automatic casing True->1, 6 is the default.
    if force_continuity:
        fc = force_continuity
        kmax = np.argmax(np.sum(data, axis=(1,2)))
        com = np.argmax(np.sum(logPjlk, axis=1)[:, kmax])
        logPjlk[:, :, kmax] += log_gauss(com, np.arange(Nj), fc)[:, None]
        for k in range(kmax+1, Nk):
            bias = np.argmax(np.sum(logPjlk, axis=1)[:, k-1])
            logPjlk[:, :, k] += log_gauss(bias, np.arange(Nj), fc)[:, None]
        for k in range(kmax-1, -1, -1):
            bias = np.argmax(np.sum(logPjlk, axis=1)[:, k+1])
            logPjlk[:, :, k] += log_gauss(bias, np.arange(Nj), fc)[:, None]
            
    # pragmatic pre-normalization to avoid overflow
    t3 = time.time()
    logPjlk -= np.max(logPjlk, axis=(0,1))
    Pjlk = np.exp(logPjlk)
    Pjlk /= np.sum(Pjlk, axis=(0,1))
    
    # then carry out the likelihood maximization (M) step
    W = build_model(data, Pjlk, ml=ml, roll_center=roll_center, nproc=nproc)

    # optionally analyze the rocking curve direction, which if
    # wrong would cause an erroneous Dmax. This builds on the
    # coordinate convention and the fact that the diffraction
    # intensity should be seen at higher q1 values for higher q3
    # values.
    if find_direction:
        vert_inds = np.indices(W.shape)[1]
        vert_com = np.sum(vert_inds * W, axis=(1,2)) / np.sum(W, axis=(1,2))
        vert_com = vert_com[Nj//4:-Nj//4]
        slope = np.sum((np.arange(len(vert_com)) - len(vert_com)/2) * (vert_com - np.mean(vert_com)))
        if slope > 0:
            print('slope is', slope, 'so flipping!')
            W = np.flip(W, axis=0)
            Pjlk = np.flip(Pjlk, axis=0)

    t4 = time.time()
    timing = {'total': t4-t0, 'Pjlk calculation': t2-t1,
              'continuity enforcement': t3-t2,
              'likelihood maximization': t4-t3}
    return W, Pjlk, timing

def C(W, envelope):
    ft = np.fft.fftn(W)
    error = np.sum(np.abs((1-envelope)*ft))
    W = np.abs(np.fft.ifftn(ft * envelope))
    return W, error

def generate_envelope(N, shape, support=0.25, Q=None, theta=0.0, Dmax=None):
    """
    Generates the envelope with which to constrain the autocorrelation
    function, effectively placing a low-pass filter on the assembled
    intensity.

    N:          the number of rocking positions (int)
    shape:      the linear dimension of each image (int)
    support:    fraction of the autocorrelation grid to keep,
                along each dimension (q3, q1, q2) where the first (q3)
                corresponds to Nj. This is directly applied in the
                natural coordinate system, so theta, Dmax and Q are
                ignored. Can also be a float.
    Q:          The q-ranges (Q3, Q1, Q2) spanned by q3, q1, and q2.
    Dmax:       The maximum extent of the particle described along each
                natural real-space dimension (r3, r1, r2), where the
                first (r3) corresponds to the third dimension. Can also
                be a float. See citation below.
    theta:      The Bragg angle in degrees.
    """
    envelope = np.ones((N, shape, shape), dtype=int)
    if Q is None:
        # no physics, just keep a fraction of the autocorrelation along
        # each natural sampling dimension.
        if np.isscalar(support):
            support = (support,) * 3
        n1 = int(np.floor(shape * (support[1] / 2)))
        n2 = int(np.floor(shape * (support[2] / 2)))
        n3 = int(np.floor(N * (support[0] / 2)))
        envelope[n3:-n3] = 0
        envelope[:, n1:-n1] = 0
        envelope[:, :, n2:-n2] = 0
    else:
        # take the geometry into account and confine the autocorrelation
        # in natural-coordinate real space. See this paper for details:
        # Berenguer et al, PRB 2013 10.1103/PhysRevB.88.144101
        if np.isscalar(Q):
            Q = (Q,) * 3
        Q3, Q1, Q2 = Q
        N3, N1, N2 = N, shape, shape
        dq1, dq2, dq3 = np.array((Q1, Q2, Q3)) / np.array((N1, N2, N3))
        costheta = np.cos(theta / 180. * np.pi)
        sintheta = np.sin(theta / 180. * np.pi)
        dr1 = 2 * np.pi / (N1 * dq1 * costheta)
        dr2 = 2 * np.pi / (N2 * dq2)
        dr3 = 2 * np.pi / (N3 * dq3 * costheta)
        r3, r1, r2 = (np.indices(envelope.shape))
        r1 = (r1 - N1 / 2) * dr1
        r2 = (r2 - N2 / 2) * dr2
        r3 = (r3 - N3 / 2) * dr3
        envelope[np.where((np.abs(r3) > Dmax[0]) |
                          (np.abs(r1) > Dmax[1]) |
                          (np.abs(r2) > Dmax[2]))] = 0
        envelope = np.roll(envelope, (N//2, shape//2, shape//2), axis=(0,1,2))
    return envelope

def generate_initial(data, Nj, sigma=1.):
    """
    Generates an initial model W based on a simle ramp,
    with no rotation.
    """
    initial = np.linspace(0, Nj-1, len(data))
    Nk = len(data)
    jj = np.indices((Nj, Nk))[0]
    Pjk = np.exp(-(initial-jj)**2 / sigma**2 / 2)
    Pjk /= Pjk.sum(axis=0)
    Pjlk = Pjk.reshape((Nj, 1, Nk))
    return build_model(data, Pjlk) + 1e-20

def pre_align_rolls(data, roll_center, threshold=None, plot=False):

    rolls = np.zeros(len(data), dtype=np.int)
    ii, jj = np.indices(data[0].shape)
    mask = (data[0] >= 0)

    if plot:
        plt.ion()
        fig, ax = plt.subplots(ncols=3)
        fig.suptitle('Pre-alignment of roll direction')
        ax[0].imshow(np.log10(data.sum(axis=1)))
        ax[0].set_title('raw data')

    if threshold:
        data_ = data.copy()
        data_[data_ < threshold] = 0
    else:
        data_ = data

    if plot:
        ax[1].imshow(np.log10(data_.sum(axis=1)))
        ax[1].set_title('thresholded')

    for k in range(len(data)):
        com = np.sum(jj * data_[k] * mask) / np.sum(data_[k] * mask)
        try:
            shift = int(np.round(data.shape[-1]//2 - com))
        except ValueError:
            print("not pre-aligning frame %d"%k)
            shift = 0
        data[k] = roll(data[k], shift, roll_center)
        rolls[k] = shift

    if plot:
        ax[2].imshow(np.log10(data.sum(axis=1)))
        ax[2].set_title('pre-aligned')

    return data, rolls

class ProgressPlot(object):
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(ncols=5, figsize=(12,3))
        plt.pause(.1)

    def update(self, W, Pjlk, errors, vmax=0.1):
        ax = self.ax
        Nj = Pjlk.shape[0]
        [a.clear() for a in ax]
        ax[0].imshow(W[:,64,:], vmax=W[:,64,:].max()*vmax)
        ax[1].imshow(W[Nj//2,:,:], vmax=W[Nj//2].max()*vmax)
        Pjk = np.sum(Pjlk, axis=1)
        ax[2].imshow(np.abs(Pjk), vmax=np.abs(Pjk).max()/10, aspect='auto')
        Plk = np.sum(Pjlk, axis=0)
        ax[3].imshow(np.abs(Plk), aspect='auto')#, vmax=np.abs(Plk).max()/10)
        ax[4].plot(errors)
        ax[0].set_title('model from above')
        ax[1].set_title('central model slice')
        ax[2].set_title('|Pjk|')
        ax[3].set_title('|Plk|')
        ax[4].set_title('Error')
        plt.draw()
        plt.pause(.01)

def mean_spread(a):
    inds = np.indices(a.shape)
    com = np.sum(inds * a, axis=(1,2,3)) / np.sum(a)
    dist = np.sum((inds - com.reshape((3,1,1,1)))**2, axis=0)**(1/2)
    spread = np.sum(a * dist) / np.sum(a)
    return spread

def rectify_volume(W, Q, theta, find_order=True):
    """
    Resamples the diffraction volume model W on an orthogonal grid.
    """
    W1, Qnew = _rectify_volume(W, Q, theta)
    W2, Qnew = _rectify_volume(np.flip(W, axis=0), Q, theta)
    if not find_order or (mean_spread(W1) < mean_spread(W2)):
        return W1, Qnew
    else:
        return W2, Qnew

def _rectify_volume(W, Q, theta):
    """
    Takes the model W, defined over the ranges Q, corresponding to the
    Bragg angle theta, and interpolates it on a regular, orthogonal grid.

    The model W is indexed as:
        q3 (low to high)
        q1 (high to low)
        q2 (low to high)

    Returns W, Qnew (where the latter is the new q-range).
    """
    Q3, Q1, Q2 = Q
    dq2 = Q1 / W.shape[-1]
    dq1 = -dq2 # image indices run q1 (high to low), q2 (low to high)
    dq3 = Q3 / W.shape[0]
    # make up the existing natural q grid
    qbase = np.indices(W.shape)
    qbase = qbase - (np.array(W.shape).reshape((3,1,1,1)) - 1) / 2
    q3_old, q1_old, q2_old = qbase * np.array((dq3, dq1, dq2)).reshape((3,1,1,1))
    # define a new q(xyz) grid and the corresponding q(123) coordinates
    costheta = np.cos(theta / 180 * np.pi)
    sintheta = np.sin(theta / 180 * np.pi)
    qx, qz, qy = qbase * np.array((dq3, dq1*costheta, dq2)).reshape((3,1,1,1))
    Qnew = (qx.ptp(), qy.ptp(), qz.ptp())
    q3_new = qx + qz * sintheta / costheta
    q2_new = qy
    q1_new = qz / costheta
    # work out what indices these values would have, and interpolate
    n3 = (q3_new - q3_old[0,0,0]) / dq3
    n2 = (q2_new - q2_old[0,0,0]) / dq2
    n1 = (q1_new - q1_old[0,0,0]) / dq1
    return map_coordinates(W, (n3, n1, n2)), Qnew

def rectify_sample(p, dr, theta, find_order=True, interp=3):
    """
    Resamples the reconstructed sample on an orthogonal grid.
    """
    p1, psize = _rectify_sample(p, dr, theta, interp=interp)
    p2, psize = _rectify_sample(np.flip(p, axis=0), dr, theta, interp=interp)
    if not find_order:
        return p1, psize
    elif mean_spread(p1) < mean_spread(p2):
        print('r3 axis seems to be right - not flipping.')
        return p1, psize
    else:
        print('flipping r3 axis - seems to be reversed!')
        return p2, psize

def _rectify_sample(p, dr, theta, interp=3):
    """
    Takes the sample p, samples with dr = (dr3, dr1, dr2), corresponding
    to the Bragg angle theta, and interpolates it on a regular, orthogonal
    grid.

    Indexing:
        r3 (low to high)
        r1 (high to low)
        r2 (low to high)

    The orthogonal frame p is indexed
        x (low to high)
        z (high to low)
        y (low to high)

    Returns p, psize
    """
    # make up the existing natural grid
    dr = np.abs(np.array(dr))
    dr[1] *= -1
    tmp = np.indices(p.shape) - (np.array(p.shape).reshape((3,1,1,1)) - 1) // 2
    r3_old, r1_old, r2_old = tmp * np.array(dr).reshape((3,1,1,1))
    dr3, dr1, dr2 = dr
    # define a new xyz grid and find its r123 coordinates
    costheta = np.cos(theta / 180 * np.pi)
    sintheta = np.sin(theta / 180 * np.pi)
    psize = np.abs(dr[1]) # |dr1|
    x, z, y = psize * (np.indices(p.shape) - (np.array(p.shape).reshape((3,1,1,1)) - 1) // 2)
    z = np.flip(z, axis=1) # to account for the decreasing indexing
    r1_new = z - sintheta/costheta * x
    r2_new = y
    r3_new = x / costheta
    # work out what indices these values would have, and interpolate
    n3 = (r3_new - r3_old[0,0,0]) / dr3
    n2 = (r2_new - r2_old[0,0,0]) / dr2
    n1 = (r1_old[0,0,0] - r1_new) / np.abs(dr1)
    ampl = map_coordinates(np.abs(p), (n3, n1, n2), order=interp)
    phase = map_coordinates(np.angle(p), (n3, n1, n2), order=interp)
    return ampl * np.exp(1j * phase), psize
