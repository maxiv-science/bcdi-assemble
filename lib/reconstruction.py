import numpy as np

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

def build_model(data, Pjlk, ml=1):
    """
    Returns the maximum likelihood model W[j] based on the data[k] and
    probability matrix P[j,k]. This is Loh et al PRE 2009 eqn 11.
    """
    Nj, Nl, Nk = Pjlk.shape
    Npix = data.shape[-1]
    W = np.zeros((Nj, Npix, Npix), dtype=np.float64)
    for j in range(Nj):
        W[j][:] = 0.0
        for l in range(Nl):
            for k in range(Nk):
                rolled = np.roll(data[k], ml*(l-Nl//2), axis=-1)
                W[j][:] = W[j] + Pjlk[j, l, k] * rolled
        W[j][:] = W[j] / (np.sum(Pjlk[j, :, :]) + 1e-20)
    return W

def M(W, data, Nl=1, ml=1, beta=1.0, force_continuity=True):
    """
    Performs the M update rule, Loh et al PRE 2009 eqns 8-11, with the
    addition of the fudge factor from Ayyer et al J Appl Cryst 2016.
    """
    Nj = W.shape[0]
    Nk = data.shape[0]
    logRjlk = np.empty((Nj, Nl, Nk), dtype=np.float64)

    # first, calculate the probabilities Pjlk based on the current model
    for j in range(Nj):
        for k in range(Nk):
            for l in range(Nl):
                rolled = np.roll(data[k], ml*(l-Nl//2), axis=-1)
                logRjlk[j, l, k] = np.sum(rolled * np.log(W[j] + 1e-20) - W[j])
    logPjlk = beta * logRjlk

    # optionally force Pjk to describe something continuous
    if force_continuity==True:
        kmax = np.argmax(np.sum(data, axis=(1,2)))
        com = np.argmax(np.sum(logPjlk, axis=1)[:, kmax])
        logPjlk[:, :, kmax] += log_gauss(com, np.arange(Nj), Nj/4)[:, None]
        for k in range(kmax+1, Nk):
            bias = np.argmax(np.sum(logPjlk, axis=1)[:, k-1])
            logPjlk[:, :, k] += log_gauss(bias, np.arange(Nj), Nj/4)[:, None]
        for k in range(kmax-1, -1, -1):
            bias = np.argmax(np.sum(logPjlk, axis=1)[:, k+1])
            logPjlk[:, :, k] += log_gauss(bias, np.arange(Nj), Nj/4)[:, None]
            
    # pragmatic pre-normalization to avoid overflow
    logPjlk -= np.max(logPjlk, axis=(0,1))
    Pjlk = np.exp(logPjlk)
    Pjlk /= np.sum(Pjlk, axis=(0,1))
    
    # then carry out the likelihood maximization (M) step
    W = build_model(data, Pjlk)

    return W, Pjlk

def C(W, envelope):
    ft = np.fft.fftn(W)
    error = np.sum(np.abs((1-envelope)*ft))
    W = np.abs(np.fft.ifftn(ft * envelope))
    return W, error

def generate_envelope(N, shape, support=0.5, type='box'):
    # the actual envelope - need to figure out the pixel sizes of the autocorrelation...
    envelope = np.ones((N, shape, shape), dtype=int)
    n1, n2 = (int(np.floor(s * (support/2))) for s in (N, shape))
    if type == 'box':
        envelope[n1:-n1] = 0
        envelope[:, n2:-n2] = 0
        envelope[:, :, n2:-n2] = 0
    elif type == 'sphere':
        inds = np.indices(envelope.shape)
        center = np.array(envelope.shape) // 2
        ri, rj, rk = inds - center.reshape((-1, 1, 1, 1))
        r = ri**2 / n1**2 + rj**2 / n2**2 + rk**2 / n2**2
        envelope[np.where(r > 1)] = 0
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
