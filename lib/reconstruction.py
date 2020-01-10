import numpy as np

def first_axis_com(a):
    """
    Calculates the center of mass of an array along each column.
    """
    inds = np.arange(a.shape[0], dtype=int).reshape((-1, 1))
    return np.sum(a*inds, axis=0) / np.sum(a, axis=0)

def build_model(data, Pjk):
    """
    Returns the maximum likelihood model W[j] based on the data[k] and
    probability matrix P[j,k]. This is Loh et al PRE 2009 eqn 11.
    """
    Nj, Nk = Pjk.shape
    Npix = data.shape[-1]
    W = np.zeros((Nj, Npix, Npix), dtype=np.complex128)
    for j in range(Nj):
        W[j][:] = 0.0
        for k in range(Nk):
            W[j][:] = W[j] + Pjk[j][k] * data[k]
        W[j][:] = W[j] / (np.sum(Pjk[j, :]) + 1e-20)
    return W

def M(W, priors, data, prior_decay=1.0, beta=1.0):
    """
    Performs the M update rule, Loh et al PRE 2009 eqns 8-11,
    with the addition of a home-made prior and the fudge factor
    from Ayyer et al J Appl Cryst 2016.
    """
    Nj = W.shape[0]
    Nk = data.shape[0]
    logRjk = np.empty((Nj, Nk), dtype=np.complex128)
    wjk = np.empty((Nj, Nk), dtype=float)

    # first, calculate the probabilities Pjk based on the current model
    W[:] = W / np.mean(np.abs(W)) * np.mean(data)
    for j in range(Nj):
        for k in range(Nk):
            logRjk[j, k] = np.sum(data[k] * np.log(W[j]) - W[j])
            # prior weight for each observation biased towards where it was placed last time:
            wjk[j, k] = 1 if prior_decay is None else np.exp(-np.abs(j-priors[k])/prior_decay)
    logPjk = np.log(wjk) + beta * logRjk
    # pragmatic pre-normalization to avoid overflow
    logPjk -= np.max(logPjk, axis=0)
    Pjk = np.exp(np.real(logPjk))
    Pjk /= np.sum(Pjk, axis=0)
    
    # then carry out the likelihood maximization (M) step
    W = build_model(data, Pjk)

    # then update the prior weights
    priors = first_axis_com(np.abs(Pjk))

    return W, priors, Pjk

def C(W, envelope):
    ft = np.fft.fftn(W)
    error = np.sum(np.abs((1-envelope)*ft))
    W[:] = np.fft.ifftn(ft * envelope)
    return W, error

def generate_envelope(N, shape, support=0.5):
    # the actual envelope - need to figure out the pixel sizes of the autocorrelation...
    envelope = np.ones((N, shape, shape), dtype=int)
    n1, n2 = (int(np.floor(s * (support/2))) for s in (N, shape))
    envelope[n1:-n1] = 0
    envelope[:, n2:-n2] = 0
    envelope[:, :, n2:-n2] = 0
    return envelope

def generate_initial(data, Nj, sigma=1.):
    """
    Generates an initial model W based on a simle ramp.
    """
    initial = np.linspace(0, Nj-1, len(data))
    Nk = len(data)
    jj = np.indices((Nj, Nk))[0]
    Pjk = np.exp(-(initial-jj)**2 / sigma**2 / 2)
    Pjk /= Pjk.sum(axis=0)
    return build_model(data, Pjk) + 1e-20
