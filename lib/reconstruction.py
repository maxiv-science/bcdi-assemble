import numpy as np

def first_axis_com(a):
    """
    Calculates the center of mass of an array along each column.
    """
    inds = np.arange(a.shape[0], dtype=int).reshape((-1, 1))
    return np.sum(a*inds, axis=0) / np.sum(a, axis=0)

def M(W, priors, data, prior_decay=1.0):
    Nj = W.shape[0]
    Nk = data.shape[0]
    Rjk = np.empty((Nj, Nk), dtype=np.complex128)
    wjk = np.empty((Nj, Nk), dtype=float)

    # first, calculate the probabilities Pjk based on the current model
    W[:] = W / np.mean(np.abs(W)) * np.mean(data)
    for j in range(Nj):
        for k in range(Nk):
            Rjk[j, k] = np.exp(np.sum(data[k] * np.log10(W[j]+1e-6) - W[j]))
            # prior weight for each observation biased towards where it was placed last time:
            wjk[j, k] = np.exp(-np.abs(j-priors[k])/prior_decay)
    Pjk = wjk*Rjk / np.sum(wjk*Rjk, axis=0)
    
    # then carry out the likelihood maximization (M) step
    for j in range(Nj):
        W[j][:] = 0.0
        for k in range(Nk):
            W[j][:] = W[j] + Pjk[j][k] * data[k]
        W[j][:] = W[j] / np.sum(Pjk[j, :])

    # then update the prior weights
    priors = first_axis_com(np.abs(Pjk))

    return W, priors, Pjk

def C(W, envelope):
    ft = np.fft.fftn(W)
    error = np.sum(np.abs((1-envelope)*ft))
    W[:] = np.fft.ifftn(ft * envelope)
    return W, error

def generate_initial(N, shape, support=0.5):
    # the actual envelope - need to figure out the pixel sizes of the autocorrelation...
    envelope = np.ones((N, shape, shape), dtype=int)
    n1, n2 = (int(np.floor(s * (support/2))) for s in (N, shape))
    envelope[n1:-n1] = 0
    envelope[:, n2:-n2] = 0
    envelope[:, :, n2:-n2] = 0

    # the initial model
    ft = np.ones((N, shape, shape)) * envelope
    W = np.fft.fftshift(np.fft.ifftn(ft))

    return W, envelope
