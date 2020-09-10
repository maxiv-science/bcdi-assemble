import ptypy
import nmutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from .utils import roll
plt.ion()

try:
    from nmutils.utils.bodies import TruncatedOctahedron
except ImportError:
    raise Exception('This code uses the NanoMAX nmutils package, see\n'
        +'https://github.com/maxiv-science/nanomax-analysis-utils\n')

try:
    ptypy.core.geometry_bragg.Geo_BraggProjection
except AttributeError:
    raise Exception('Use 3dBPP ptypy branch!')

def simulate_octahedron(offsets, rolls, photons_in_central_frame=1e6, plot=True,
                        strain_type=None, strain_size=.25, roll_center=None):
    """
    Simulate a particle rotating through its rocking curve in a complicated
    way, as well as diffusion along the powder ring.

    Defines a truncated-octahedral particle, rotates it, and applies the
    projection operator before doing the 2D FT.

    strain_type: optionally add a phase pattern to the particle, can be
                 'body' or 'surface'.
    strain_size: size of the phase structures (radians)
    """

    ### define the physics
    a = 4.065e-10
    E = 10000.
    d = a / np.sqrt(3) # (111)
    hc = 4.136e-15 * 3.000e8
    theta = np.arcsin(hc / (2 * d * E)) / np.pi * 180
    psize = 55e-6
    distance = .25
    diameter = 90e-9
    truncation = .69
    angle = 30.0

    ### Define a Bragg geometry
    g = ptypy.core.geometry_bragg.Geo_BraggProjection(psize=(psize, psize),
        shape=(31, 128, 128), energy=E*1e-3, distance=distance, theta_bragg=theta,
        bragg_offset=0.0, r3_spacing=5e-9)

    ### make the object container and storage
    C = ptypy.core.Container(data_type=np.complex128, data_dims=3)
    pos = [0, 0, 0]
    pos_ = g._r3r1r2(pos)
    v = ptypy.core.View(C, storageID='Sobj', psize=g.resolution, coord=pos_, shape=g.shape)
    S = C.storages['Sobj']
    C.reformat()

    ### set up the particle
    o = TruncatedOctahedron(truncation)
    o.shift([-.5, -.5, -.5])
    o.rotate('z', 45)
    o.rotate('y', 109.5/2)
    o.scale(diameter)
    # now we have an octahedron lying down on the its xy plane.
    # the conversion between the right handed octahedron's coordinates
    # and Berenguer's is just yB = -y
    o.rotate('z', angle)
    xx, zz, yy = g.transformed_grid(S, input_space='real', input_system='natural')
    v.data[:] = o.contains((xx, -yy, zz))

    ### optionally add strain
    if strain_type == 'body':
        #v.data[np.where(xx[0] * yy[0] * zz[0] > 0)] *= np.exp(1j * strain_size)
        from scipy.special import sph_harm
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        azimuth = np.arctan(yy / (xx + 1e-30))
        polar = np.arccos(zz / (r + 1e-30))
        l, m = 3, 2
        Y = sph_harm(m, l, azimuth, polar).real
        R = np.sin(r / (diameter * truncation / 2) * np.pi)
        u = Y * R
        u = u / u.ptp() * strain_size
        v.data *= np.exp(1j * u[0])
    elif strain_type == 'surface':
        o.scale(.8)
        v.data[np.where(1 - o.contains((xx[0], -yy[0], zz[0])))] *= np.exp(1j * strain_size)
    if strain_type:
        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
        ax[0].imshow(np.abs(v.data[16]))
        ax[1].imshow(np.angle(v.data[16]))

    ### calculate
    data = []
    for ioffset, offset in enumerate(offsets):
        g.bragg_offset = offset

        I = np.abs(g.propagator.fw(v.data))**2
        I = roll(I, rolls[ioffset], roll_center)
        exit = g.overlap2exit(v.data)
        data.append({'offset': offset,
                     'angle': angle,
                     'diff': I,
                     'exit': exit})

    ### add shot noise
    frames = [d['diff'] for d in data]
    if photons_in_central_frame is not None:
        central = np.argmin(np.abs(offsets))
        photons_per_intensity = photons_in_central_frame / frames[central].sum()
        global_max = frames[central].max() * photons_per_intensity
        noisy_frames = []
        for i, frame in enumerate(frames):
            mask = frame < 0
            frame[mask] = 0
            noisy = nmutils.utils.noisyImage(frame, photonsTotal=photons_per_intensity*frame.sum())
            noisy[mask] = -1
            noisy_frames.append(noisy)
        frames = noisy_frames
    else:
        central = np.argmin(np.abs(offsets))
        global_max = frames[central].max()

    ### plot if requested
    if plot:
        n = int(np.ceil(np.sqrt(len(offsets))))
        fig, ax = plt.subplots(ncols=n, nrows=n, figsize=(13, 7.55))
        ax = ax.flatten()
        fig.subplots_adjust(hspace=0, wspace=0, right=.85, left=.06, bottom=.1, top=.99)
        for i, frame in enumerate(frames):
            ax[i].imshow(frame, vmax=global_max, cmap='jet', norm=matplotlib.colors.LogNorm())
        plt.pause(.1)

    return frames, v.data
