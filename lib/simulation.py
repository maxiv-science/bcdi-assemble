import ptypy
import nmutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches
plt.ion()

try:
    from nmutils.utils.bodies import TruncatedOctahedron
except ImportError:
    raise Exception('This code uses the NanoMAX nmutils package, see github.')

try:
    ptypy.core.geometry_bragg.Geo_BraggProjection
except AttributeError:
    raise Exception('Use 3dBPP ptypy version!')

def simulate_octahedron(offsets, rolls):
    """
    Simulate a particle rotating through its rocking curve in a complicated
    way, as well as diffusion along the powder ring.

    Defines a truncated-octahedral particle, rotates it, and applies the
    projection operator before doing the 2D FT.
    """

    # physics
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

    ### prepare diffraction plots
    diff_max = 6e6
    exit_max = 13.
    n = int(np.ceil(np.sqrt(len(offsets))))
    fig, ax = plt.subplots(ncols=n, nrows=n, figsize=(13, 7.55))
    ax = ax.flatten()
    fig.subplots_adjust(hspace=0, wspace=0, right=.85, left=.06, bottom=.1, top=.99)
    cbar_ax = fig.add_axes((.91, .2, .03, .6))

    fig.text(.02, .55, 'rocking angle (degrees)', rotation=90, fontsize=16, ha='left', va='center')
    fig.text(.5, .02, 'phi angle (degrees)', ha='center', va='bottom', fontsize=16)
    plt.pause(0.1)

    ### the main calculation loop
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


    ### uneven data
    data = []
    for ioffset, offset in enumerate(offsets):
        g.bragg_offset = offset

        I = np.abs(g.propagator.fw(v.data))**2
        I = np.roll(I, rolls[ioffset], axis=1)
        exit = g.overlap2exit(v.data)
        data.append({'offset': offset,
                     'angle': angle,
                     'diff': I,
                     'exit': exit})

        # add shot noise and plot diffraction
        if ioffset == 0:
            photons_per_intensity = 2e4 / np.sum(I)
        photons = photons_per_intensity * np.sum(I)
        diff = nmutils.utils.noisyImage(I, photonsTotal=photons)
        ax[ioffset].imshow(diff, interpolation='none', vmax=diff_max * photons_per_intensity, cmap='jet', norm=matplotlib.colors.LogNorm())
        plt.setp(ax[ioffset], 'xticks', [], 'yticks', [])

        plt.draw()
        plt.pause(.01)

    frames = [d['diff'] for d in data]
    return frames
