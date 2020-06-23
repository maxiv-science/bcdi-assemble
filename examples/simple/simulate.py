"""
Simulates a particle rocking through an arbitrary and noisy rocking
curve, as well as along the powder ring.
"""

from bcdiass.simulation import simulate_octahedron
import matplotlib.pyplot as plt
import numpy as np

Nk = 80

# theta
theta = np.linspace(-1, 1, Nk)
period, ampl = Nk//5, .2
envelope = (1 - np.abs(np.arange(Nk)/(Nk//2) - 1))
theta += envelope * np.sin(np.arange(Nk) * 2 * 3.14 / period) * ampl
noise = np.random.rand(Nk) - .5
noise = np.fft.ifft(np.fft.fft(noise) * (np.abs(np.arange(Nk))<Nk//4))
noise = np.real(noise)
noise = noise / np.abs(noise).max() * .05
theta += noise

# phi
nodes = 5
ampl = 25
phi = np.interp(np.arange(len(theta)), np.arange(nodes+1)*len(theta)/nodes, np.random.rand(nodes+1))
phi = np.round(phi*ampl).astype(int)
phi -= phi[Nk//2]

fig, ax = plt.subplots(nrows=2)
ax[0].plot(theta)
ax[1].plot(phi)

frames = simulate_octahedron(offsets=theta, rolls=phi, roll_center=None)
np.savez_compressed('data.npz', offsets=theta, rolls=phi, frames=frames)
