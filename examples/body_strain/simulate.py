"""
Simulates a particle rocking through an N-shaped and noisy curve.
"""

from diffassemble.simulation import simulate_octahedron
import numpy as np

# the last and trickiest case from ten_simulations.py
Nk = 80
noise = np.random.rand(Nk) - .5
noise = np.fft.ifft(np.fft.fft(noise) * (np.abs(np.arange(Nk))<Nk//4))
noise = np.real(noise)
noise = noise / np.abs(noise).max() * .05
theta = np.linspace(-1, 1, Nk)
period, ampl = Nk//5, .2
envelope = (1 - np.abs(np.arange(Nk)/(Nk//2) - 1))
theta += envelope * np.sin(np.arange(Nk) * 2 * 3.14 / period) * ampl
theta += noise
ampl = 15
rolls = np.interp(np.linspace(0, 1, Nk), (0,.2, .4, .6, .8, 1.), (0, 1, -1, .8, -.6, .4))
phi = (rolls * ampl).astype(int)

for strain in (.1, .25, .5, 1.0,):
	frames, particle = simulate_octahedron(offsets=theta, rolls=phi,
					           			   strain_type='body', strain_size=strain,
								           plot=True)
	np.savez_compressed('simulated_%.2f.npz'%strain, offsets=theta, rolls=phi, frames=frames, particle=particle)
