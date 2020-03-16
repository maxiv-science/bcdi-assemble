"""
Simulates a particle rocking through an N-shaped and noisy curve.
"""

from diffassemble.simulation import simulate_octahedron
import numpy as np

# non-monotonous N-shaped rocking curve
theta = np.concatenate((
   np.linspace(-1,.2,40),
   np.linspace(.2,0,16),
   np.linspace(0,1.,24),
   )) + np.random.rand(80)*.05

# optionally add variation on the phi angle too!
phi = 10 * np.sin(np.arange(len(theta)) * np.pi / 10)

frames = simulate_octahedron(offsets=theta, rolls=phi, roll_center=[100,64])
np.savez_compressed('../data/test_data.npz', offsets=theta, rolls=phi, frames=frames)
