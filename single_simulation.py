"""
Simulates a particle rocking through an N-shaped and noisy curve.
"""

from lib.simulation import simulate_octahedron
import numpy as np

# non-monotonous N-shaped rocking curve
theta = np.concatenate((
   np.linspace(-1,.2,40),
   np.linspace(.2,0,16),
   np.linspace(0,1.,24),
   )) + np.random.rand(80)*.05

# optionally add variation on the phi angle too!
phi = np.zeros_like(theta)

frames = simulate_octahedron(offsets=theta, rolls=phi)
np.savez_compressed('data/test_data.npz', offsets=theta, rolls=phi, frames=frames)
