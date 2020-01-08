"""
Simulates a particle rocking through 10 different rocking curves,
with and without rolling.
"""

from lib.simulation import simulate_octahedron
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

trajectories = []
Nk = 80

# 1. constant slope
theta = np.linspace(-1., 1., Nk)
trajectories.append({'theta': theta, 'phi':np.zeros_like(theta)})

# 2. monotonous but with different slopes
assert (Nk == 80)
theta = np.concatenate((
   np.linspace(-1,-.22, 30),
   np.linspace(-.2,-.02,20),
   np.linspace(0,.4, 10),
   np.linspace(.42,1.,20),
   ))
trajectories.append({'theta': theta, 'phi':np.zeros_like(theta)})

# 3. non-monotonous N-shaped rocking curve
assert (Nk == 80)
theta = np.concatenate((
   np.linspace(-1,.2,40),
   np.linspace(.2,-.1,16),
   np.linspace(-.1,1.,24),
   ))
trajectories.append({'theta': theta, 'phi':np.zeros_like(theta)})

# 4. non-monotonous N-shaped rocking curve with slow noise
noise = np.random.rand(Nk) - .5
noise = np.fft.ifft(np.fft.fft(noise) * (np.abs(np.arange(Nk))<Nk//4))
noise = np.real(noise)
noise = noise / np.abs(noise).max() * .05
trajectories.append({'theta': theta+noise, 'phi':np.zeros_like(theta)})

# 5. oscillation on a ramp
theta = np.linspace(-1, 1, Nk)
period, ampl = Nk//5, .2
envelope = (1 - np.abs(np.arange(Nk)/(Nk//2) - 1))
theta += envelope * np.sin(np.arange(Nk) * 2 * 3.14 / period) * ampl
theta += noise
trajectories.append({'theta': theta, 'phi':np.zeros_like(theta)})

# the same five rocking curves with additional roll
nodes = 5
ampl = 30
rolls = np.interp(np.arange(len(theta)), np.arange(nodes+1)*len(theta)/nodes, np.random.rand(nodes+1))
rolls = np.round(rolls*ampl).astype(int)
rolls -= ampl//2
for i in range(len(trajectories)):
    traj = {'theta':trajectories[i]['theta'], 'phi': rolls}
    trajectories.append(traj)

# plot the trajectories
fig, ax = plt.subplots(nrows=len(trajectories)//2, ncols=2, sharex=True, sharey=True)
ax = ax.T.flatten()
for i, traj in enumerate(trajectories):
    ax[i].plot(traj['theta'], 'r')
    ax[i].twinx().plot(traj['phi'], 'b')
    ax[i].set_title(i)

for i, traj in enumerate(trajectories):
    frames = simulate_octahedron(offsets=traj['theta'], rolls=traj['phi'],
        photons_in_central_frame=1e6, plot=False)
    np.savez_compressed('data/ten_simulations_%u.npz'%i, offsets=traj['theta'], rolls=traj['phi'], frames=frames)
