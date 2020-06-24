This is a utility library which enables the assembly of Bragg Coherent Diffraction Imaging (BCDI) frames of unknown orientations, such as collected via spontaneous rocking curves. The library is based on a Python module `bcdiass` which provides the basic operators of the algorithm as well as utilities for both simulation of experimental data and assembly of the same.

## Dependencies
The core assembly library depends on standard python components such as numpy, scipy, matplotlib, scikit-image. In addition, simulating datasets from geometrical model particles currently depends on [the 3dBPP branch of ptypy](https://github.com/ptycho/ptypy/tree/3dBPP) (registration needed) as well as the [NanoMAX beamline utitility library](https://github.com/maxiv-science/nanomax-analysis-utils).

## Installation
The library installed with setuptools, for example like
```
cd bcdi-assembly
python setup.py --user
```

## Usage
Assembly procedures are easily scripted with the utilities provided. Examples are provided in the `examples` folder, and a minimal example might look as follows.
```python
import numpy as np
from bcdiass.utils import *

# input and parameters
data = np.load('data.npz')['frames']
Nj, Nl = 25, 25

# initial model
envelope = generate_envelope(Nj, data.shape[-1], support=.2)
W = generate_initial(data, Nj)

# assembly loop
for i in range(100):
    W, Pjlk, timing = M(W, data, Nl=Nl, beta=1e-4)
    W, error = C(W, envelope)
```
