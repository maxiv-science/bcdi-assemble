import multiprocessing
import numpy as np
from functools import partial

# normal
a1 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        a1[i, j] = i*10 + j
print(a1)

# parallel with Pool
def inner(i):
    a = np.zeros(10)
    for j in range(10):
        a[j] = i*10 + j
    return a
pool = multiprocessing.Pool(4)
a2 = np.array(pool.map(inner, range(10)))
print(a2)

# parallel with Pool and extra data through partial
def inner(i, data):
    a = np.zeros(10)
    for j in range(10):
        a[j] = i*10 + j
    print('data: ', data)
    return a
pool = multiprocessing.Pool(4)
data = 'hello'
a3 = np.array(pool.map(partial(inner, data=data), range(10)))
print(a3)

assert np.alltrue(a1 == a2)
assert np.alltrue(a1 == a3)
