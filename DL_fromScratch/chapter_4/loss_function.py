import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

r = mean_squared_error(np.array(y), np.array(t))
print(r)

rc = cross_entropy_error(np.array(y), np.array(t))
print(rc)


y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
r2 = mean_squared_error(np.array(t), np.array(y))
print(r2)

rc2 = cross_entropy_error(np.array(t), np.array(y))
print(rc2)