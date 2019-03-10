import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# print(x_batch.shape)
# print(t_batch.shape)
# print(x_batch[0])
# print(t_batch[0])

def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x
    
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)
    

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x, y)
# plt.show()

d5 = numerical_diff(function_1, 5)
d10 = numerical_diff(function_1, 10)

print(d5)
print(d10)