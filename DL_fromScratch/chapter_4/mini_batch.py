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

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

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


g1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
g2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
g3 = numerical_gradient(function_2, np.array([3.0, 0.0]))

print(g1)
print(g2)
print(g3)


init_x = np.array([-3.0, 4.0])
gd = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(gd)
