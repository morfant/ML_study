import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network



x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)

print(W1.shape)
print(W2.shape)
print(W3.shape)
