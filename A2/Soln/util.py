from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
import numpy as np


# ========== MATH ==========
def softmax(y):
    """
    Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases
    """
    return np.exp(y) / np.tile(np.sum(np.exp(y), 0), (len(y), 1))


def tanh_layer(y, W, b):
    """
    Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases
    """
    return np.tanh(np.dot(W.T, y) + b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = np.dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


def NLL(y, y_):
    return -np.sum(y_ * np.log(y))


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    """Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network"""
    dCdL1 = y - y_
    dCdW1 = np.dot(L0, dCdL1.T)


def linear_forward(x, W, b):
    """
    Compute the given network's output.
    The first output layer has linear activation.
    The final output has softmax activation.
    """
    lin_output = np.dot(W.T, x) + b
    return softmax(lin_output)


def loss(x, W, b, y):
    """
    The cost function for linear forward
    """
    p = linear_forward(x, W, b)
    return -np.sum(y * np.log(p)) / x.shape[1]


def dlossdw(x, W, b, y):
    """
    The gradient for linear forward cost w.r.t weight w
    """
    p = linear_forward(x, W, b)
    return np.matmul((p - y), x.T).T


def dlossdb(x, W, b, y):
    """
    The gradient for linear forward cost w.r.t weight b
    """
    p = linear_forward(x, W, b)
    return np.matmul((p - y), x.T).T


def grad_descent(loss, dlossdx, x, y, init_theta, alpha):
    """
    Gradient descent for linear forward
    """
    print "----------- Starting Gradient Descent -----------"
    eps = 1e-5
    prev_theta = init_theta - 10 * eps
    theta = init_theta.copy()
    max_iter = 100000
    i = 0

    while norm(theta - prev_theta) > eps and i < max_iter:
        prev_theta = theta.copy()
        theta -= alpha * dlossdx(x, y, theta)
        if i % 5000 == 0 or i == max_iter - 1:
            print "Iteration: {}\nCost:{}\n".format(i, loss(x, y, theta))
        i += 1
    return theta
