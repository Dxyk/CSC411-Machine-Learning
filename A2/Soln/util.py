from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
import numpy as np
import pickle


# ========== HELPER FUNCTIONS ==========
def compute_accuracy(x, W, y):
    p = linear_forward(x, W)

    num_correct, num_incorrect = 0, 0
    for i in range(y.shape[0]):
        if np.argmax(y[i][:]) == np.argmax(p.T[i]):
            num_correct += 1
        else:
            num_incorrect += 1
    if num_correct + num_incorrect == 0:
        return 0
    return float(num_correct) / float((num_incorrect + num_correct)) * 100.0



# ========== MATH ==========
def softmax(y):
    """
    Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases
    """
    return np.exp(y) / np.tile(np.sum(np.exp(y), 0), (len(y), 1))


def tanh_layer(W, b, y):
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


def linear_forward(x, W):
    """
    Compute the given network's output.
    The first output layer has linear activation.
    The final output has softmax activation.
    """
    lin_output = np.dot(W.T, x)
    return softmax(lin_output)


def loss(x, W, y):
    """
    The cost function for linear forward
    """
    p = linear_forward(x, W)
    return -np.sum(y * np.log(p)) / x.shape[1]


def dlossdw(x, W, y):
    """
    The gradient for linear forward cost w.r.t weight w
    """
    p = linear_forward(x, W)
    return np.dot((p - y), x.T).T


def grad_descent(loss, dlossdw, x_train, y_train, x_test, y_test, init_W,
                 alpha = 0.00001, gamma = 0, max_iter = 10000, plot = False, plot_path = ""):
    """
    Gradient descent for linear forward
    """
    iterations = []
    train_results = []
    test_results = []

    print "----------- Starting Gradient Descent -----------"
    eps = 1e-5
    prev_W = init_W - 10 * eps
    W = init_W.copy()
    i = 0

    while i < max_iter and norm(W - prev_W) > eps:
        prev_W = W.copy()
        W -= alpha * dlossdw(x_train, W, y_train)
        W -= gamma*prev_W + grad

        if i % (max_iter // 10) == 0 or i == max_iter - 1:
            print "Iteration: {}\n\tCost:{}\n".format(i, loss(x_train, W, y_train))
            train_results.append(loss(x_train, W, y_train))
            test_results.append(loss(x_test, W, y_test))
            iterations.append(i)

        i += 1
    print "----------- Done Gradient Descent -----------"

    if plot:
        plt.plot(iterations, train_results, 'r', iterations, test_results, 'b')
        plt.xlabel("iterations")
        plt.ylabel("loss")
        if plot_path == "":
            print "Please provide a plot path"
            print "Showing instead"
            plt.show()
        plt.savefig(plot_path)

    return W
