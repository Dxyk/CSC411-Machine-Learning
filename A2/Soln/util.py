from pylab import *
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from scipy.io import loadmat
import numpy as np
import pickle

# ========== CONSTANTS ==========
BUFF_SIZE = 65536
ACTORS_FILE = "Resource/subset_actors.txt"
UNCROPPED = "Resource/uncropped/"
CROPPED32 = "Resource/cropped_32/"
CROPPED64 = "Resource/cropped_64/"


# ----------- GLOBAL VARIABLES -----------
# list of actors
actors = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin",
          "Bill Hader", "Steve Carell"]
# list of actors with their lower case last name
actor_names = [actor.split()[1].lower() for actor in sorted(actors)]
# a dict of actor : count
actor_count = {actor_name: 0 for actor_name in actor_names}


# ========== HELPER FUNCTIONS ==========
def count_actors(path = "./Resource/uncropped"):
    """
    Count all actor's images
    Args:
        path (str): the path to the data set
    Returns:
        A dict of actor if actor is not given
    """
    for root, dirs, images in os.walk(path):
        for image in images:
            for actor_name in actor_names:
                if image.find(actor_name) != -1:
                    actor_count[actor_name] += 1
    return actor_count


def get_accuracy(x, W, y):
    p = linear_forward(x, W)

    num_correct, num_incorrect = 0, 0
    for i in range(y.shape[1]):
        # print y[:, i]
        if np.argmax(y[:, i]) == np.argmax(p[:, i]):
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
    # try:
    #     np.sum(y * np.log(p)) / x.shape[1]
    # except FloatingPointError, e:
    #     import time
    #     print "============================="
    #     print "X: \n", x
    #     print "W: \n", W
    #     print "Y: \n", y
    #     print "============================="
    #     time.sleep(1)
    return -np.sum(y * np.log(p)) / x.shape[1]


def dlossdw(x, W, y):
    """
    The gradient for linear forward cost w.r.t weight w
    """
    p = linear_forward(x, W)
    return np.dot((p - y), x.T).T


def grad_descent(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test,
                 init_W, alpha = 0.00001, gamma = 0, max_iter = 10000,
                 plot_path = ""):
    """
    Gradient descent for linear forward
    """
    iterations = []
    train_results = []
    val_results = []
    test_results = []

    print "----------- Starting Gradient Descent -----------"
    eps = 1e-4
    prev_W = init_W - 10 * eps
    W = init_W.copy()
    i = 0
    V = np.zeros(W.shape)

    while i < max_iter and norm(W - prev_W) > eps:
        prev_W = W.copy()
        # W -= alpha * dlossdw(x_train, W, y_train)
        V = gamma * V + alpha * dlossdw(x_train, W, y_train)
        W -= V

        if i % 10 == 0 or i == max_iter - 1:
            train_result = get_accuracy(x_train, W, y_train)
            val_result = get_accuracy(x_val, W, y_val)
            test_result = get_accuracy(x_test, W, y_test)
            train_results.append(train_result)
            val_results.append(val_result)
            test_results.append(test_result)
            iterations.append(i)
            if i % (max_iter // 10) == 0 or i == max_iter - 1:
                print "Iteration: {}".format(i)
                print "\tTrain Loss:{}".format(loss(x_train, W, y_train))
                print "\tValidation Loss:{}".format(loss(x_val, W, y_val))
                print "\tTest Loss:{}".format(loss(x_test, W, y_test))
                print "\tTrain rate:{}".format(get_accuracy(x_train, W, y_train))
                print "\tValidation rate:{}".format(get_accuracy(x_val, W, y_val))
                print "\tTest rate:{}".format(get_accuracy(x_test, W, y_test))

        i += 1
    print "----------- Done Gradient Descent -----------"

    if plot_path != "":
        plt.plot(iterations, train_results, 'b', iterations, val_results, 'g',
                 iterations, test_results, 'r')
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend(("Training", "Validation", "Test"), loc = "best")
        plt.title("Learning curve")
        plt.savefig(plot_path)

    return W


def grad_descent_6(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test,
                 init_W, w1, w2, alpha = 0.00001, gamma = 0, max_iter = 10000):
    print "----------- Starting Gradient Descent -----------"
    eps = 1e-4
    prev_W = init_W - 10 * eps
    W = init_W.copy()
    i = 0
    V = np.zeros(W.shape)
    rec = [(W[w1, 5], W[w2, 5])]

    while i < max_iter and norm(W - prev_W) > eps:
        prev_W = W.copy()
        # W -= alpha * dlossdw(x_train, W, y_train)
        V = gamma * V + alpha * dlossdw(x_train, W, y_train)
        W[w1, 5] -= V[w1, 5]
        W[w2, 5] -= V[w2, 5]
        rec.append((W[w1, 5], W[w2, 5]))

        if i % (max_iter // 10) == 0 or i == max_iter - 1:
            print "Iteration: {}".format(i)
            print "\tTrain Loss:{}".format(loss(x_train, W, y_train))
            print "\tValidation Loss:{}".format(loss(x_val, W, y_val))
            print "\tTest Loss:{}".format(loss(x_test, W, y_test))
            print "\tTrain rate:{}".format(get_accuracy(x_train, W, y_train))
            print "\tValidation rate:{}".format(get_accuracy(x_val, W, y_val))
            print "\tTest rate:{}".format(get_accuracy(x_test, W, y_test))

        i += 1
    print "----------- Done Gradient Descent -----------"

    return W, rec

count_actors()
