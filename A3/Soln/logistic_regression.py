from util import *

import numpy as np
from numpy.linalg import *
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ========== Logistic Regression ==========
def generate_np_data(word_map, data_set, label, rm_skip_word = False):
    """
    Generate the np data sets and labels given the list data sets
    """
    np_data_set = np.zeros((len(word_map), len(data_set)))
    np_label = np.zeros((len(label),))
    for idx, headline in enumerate(data_set):
        for word in headline.strip().split():
            if not rm_skip_word or word not in ENGLISH_STOP_WORDS:
                np_data_set[word_map.index(word), idx] += 1
        np_label[idx] = label[idx]

    np_data_set = np.vstack((np_data_set, np.ones((np_data_set.shape[1]))))
    return np_data_set, np_label


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def forward(x, theta):
    """
    Forward function
    """
    return sigmoid(np.dot(theta.T, x))


def loss_fn(x, y, theta, reg_lambda):
    """
    Logistic Cross Entropy loss function with L2 Regularization
    """
    output = forward(x, theta)
    return -sum(y * np.log(output) + (1 - y) * np.log((1 - output))) + \
           reg_lambda * np.dot(theta.T, theta)


def dlossdw(x, y, theta, reg_lambda):
    """
    Derivative of the loss function
    """
    output = forward(x, theta)
    return np.dot(x, (output - y).T) + 2 * reg_lambda * theta


def performance(x, y, theta):
    """
    Calculate the performance of the current set of weights
    """
    pred = np.dot(theta.T, x)
    total_count = y.shape[0]
    correct_count = 0
    for i in range(y.shape[0]):
        if (y[i] == 1 and pred[i] > 0) or (y[i] == 0 and pred[i] < 0):
            correct_count += 1
    return float(correct_count) / float(total_count)


def grad_descent(loss_fn, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test,
                 init_w, alpha, reg_lambda, max_iter = 10000, check_point_len = 50,
                 print_res = True):
    """
    Logistic gradient descent
    """
    print "Training with alpha = {}, reg_lambda = {}".format(alpha, reg_lambda)
    eps = 1e-5
    prev_theta = init_w - 10 * eps
    theta = init_w.copy()
    iter = 0
    train_res = []
    val_res = []
    test_res = []
    iters = []
    while norm(theta - prev_theta) > eps and iter < max_iter:
        # grad descent
        prev_theta = theta.copy()
        theta -= alpha * dlossdw(x_train, y_train, theta, reg_lambda)
        # record performance
        if iter % check_point_len == 0:
            if print_res and iter % (10 * check_point_len) == 0:
                print "Iter: {}".format(iter)
                print "Cost: {}".format(loss_fn(x_train, y_train, theta, reg_lambda))
            iters.append(iter)
            train_res.append(performance(x_train, y_train, theta))
            val_res.append(performance(x_val, y_val, theta))
            test_res.append(performance(x_test, y_test, theta))
        iter += 1

    print "Training Completed"

    return theta, train_res, val_res, test_res, iters


def tune_lr_params(x_train, y_train, x_val, y_val, x_test, y_test, init_w, max_iter,
                   check_point_len):
    max_val_res = 0
    opt_alpha = 0
    opt_reg_lambda = 0
    print "========== Tuning =========="
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
        for reg_lambda in [0.001, 0.01, 0.1]:
            print "Tuning on alpha = {}, reg_lambda = {}".format(alpha, reg_lambda)
            res = grad_descent(loss_fn, dlossdw, x_train, y_train, x_val, y_val,
                               x_test, y_test, init_w, alpha, reg_lambda, max_iter,
                               check_point_len, print_res = False)
            w, train_res, val_res, test_res, iters = res
            print "Result on Train: {}".format(train_res[-1])
            print "Result on Validation: {}".format(val_res[-1])
            print "Result on Test: {}".format(test_res[-1])
            if val_res[-1] > max_val_res:
                max_val_res = val_res[-1]
                opt_alpha = alpha
                opt_reg_lambda = reg_lambda
    print "========== Done Tuning =========="
    print "optimum alpha: {}, optimum reg_lambda: {}".format(opt_alpha,
                                                             opt_reg_lambda)
    pickle.dump((opt_alpha, opt_reg_lambda), open("./data/lr_params.p", mode = "wb"))
    return opt_alpha, opt_reg_lambda


if __name__ == "__main__":
    sets = separate_sets(seed = 0, overwrite = False)
    train_set = sets[TRAIN_SET]
    train_label = sets[TRAIN_LABEL]
    val_set = sets[VAL_SET]
    val_label = sets[VAL_LABEL]
    test_set = sets[TEST_SET]
    test_label = sets[TEST_LABEL]
    word_dict = pickle.load(open("./data/total_dict.p", mode = "rb"))
    word_map = sorted(word_dict.keys())
    # Process data to np sets
    x_train, y_train = generate_np_data(word_map, train_set, train_label)
    x_val, y_val = generate_np_data(word_map, val_set, val_label)
    x_test, y_test = generate_np_data(word_map, test_set, test_label)
    print x_train.shape, y_train.shape
    print x_val.shape, y_val.shape
    print x_test.shape, y_test.shape


