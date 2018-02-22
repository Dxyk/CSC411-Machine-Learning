from util import *
from scipy.io import loadmat
from pylab import *
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


# ==================== CONSTANTS ====================
TRAIN = "train"
TEST = "test"


# ==================== HELPER FUNCTIONS ====================
def load_data():
    print "===== Loading Data ====="
    data = loadmat("mnist_all.mat")
    print "===== Data Loaded ====="
    return data


def generate_sets(data, train_size = -1, test_size = -1):
    """
    generate the training set from data
    :param data: the given data
    :type data: dict
    :param train_size: The training size of every digit
    :type train_size: int
    :param test_size: the size of set of every digit
    :type test_size: int
    :return: the training set and the corresponding answer
    :rtype: tuple (list, list)
    """
    print "===== Generating Sets ====="
    if train_size == -1:
        # We take min to make sure every digit gets trained the same amount
        train_val_arr_size = min([data[TRAIN + str(i)].shape[0] for i in range(10)])
    else:
        train_val_arr_size = \
            min(min([data[TRAIN + str(i)].shape[0] for i in range(10)]), test_size)
    train_arr_size = int(train_val_arr_size * .8)
    val_arr_size = int(train_val_arr_size - train_arr_size)

    if test_size == -1:
        test_arr_size = min([data[TEST + str(i)].shape[0] for i in range(10)])
    else:
        test_arr_size = min(min([data[TEST + str(i)].shape[0] for i in range(10)]),
                            test_size)

    train_set = np.zeros((28 * 28, 10 * train_arr_size))
    train_soln_set = np.zeros((10, 10 * train_arr_size))
    val_set = np.zeros((28 * 28, 10 * val_arr_size))
    val_soln_set = np.zeros((10, 10 * val_arr_size))
    test_set = np.zeros((28 * 28, 10 * test_arr_size))
    test_soln_set = np.zeros((10, 10 * test_arr_size))

    for i in range(10):
        curr_key = TRAIN + str(i)
        data_set = data[curr_key] / 255.0
        # separate dataset to training and validation sets
        train_data_set = data_set[: train_arr_size]
        val_data_set = data_set[train_arr_size: train_arr_size + val_arr_size]
        # generate training
        train_set[:, i * train_arr_size: (i + 1) * train_arr_size] = train_data_set.T
        train_soln_set[i, i * train_arr_size: (i + 1) * train_arr_size] = 1
        # generate validation
        val_set[:, i * val_arr_size: (i + 1) * val_arr_size] = val_data_set.T
        val_soln_set[i, i * val_arr_size: (i + 1) * val_arr_size] = 1

    for i in range(10):
        curr_key = TEST + str(i)
        data_set = data[curr_key] / 255.0
        # generate test
        test_data_set = data_set[: test_arr_size]
        test_set[:, i * test_arr_size: (i + 1) * test_arr_size] = test_data_set.T
        test_soln_set[i, i * test_arr_size: (i + 1) * test_arr_size] = 1

    print "===== Sets Generated ====="

    return (train_set, train_soln_set), (val_set, val_soln_set), \
           (test_set, test_soln_set)


# ==================== MAIN FUNCTIONS ====================
# Part 1
def part1():
    """
    Load the data and store 10 images for each digit
    """
    # Load the MNIST digit data
    data = load_data()

    # # get first 10 image for each digit
    # for number in range(10):
    #     for i in range(10):
    #         im = data["train" + str(number)][i].reshape((28, 28)) / 255.0
    #         file_name = "report/images/1/{}_{}.png".format(str(number), str(i))
    #         imsave(file_name, im, cmap = cm.gray)

    # plot a random sample
    fig, ax = plt.subplots(10, 10)

    for i in range(10):
        num_key = 'train' + str(i)
        data_length = len(data[num_key])
        # take 10 random samples
        rand_idxs = random.sample(range(0, data_length), 10)
        for j in range(10):
            rand_idx = rand_idxs[j]
            img = data[num_key][rand_idx].reshape((28, 28))
            plt.sca(ax[i, j])
            plt.imshow(img, cmap = cm.gray)
            plt.axis('off')

    # plt.show()
    plt.savefig("report/images/1/sample.png")
    plt.close()
    return


# Part 2
def part2(x, W):
    """
    Compute the given network's output.
    The first output layer has linear activation.
    The final output has softmax activation.
    """
    return linear_forward(x, W)


# Part 3
def part3(x, W, y):
    """
    Compute the network loss function's gradient
    """
    return dlossdw(x, W, y)


# Test Part3
def test_part3():
    data = loadmat("mnist_all.mat")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        generate_sets(data, train_size = size, test_size = size)
    W = np.random.rand(28 * 28 + 1, 10)
    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))

    grad = part3(x_train, W, y_train)

    h = 0.000001
    # Taking points in the middle of the image
    points = [i * 28 + j for i in range(13, 16) for j in range(13, 16)]
    for point in points:
        W_calc = W.copy()
        W_calc[point, 0] = W[point, 0] + h
        approx_grad = (loss(x_train, W_calc, y_train) - loss(x_train, W, y_train)) / h

        print "gradient for [{}]: {:.3f}. Difference: " \
              "{:.3f}".format(point, grad[point, 0], approx_grad)

    return


# Part 4
def part4():
    data = load_data()

    # Hyperparameters
    data_size = 100
    alpha = 0.00001
    max_iter = 20000
    plot_path = "Report/images/4/gradient_descent.png"

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        generate_sets(data, train_size = data_size, test_size = data_size)

    W = np.ones((28 * 28 + 1, 10)) * .25
    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
    x_val = np.vstack((x_val, np.ones(x_val.shape[1])))
    x_test = np.vstack((x_test, np.ones(x_test.shape[1])))

    W = grad_descent(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test, W, alpha = alpha, max_iter = max_iter, plot_path = plot_path)

    # For testing
    pickle.dump(W, open("temp/4/weight.p", "wb"))

    W = pickle.load(open("temp/4/weight.p", "rb"))

    fig, ax = plt.subplots(5, 2)

    for i in range(len(W.T)):
        num_weights = W.T[i]
        num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
        plt.sca(ax[i // 2, i % 2])
        plt.imshow(num_matrix, cmap = cm.gray)
        plt.axis('off')
    plt.savefig("report/images/4/weights.png")
    plt.close()

    return W


# Part 5
def part5():
    data = load_data()

    # Hyperparameters
    data_size = 100
    alpha = 0.00005
    max_iter = 2000
    gamma = 0.99
    plot_path = "Report/images/5/gradient_descent.png"

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        generate_sets(data, train_size = data_size, test_size = data_size)

    W = np.ones((28 * 28 + 1, 10)) * .25
    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
    x_val = np.vstack((x_val, np.ones(x_val.shape[1])))
    x_test = np.vstack((x_test, np.ones(x_test.shape[1])))

    W = grad_descent(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test, W, alpha = alpha, gamma = gamma, max_iter = max_iter, plot_path = plot_path)

    # For testing
    pickle.dump(W, open("temp/5/weight.p", "wb"))

    W = pickle.load(open("temp/5/weight.p", "rb"))

    fig, ax = plt.subplots(5, 2)

    for i in range(len(W.T)):
        num_weights = W.T[i]
        num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
        plt.sca(ax[i // 2, i % 2])
        plt.imshow(num_matrix, cmap = cm.gray)
        plt.axis('off')
    plt.savefig("report/images/5/weights.png")
    plt.close()

    return W


# Part 6
def part6a(data, W, w1, w2, digit):
    # Hyperparameters
    data_size = 100

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        generate_sets(data, train_size = data_size, test_size = data_size)

    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))

    w1s = np.arange(-10, 10, 0.1)
    w2s = np.arange(-10, 10, 0.1)
    w1_z, w2_z = np.meshgrid(w1s, w2s)

    C = np.zeros((w1s.size, w2s.size))
    W_ = W.copy()
    for i, weight1 in enumerate(w1s):
        for j, weight2 in enumerate(w2s):
            W_[w1, digit] = W[w1, digit] + w1s[i]
            W_[w2, digit] = W[w1, digit] + w2s[j]
            C[i, j] = loss(x_train, W_, y_train)

    return w1_z, w2_z, C


def part6bc(data, W, w1, w2, digit):
    # Hyperparameters
    data_size = 100
    alpha_gd = 9
    alpha_mo = 0.9
    max_iter = 20
    gamma = 0.9
    distance = 5

    W_init = W.copy()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        generate_sets(data, train_size = data_size, test_size = data_size)

    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
    x_val = np.vstack((x_val, np.ones(x_val.shape[1])))
    x_test = np.vstack((x_test, np.ones(x_test.shape[1])))

    W_init[w1, digit], W_init[w2, digit] = W[w1, digit] - distance, W[w2, digit] + distance
    Wb, rec_b = grad_descent_6(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test,
                 W_init, w1, w2, alpha = alpha_gd, gamma = 0, max_iter = max_iter)

    W_init[w1, digit], W_init[w2, digit] = W[w1, digit] - distance, W[w2, digit] + distance
    Wc, rec_c = grad_descent_6(loss, dlossdw, x_train, y_train, x_val, y_val, x_test, y_test,
                 W_init, w1, w2, alpha = alpha_mo, gamma = gamma, max_iter = max_iter)

    return Wb, rec_b, Wc, rec_c


def part6():
    """
    We pick weights for digit 5
    Let w1 be at [13, 13] (13 * 28 + 13 = 377) and
    Let w2 be at [14, 14] (14 * 28 + 14 = 406)
    """
    data = load_data()
    W = pickle.load(open("temp/5/weight.p", 'rb'))

    digit = 5
    w1 = 377
    w2 = 406

    # # pick poor coords to be 1 and 28 * 28 = 784
    # w1 = 1
    # w2 = 5

    w1_pts, w2_pts, C = part6a(data, W, w1, w2, digit)
    Wb, rec_b, Wc, rec_c = part6bc(data, W, w1, w2, digit)
    print rec_b
    print rec_c

    # plt.contour(w1_pts, w2_pts, C, 500, camp=cm.RdBu)
    plt.contour(w1_pts, w2_pts, C, 50)
    plt.plot([a for a, b in rec_b], [b for a, b in rec_b], 'yo-', label="No Momentum")
    plt.plot([a for a, b in rec_c], [b for a, b in rec_c], 'go-', label="Momentum")
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.legend(loc='best')
    plt.title('Contour plot')
    fig = plt.gcf()
    fig.savefig("report/images/6/a.png")
    # fig.savefig("report/images/6/b.png")
    return


# Part 7
def part7():

    return


if __name__ == "__main__":
    np.random.seed(0)
    # part1()
    # part2()
    # part3()
    # test_part3()
    # part4()
    # part5()
    # part6()
