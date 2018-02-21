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
    return loadmat("mnist_all.mat")


def generate_set(data, set_type, size = -1):
    """
    generate the training set from data
    :param data: the given data
    :set_type data: dict
    :param set_type: "train" or "test"
    :set_type set_type: str
    :param size: the size of set
    :set_type size: int
    :return: the training set and the corresponding answer
    :rtype: tuple (list, list)
    """
    if set_type not in [TEST, TRAIN]:
        print "generate_set: Invalid set_type"
        return None

    if size == -1:
        arr_size = sum([data[set_type + str(i)].shape[0] for i in range(10)])
    else:
        arr_size = min(sum([data[set_type + str(i)].shape[0] for i in range(10)]),
                       size)

    output_set = np.zeros((28 * 28, arr_size,))
    soln_set = np.zeros((10, arr_size))

    if size == -1:
        count = 0
        for i in range(10):
            data_set = data[set_type + str(i)] / 255.0
            output_set[:, count: count + data_set.shape[0]] = data_set.T
            soln_set[i, count: count + data_set.shape[0]] = 1
            count += data_set.shape[0]
    else:
        count = 0
        for i in range(10):
            curr_set_type = set_type + str(i)
            curr_size = size // 10
            rand_nums = random.sample(range(0, len(data[curr_set_type])), curr_size)
            data_set = np.array([data[curr_set_type][j, :] / 255.0 for j in rand_nums])
            output_set[:, count: count + curr_size] = data_set.T
            soln_set[i, count: count + curr_size] = 1
            count += curr_size

    return output_set, soln_set


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

    x, y = generate_set(data, TRAIN)
    W = np.random.rand(28 * 28 + 1, 10)
    x = np.vstack((x, np.ones(x.shape[1])))

    grad = part3(x, W, y)

    h = 0.000001
    # Taking points in the middle of the image
    points = [i * 28 + j for i in range(13, 16) for j in range(13, 16)]
    for point in points:
        W_calc = W.copy()
        W_calc[point, 0] = W[point, 0] + h
        approx_grad = (loss(x, W_calc, y) - loss(x, W, y)) / h

        print "gradient for [{}]: {:.3f}. Difference: " \
              "{:.3f}".format(point, grad[point, 0], approx_grad)

    return


# Part 4
def part4():
    data = load_data()

    # Hyperparameters
    size = 1000
    alpha = 0.00005
    max_iter = 20000
    plot = True
    plot_path = "Report/images/4/gradient_descent.png"

    x_train, y_train = generate_set(data, TRAIN, size)
    x_test, y_test = generate_set(data, TEST, size)
    print y_train.shape

    W = np.ones((28 * 28 + 1, 10)) * .5
    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
    x_test = np.vstack((x_test, np.ones(x_test.shape[1])))

    W = grad_descent(loss, dlossdw, x_train, y_train, x_test, y_test, W, alpha = alpha, max_iter = max_iter, plot = plot, plot_path = plot_path)

    # For testing
    pickle.dump(W, open("temp/grad_desc.p", "wb"))

    W = pickle.load(open("temp/grad_desc.p", "rb"))

    fig, ax = plt.subplots(5, 2)

    for i in range(len(W.T)):
        num_weights = W.T[i]
        num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
        plt.sca(ax[i // 2, i % 2])
        plt.imshow(num_matrix, cmap = cm.gray)
        plt.axis('off')
    plt.savefig("report/images/4/weights.png")
    plt.close()

    # for i in range(len(W.T)):
    #     num_weights = W.T[i]
    #     num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
    #     mpimg.imsave("Report/images/4/weight{}.png".format(str(i)), num_matrix,
    #                  cmap=cm.gray)

    return W


# Part 5
def part5():
    data = load_data()

    # Hyperparameters
    size = 1000
    alpha = 0.00005
    max_iter = 20000
    gamma = 0.99
    plot = True
    plot_path = "Report/images/5/gradient_descent.png"

    x_train, y_train = generate_set(data, TRAIN, size)
    x_test, y_test = generate_set(data, TEST, size)

    W = np.ones((28 * 28 + 1, 10)) * .5
    x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
    x_test = np.vstack((x_test, np.ones(x_test.shape[1])))

    W = grad_descent(loss, dlossdw, x_train, y_train, x_test, y_test, W, alpha = alpha, gamma = gamma, max_iter = max_iter, plot = plot, plot_path = plot_path)

    # For testing
    pickle.dump(W, open("temp/grad_desc.p", "wb"))

    W = pickle.load(open("temp/grad_desc.p", "rb"))

    fig, ax = plt.subplots(5, 2)

    for i in range(len(W.T)):
        num_weights = W.T[i]
        num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
        plt.sca(ax[i // 2, i % 2])
        plt.imshow(num_matrix, cmap = cm.gray)
        plt.axis('off')
    plt.savefig("report/images/5/weights.png")
    plt.close()

    # for i in range(len(W.T)):
    #     num_weights = W.T[i]
    #     num_matrix = num_weights[:-1].reshape((28, 28)) / 255.0
    #     mpimg.imsave("Report/images/4/weight{}.png".format(str(i)), num_matrix,
    #                  cmap=cm.gray)

    return W


if __name__ == "__main__":
    np.random.seed(0)
    # part1()
    # part2()
    # part3()
    # test_part3()
    # part4()
    part5()
