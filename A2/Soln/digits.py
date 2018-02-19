from util import *
from scipy.io import loadmat
from pylab import *
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ==================== HELPER FUNCTIONS ====================
def load_data():
    return loadmat("mnist_all.mat")


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
    #         file_name = "report/img/1/{}_{}.png".format(str(number), str(i))
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
    plt.savefig("report/img/1/sample.png")
    plt.close()
    return


# Part 2
def part2(x, W, b):
    """
    Compute the given network's output.
    The first output layer has linear activation.
    The final output has softmax activation.
    """
    return linear_forward(x, W, b)


# Part 3
def part3(x, W, b, y):
    """
    Compute the network loss function's gradient
    """
    return dlossdw(x, W, b, y)


# Test Part3
def test_part3():
    data = loadmat("mnist_all.mat")

    n = 28 * 28
    k = 10
    num = sum([data["train" + str(i)].shape[0] for i in range(10)])

    x = np.zeros((n, num))
    W = np.random.rand(n, k)
    b = np.ones((k, 1))
    y = np.zeros((k, num))

    count = 0
    for i in range(10):
        data_set = data["train" + str(i)] / 255.0
        x[:, count: count + data_set.shape[0]] = data_set.T
        y[i, count: count + data_set.shape[0]] = 1
        count += data_set.shape[0]

    grad = dlossdw(x, W, b, y)

    h = 0.000001
    # Taking points in the middle of the image
    points = [i * 28 + j for i in range(13, 16) for j in range(13, 16)]
    for point in points:
        W_calc = W.copy()
        W_calc[point, 0] = W[point, 0] + h
        approx_grad = (loss(x, W_calc, b, y) - loss(x, W, b, y)) / h

        print "gradient for [{}]: {:.3f}. Difference: {:.3f}".format(point, grad[point, 0], approx_grad)

    return


# Part 4
def part4():
    # Load the MNIST digit data
    data = loadmat("mnist_all.mat")
    return


if __name__ == "__main__":
    np.random.seed(0)
    # part1()
    # part2()
    # part3()
    test_part3()
