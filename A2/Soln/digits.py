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
    Load the data and save 10 images each from 10 digits
    """
    # Load the MNIST digit data
    data = load_data()

    # loop over training numbers and pick 10 number each
    for number in range(10):
        for i in range(10):
            im = data["train" + str(number)][i].reshape((28, 28)) / 255.0
            file_name = "report/img/1/{}_{}.png".format(str(number), str(i))
            imsave(file_name, im, cmap = cm.gray)

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
def part3(x, W, y, b):
    """
    Compute the network loss function's gradient
    """
    return dlossdw(x, W, y, b)


# Test Part3
def test_part3():
    #PART 3b) Verification Code
    M = loadmat("mnist_all.mat")

    #For our images, n = 7084, and our result ranges from 0 through 9, therefore k = 10.
    # W is (n * k)
    # W = np.ones(shape = (N_NUM,K_NUM))
    N_NUM = 28 * 28
    K_NUM = 10
    M_TRAIN = 60000
    W = np.random.rand(N_NUM, K_NUM)
    b = np.ones(shape = (K_NUM,1))

    x = np.zeros(shape = (N_NUM, M_TRAIN))
    y = np.zeros(shape = (K_NUM, M_TRAIN))

    count = 0
    #Load our example
    for i in range(10):
        currSet = M["train" + str(i)].T / 255.0
        x[:, count: count + currSet.shape[1]] = currSet
        y[i, count: count + currSet.shape[1]] = 1
        count += currSet.shape[1]

    scale = 2 #scale to only include a subsection of the 60000 images
    b = b[:, 0:M_TRAIN / scale]
    x = x[:, 0:M_TRAIN / scale]
    y = y[:, 0:M_TRAIN / scale]

    h = 0.00001
    grad = dlossdw(x, W, b, y)

    print "done calculating gradient"

    for i in range(10):
        W_perturbed = W.copy()
        W_perturbed[350 + i,0] = W[350 + i, 0] + h
        approx_grad = (loss(x,W_perturbed,b,y) - loss(x,W,b,y))/h

        print "grad: %.3f approximate grad: %.3f" %(grad[350 + i, 0], approx_grad)

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
