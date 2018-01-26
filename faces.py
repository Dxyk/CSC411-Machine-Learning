from util import *
import get_data
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import *
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


# ----------- HELPER FUNCTIONS -----------
# noinspection PyTypeChecker
def predict(im, theta):
    """
    Predicts the image using trained theta.
    Args:
        im (str): the image file name
        theta (vector[float]): trained theta
    Returns:
        prediction based on the trained theta
    """
    data = imread("./cropped/" + im) / 225.
    data = reshape(data, 1024)
    data = np.insert(data, 0, 1)
    prediction = np.dot(data, theta)
    return prediction


# ----------- Answers -----------

# Part 1
# See faces.pdf


# Part 2
def divide_sets(actor, path = "./cropped"):
    """
    Given the downloaded data set and a selected actor, return three randomized list
    of training set, validation set and test set.
    Args:
        actor (str): The selected actor
        path (str): The path to the cropped files
    Returns:
        ([training set], [validation set], [test set])
        where   |training set|      >= 70
                |validation set|    == 10
                |test set|          == 10
    """
    if actor not in actor_count.keys():
        print "Error: actor [{1}] is not included in the data set".format(actor)
        raise ValueError("Actor not in the data set")
    if actor_count[actor] < 90:
        print "Warning: actor [{0}] only has [{1}] of images, which does not have " \
              "enough photos to satisfy the training " \
              "requirement".format(actor, actor_count[actor])
    all_actor_image = [image for image in os.listdir(path) if actor in image]
    np.random.shuffle(all_actor_image)
    test_set = all_actor_image[0: 10]
    validation_set = all_actor_image[10:20]
    training_set = all_actor_image[20:]
    return training_set, validation_set, test_set


# Part 3
# noinspection PyTypeChecker
def classify(actor1 = "baldwin", actor2 = "carell"):
    """
    Train and apply a linear classifier on actors 1 and 2
    Args:
        actor1 (str): name of the first actor. We label actor1 as 1
        actor2 (str): name of the second actor. We label actor2 as 0
    Returns:
    """
    if actor1 not in actor_names or actor2 not in actor_names:
        print "Error: actor(s) given is not in the data set"
        raise ValueError

    # divide all sets
    actor1_training_set, actor1_validation_set, actor1_test_set = divide_sets(actor1)
    actor2_training_set, actor2_validation_set, actor2_test_set = divide_sets(actor2)
    training_set = actor1_training_set + actor2_training_set
    validation_set = actor1_validation_set + actor2_validation_set
    test_set = actor1_test_set + actor2_test_set

    # initialize input, output and theta to zeros
    # Note: we are removing the bias term by adding a dummy term in x: x_0
    # x: N * D matrix
    # y: N * 1 vector
    # theta: D * 1 vector
    x = np.zeros((len(training_set), 1025))
    y = np.zeros((len(training_set), 1))
    theta = np.zeros((1025, 1))

    # fill the data with given data set
    i = 0
    for image in actor1_training_set:
        data = imread("./cropped/" + image) / 255.0
        data = np.reshape(data, 1024)
        data = np.insert(data, 0, 1)
        x[i] = data
        y[i] = 1
        i += 1

    for image in actor2_training_set:
        data = imread("./cropped/" + image) / 255.0
        data = np.reshape(data, 1024)
        data = np.insert(data, 0, 1)
        x[i] = data
        y[i] = 0
        i += 1

    # use gradient descent to train theta
    theta = grad_descent(loss, dlossdx, x, y, theta, 0.005)

    # validate on validation set
    print "----------- Validating -----------"
    total = len(validation_set)
    correct_count = 0
    for im in validation_set:
        prediction = predict(im, theta)
        if im in actor1_validation_set and norm(prediction) > 0.5:
            correct_count += 1
        elif im in actor2_validation_set and norm(prediction) <= 0.5:
            correct_count += 1
    print "Result on [Validation Set]: {} / {}\n".format(correct_count, total)

    print "----------- Testing -----------"
    # test on test set
    total = len(test_set)
    correct_count = 0
    for im in validation_set:
        prediction = predict(im, theta)
        if im in actor1_validation_set and norm(prediction) > 0.5:
            correct_count += 1
        elif im in actor2_validation_set and norm(prediction) <= 0.5:
            correct_count += 1
    print "Result on [Test Set]: {} / {}".format(correct_count, total)


# Part 4



# Part 5



# Part 6



# Part 7



# Part 8



if __name__ == "__main__":
    # part1

    # part2
    # actor = "baldwin"
    # a, b, c = divide_sets(actor)
    # print "{}\n{}\n{}".format(a, b, c)

    # part3
    classify()
