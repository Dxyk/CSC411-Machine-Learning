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
import itertools


# ----------- HELPER FUNCTIONS -----------
def process_image(im):
    """
    Process the given image and output the data
    Args:
        im (str): path to the image
    Returns:
        the processed data
    """
    data = imread("./cropped/" + im) / 225.
    data = reshape(data, 1024)
    data = np.insert(data, 0, 1)
    return data


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
    data = process_image(im)
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
def classify(actor1 = "baldwin", actor2 = "carell", training_size = 0,
             validate = True, test = True):
    """
    Train a linear classifier on actors 1 and 2
    Args:
        actor1 (str): name of the first actor. We label actor1 as 1
        actor2 (str): name of the second actor. We label actor2 as 0
        training_size (int): number of elements in the training set. train the full
                             set if size given is 0.
        validate (bool): indicate if the function call validates the validation set
        test (bool): indicate if the function call tests on the test set
    Returns:
        The trained theta vector
    """
    if actor1 not in actor_names or actor2 not in actor_names:
        print "Error: actor(s) given is not in the data set"
        raise ValueError

    # divide all sets
    actor1_training_set, actor1_validation_set, actor1_test_set = divide_sets(actor1)
    actor2_training_set, actor2_validation_set, actor2_test_set = divide_sets(actor2)
    if training_size != 0 and training_size <= min(actor1_training_set,
                                                   actor2_training_set):
        actor1_training_set = actor1_training_set[0: training_size]
        actor2_training_set = actor2_training_set[0: training_size]

    training_set = actor1_training_set + actor2_training_set
    validation_set = actor1_validation_set + actor2_validation_set
    test_set = actor1_test_set + actor2_test_set

    print "\n----------- Training on {} data: -----------".format(len(training_set))

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
        data = process_image(image)
        x[i] = data
        y[i] = 1
        i += 1

    for image in actor2_training_set:
        data = process_image(image)
        x[i] = data
        y[i] = 0
        i += 1

    # use gradient descent to train theta
    theta = grad_descent(loss, dlossdx, x, y, theta, 0.005)

    # validate on validation set
    if validate is True:
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

    if test is True:
        # test on test set
        print "----------- Testing -----------"
        total = len(test_set)
        correct_count = 0
        for im in validation_set:
            prediction = predict(im, theta)
            if im in actor1_validation_set and norm(prediction) > 0.5:
                correct_count += 1
            elif im in actor2_validation_set and norm(prediction) <= 0.5:
                correct_count += 1
        print "Result on [Test Set]: {} / {}".format(correct_count, total)

    return theta


# Part 4
# TODO: save RGB image
# a)
def compare_and_plot_theta(actor1 = "baldwin", actor2 = "carell", compare_size = 2):
    """
    compare the thetas of different number of training sets
    Args:
        actor1 (str): the first actor's name
        actor2 (str): the second actor's name
        compare_size (int): the comparing training set's size. train full set if 0.
    Returns:
    """
    full_theta = classify(actor1, actor2, validate = False, test = False)
    # Note: theta contains a bias term as the first element so drop it
    full_theta = np.delete(full_theta, 0)
    full_theta = np.reshape(full_theta, (32, 32))
    plt.imsave("./Report/images/4/a_full_theta.jpg", full_theta, cmap = "RdBu")
    # toimage(full_theta).save("./Report/images/4/a_full_theta.jpg")

    two_theta = classify(actor1, actor2, compare_size, validate = False,
                         test = False)
    print two_theta.shape
    two_theta = np.delete(two_theta, 0)
    two_theta = np.resize(two_theta, (32, 32))
    two_theta = np.arange(255)
    imsave("./Report/images/4/a_two_theta.jpg", two_theta)
    # plt.imsave("./Report/images/4/a_two_theta.jpg", two_theta, cmap = "RdBu")
    # toimage(two_theta).save("./Report/images/4/a_two_theta.jpg")


# b)
def visualize_gradient():
    pass


# Part 5
def overfitting():
    """
    Overfit the data
    We denote male as 1 and female as 0
    Returns:

    """
    training_actor_names = [a.split()[1].lower() for a in
                            ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon',
                             'Alec Baldwin',
                             'Bill Hader', 'Steve Carell']]
    act_genders = {'bracco': 0,
                   'chenoweth': 0,
                   'drescher': 0,
                   'ferrera': 0,
                   'gilpin': 0,
                   'harmon': 0,
                   'baldwin': 1,
                   'butler': 1,
                   'carell': 1,
                   'hader': 1,
                   'radcliffe': 1,
                   'vartan': 1}
    test_act_names = [a for a in actor_names if a not in training_actor_names]
    test_act_genders = {}
    act_training_set, act_validate_set, act_test_set = dict(), dict(), dict()
    for a in training_actor_names:
        act_training_set[a], act_validate_set[a], act_test_set[a] = divide_sets(a)
    # get all training data
    training_set = list(itertools.chain.from_iterable(act_training_set.values()))

    x = np.zeros((len(training_set), 1025))
    y = np.zeros((len(training_set), 1))
    theta = np.zeros((1025, 1))

    # fill the data with given data set
    i = 0
    for a in act_training_set.keys():
        for image in act_training_set[a]:
            data = process_image(image)
            x[i] = data
            y[i] = 1
            i += 1


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
    # classify()

    # part4
    # a)
    # compare_and_plot_theta()

    # b)

    # part5
    overfitting()
