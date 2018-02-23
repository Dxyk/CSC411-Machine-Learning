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
    prediction = np.dot(theta.T, data)
    return prediction


# ----------- Answers -----------

# Part 1
# See faces.pdf


# Part 2
def divide_sets(actor, training_size = 0, path = "./cropped"):
    """
    Given the downloaded data set and a selected actor, return three randomized list
    of training set, validation set and test set.
    Args:
        actor (str): The selected actor
        training_size (int): The size of the training set. return the full training
                             set if set to 0
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
    if training_size != 0:
        training_set = training_set[:min([training_size, len(training_set)])]
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
    actor1_training_set, actor1_validation_set, \
    actor1_test_set = divide_sets(actor1, training_size)
    actor2_training_set, actor2_validation_set, \
    actor2_test_set = divide_sets(actor2, training_size)

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
    if training_size >= 20:
        theta = grad_descent(loss, dlossdx, x, y, theta, 0.005)
    else:
        theta = grad_descent(loss, dlossdx, x, y, theta, 0.001)

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
        for im in test_set:
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
def plot_theta(actor1 = "baldwin", actor2 = "carell", compare_size = 2):
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
    # ret = np.empty((full_theta.shape[0], full_theta.shape[1], 3), dtype=np.uint8)
    # ret[:, :, 0] = full_theta
    # ret[:, :, 1] = full_theta
    # ret[:, :, 2] = full_theta
    imsave("./Report/images/4/a_full_theta.jpg", full_theta)
    # plt.imsave("./Report/images/4/a_full_theta.jpg", ret, cmap = "RdBu")
    # toimage(full_theta).save("./Report/images/4/a_full_theta.jpg")

    two_theta = classify(actor1, actor2, compare_size, validate = False,
                         test = False)
    # print two_theta.shape
    two_theta = np.delete(two_theta, 0)
    two_theta = np.resize(two_theta, (32, 32))
    # ret = np.empty((two_theta.shape[0], two_theta.shape[1], 3), dtype=np.uint8)
    # ret[:, :, 0] = two_theta
    # ret[:, :, 1] = two_theta
    # ret[:, :, 2] = two_theta
    imsave("./Report/images/4/a_two_theta.jpg", two_theta)
    # plt.imsave("./Report/images/4/a_two_theta.jpg", ret, cmap = "RdBu")
    # toimage(two_theta).save("./Report/images/4/a_two_theta.jpg")


# b)
def visualize_gradient():
    pass


# Part 5
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin',
       'Bill Hader', 'Steve Carell']


def overfitting():
    """
    Overfit the data
    We denote male as 1 and female as 0
    Returns:

    """
    training_sizes = [5, 10, 20, 50, 100, 150]
    thetas = [np.zeros((1025, 1)) for i in range(6)]
    training_actor_names = [a.split()[1].lower() for a in act]
    actor_genders = {'bracco': 0,
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
    # test_actor_names = [a for a in actor_names if a not in training_actor_names]

    training_result = dict()
    validation_result = dict()

    for i in range(len(training_sizes)):
        print "----------- Training on size {} -----------".format(training_sizes[i])
        actor_training_set, actor_validation_set, \
        actor_test_set = dict(), dict(), dict()

        for a in training_actor_names:
            actor_training_set[a], actor_validation_set[a], \
            actor_test_set[a] = divide_sets(a, training_sizes[i])
            print "[{}]: {}".format(a, len(actor_training_set[a]))

        # get all training data
        training_set = list(
            itertools.chain.from_iterable(actor_training_set.values()))
        validation_set = list(
            itertools.chain.from_iterable(actor_validation_set.values()))

        x = np.zeros((len(training_set), 1025))
        y = np.zeros((len(training_set), 1))

        # fill the data with given data set
        j = 0
        for actor in actor_training_set.keys():
            for image in actor_training_set[actor]:
                data = process_image(image)
                x[j] = data
                y[j] = actor_genders[actor]
                j += 1

        thetas[i] = grad_descent(loss, dlossdx, x, y, thetas[i], 0.005)

        # test on training set
        total = sum(
            [len(actor_training_set[actor]) for actor in actor_training_set.keys()])
        correct_count = 0
        for actor in actor_training_set.keys():
            for im in actor_training_set[actor]:
                prediction = predict(im, thetas[i])
                if actor_genders[actor] == 1 and norm(prediction) > 0.5:
                    correct_count += 1
                elif actor_genders[actor] == 0 and norm(prediction) <= 0.5:
                    correct_count += 1
        correct_rate = 100. * correct_count / total
        training_result[training_sizes[i]] = correct_rate

        print "Result on [Training Set]: {} / {}\n".format(correct_count, total)

        # test on validation set
        total = sum(
            [len(actor_validation_set[actor]) for actor in
             actor_validation_set.keys()])
        correct_count = 0
        for actor in actor_validation_set.keys():
            for im in actor_validation_set[actor]:
                prediction = predict(im, thetas[i])
                if actor_genders[actor] == 1 and norm(prediction) > 0.5:
                    correct_count += 1
                elif actor_genders[actor] == 0 and norm(prediction) <= 0.5:
                    correct_count += 1
        correct_rate = 100. * correct_count / total
        validation_result[training_sizes[i]] = correct_rate
        print "Result on [Validation Set]: {} / {}\n".format(correct_count, total)

    plt.plot(training_sizes, [training_result[size] for size in training_sizes],
             color = "r", linewidth = 2, marker = "o", label = "Training Set")
    plt.plot(training_sizes, [validation_result[size] for size in training_sizes],
             color = "b", linewidth = 2, marker = "o", label = "Validation Set")

    plt.title("Training Set Size VS Performance")
    plt.xlabel("Training Set Size / images")
    plt.ylabel("Performance / %")
    plt.legend()
    plt.savefig("./Report/images/5/1.jpg")


# Part 6
# see faces.pdf for calculations and util.py for implementations


# Part 7
def multiclass_classification(test_training = True, validate = True):
    # initialize sets
    training_actor_names = [a.split()[1].lower() for a in act]
    training_set, validation_set, test_set = dict(), dict(), dict()
    for actor in training_actor_names:
        training_set[actor], validation_set[actor], \
            test_set[actor] = divide_sets(actor)

    # get input data
    x = np.zeros((len(list(itertools.chain.from_iterable(training_set.values()))),
                  1025))
    y = np.zeros((len(list(itertools.chain.from_iterable(training_set.values()))),
                  len(training_actor_names)))

    k = 0
    for i in range(len(training_actor_names)):
        for im in training_set[training_actor_names[i]]:
            x[k] = process_image(im)
            y[k][i] = 1
            k += 1

    # train theta
    theta = np.zeros((len(training_actor_names), 1025))
    theta = grad_descent_m(loss_m, dlossdx_m, x, y, theta, 0.0000001).T

    # validate on training set
    if test_training is True:
        print "----------- Testing on Training Set -----------"
        total = len(list(itertools.chain.from_iterable(training_set.values())))
        correct_count = 0
        for i in range(len(training_actor_names)):
            for im in training_set[training_actor_names[i]]:
                prediction = predict(im, theta)
                prediction = np.argmax(prediction)
                if prediction == i:
                    correct_count += 1
        print "Result on [Training Set]: {} / {}\n".format(correct_count, total)

    # validate on validation set
    if validate is True:
        print "----------- Testing on Validation Set -----------"
        total = len(list(itertools.chain.from_iterable(validation_set.values())))
        correct_count = 0
        for i in range(len(training_actor_names)):
            for im in validation_set[training_actor_names[i]]:
                prediction = predict(im, theta)
                prediction = np.argmax(prediction)
                if prediction == i:
                    correct_count += 1
        print "Result on [Validation Set]: {} / {}\n".format(correct_count, total)

    return theta


# Part 8
def plot_theta_multiclass():
    training_actor_names = [a.split()[1].lower() for a in act]
    thetas = multiclass_classification(False, False).T

    for i in range(thetas.shape[0]):
        theta = np.delete(thetas[i], 0)
        theta = np.reshape(theta, (32, 32))
        # ret = np.empty((theta.shape[0], theta.shape[1], 3), dtype=np.uint8)
        # ret[:, :, 0] = theta
        # ret[:, :, 1] = theta
        # ret[:, :, 2] = theta
        imsave("./Report/images/8/{}.jpg".format(training_actor_names[i]), theta)
        # imsave("./Report/images/8/{}.jpg".format(i), theta, cmap="RdBu")



if __name__ == "__main__":
    # part 1

    # part 2
    actor = "baldwin"
    a, b, c = divide_sets(actor)
    print "{}\n{}\n{}".format(a, b, c)

    # part 3
    classify()

    # part 4
    # a)
    plot_theta()

    # b)

    # part 5
    overfitting()

    # part 6

    # part 7
    multiclass_classification()

    # part 8
    plot_theta_multiclass()
