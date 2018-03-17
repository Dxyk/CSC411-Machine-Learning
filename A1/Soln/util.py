import os
import numpy as np
from scipy.stats import norm

# This file contains all global variables, helper functions and calculus functions

# ----------- GLOBAL VARIABLES -----------
# list of actors
actors = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
# list of actors with their lower case last name
actor_names = [actor.split()[1].lower() for actor in actors]
# a dict of actor : count
actor_count = {actor_name: 0 for actor_name in actor_names}


# ----------- HELPER FUNCTIONS -----------
def count_actors(path = "./cropped"):
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


def print_actor_count():
    """
    Print the actor : count dictionary
    """
    for actor, count in actor_count.iteritems():
        print "{} : {}".format(actor, count)


# ----------- CALCULUS FUNCTIONS -----------
# ------ binary classifications ------
# noinspection PyTypeChecker
def loss(x, y, theta):
    """
    The loss function for binary classification
    """
    return np.sum((np.dot(x, theta) - y) ** 2) / (2.0 * x.shape[0])


def dlossdx(x, y, theta):
    """
    The derivative of the loss function
    """
    y_pred = np.matmul(x, theta)
    error = y_pred - y
    return np.matmul(x.T, error) / float(x.shape[0])


def grad_descent(loss, dlossdx, x, y, init_theta, alpha):
    """
    Gradient descent for binary classifier
    Args:
        loss (fn): the loss function
        dlossdx (fn): gradient function
        x (matrix[float]): input matrix
        y (vector[float]): target vector
        init_theta (vector[float]): the initial theta vector
        alpha (float): learning rate
    Returns:
        the theta vector after gradient descent
    """
    print "----------- Starting Gradient Descent -----------"
    eps = 1e-5
    prev_theta = init_theta - 10 * eps
    theta = init_theta.copy()
    max_iter = 100000
    i = 0

    while norm(theta - prev_theta) > eps and i < max_iter:
        prev_theta = theta.copy()
        theta -= alpha * dlossdx(x, y, theta)
        if i % 5000 == 0 or i == max_iter - 1:
            print "Iteration: {}\nCost:{}\n".format(i, loss(x, y, theta))
        i += 1
    return theta


# ------ multiclass classifications ------
def loss_m(x, y, theta):
    """
    The loss function for multiclass classification
    """
    return sum((np.dot(x, theta.T) - y) ** 2)


def dlossdx_m(x, y, theta):
    """
    The gradient for the loss function for multiclass classification
    """
    return 2*(np.dot(x.T, np.dot(x, theta.T) - y)).T


def grad_descent_m(loss_m, dlossdx_m, x, y, init_theta, alpha):
    """
    Gradient descent for multiclass classifier
    Args:
        loss (fn): the loss function
        dlossdx (fn): gradient function
        x (matrix[float]): input matrix
        y (vector[float]): target vector
        init_theta (vector[float]): the initial theta vector
        alpha (float): learning rate
    Returns:
        the theta vector after gradient descent
    """
    print "----------- Starting Gradient Descent -----------"
    eps = 1e-5
    prev_theta = init_theta - 10 * eps
    theta = init_theta.copy()
    max_iter = 100000
    i = 0

    while norm(theta - prev_theta) > eps and i < max_iter:
        prev_theta = theta.copy()
        theta -= alpha * dlossdx_m(x, y, theta)
        if i % 5000 == 0 or i == max_iter - 1:
            print "Iteration: {}\nCost:{}\n".format(i, loss_m(x, y, theta))
        i += 1
    return theta


# ----------- Initializations -----------
# Count actors before any operations
count_actors()

