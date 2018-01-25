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
def classify(actor1 = "baldwin", actor2 = "carell"):
    actor1_training_set, actor1_validation_set, actor1_test_set = divide_sets(actor1)
    actor2_training_set, actor2_validation_set, actor2_test_set = divide_sets(actor2)



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
