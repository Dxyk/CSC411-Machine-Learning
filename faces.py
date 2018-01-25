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
        print "Error: actor [{}] is not included in the data set".format(actor)
        raise ValueError("Actor not in the data set")
    if actor_count[actor] < 90:
        print "Warning: actor [{}] does not have enough photos to satisfy the " \
              "training requirement".format(actor)
    all_actor_image = [image for image in os.listdir(path) if actor in image]
    np.random.shuffle(all_actor_image)
    test_set = all_actor_image[0: 10]
    validation_set = all_actor_image[10:20]
    training_set = all_actor_image[20:]
    return training_set, validation_set, test_set


# Part 3



# Part 4



# Part 5



# Part 6



# Part 7



# Part 8



if __name__ == "__main__":
    actor = "baldwin"
    count_actors()
    a, b, c = divide_sets(actor)
    print "{}\n{}\n{}".format(a, b, c)
