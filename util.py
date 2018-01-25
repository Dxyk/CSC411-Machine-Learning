import os

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
def loss_function(x, y, theta):
    """
    The loss function for binary classification
    Args:
        x ():
        y ():
        theta ():

    Returns:

    """


# ----------- Initializations -----------
# Count actors before any operations
count_actors()

