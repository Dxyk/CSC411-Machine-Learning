from naive_bayes import *
from logistic_regression import *

import numpy as np
import os
import pickle


# ==================== Constants ====================
CLEAN_REAL = "./clean_real.txt"
CLEAN_FAKE = "./clean_fake.txt"

TRAIN_SET = "train_set"
TRAIN_LABEL = "train_label"
VAL_SET = "val_set"
VAL_LABEL = "val_label"
TEST_SET = "test_set"
TEST_LABEL = "test_label"


# ==================== Helper Functions ====================
def construct_file_word_dict(overwrite = False):
    """
    Parse the file and return three dictionaries of words: real, fake and total
    :param overwrite: the flag that indicates to overwrite the existing data
    :type overwrite: bool
    :return: three dictionaries of words: real, fake, total
    :rtype: tuple
    """
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    if os.path.isfile("./data/real_dict.p") and \
            os.path.isfile("./data/fake_dict.p") and \
            os.path.isfile("./data/total_dict.p") and not overwrite:
        return

    print "========== Constructing Dict =========="
    real_dict, fake_dict, total_dict = {}, {}, {}

    # Read through clean_real.txt and fill in dict
    for line in open(CLEAN_REAL, mode = 'r'):
        for word in set(line.strip().split()):
            if word not in real_dict.keys():
                real_dict[word] = 1
            else:
                real_dict[word] += 1
            if word not in total_dict.keys():
                total_dict[word] = 1
            else:
                total_dict[word] += 1
    # Read through clean_fake.txt and fill in dict
    for line in open(CLEAN_FAKE, mode = 'r'):
        for word in set(line.strip().split()):
            if word not in fake_dict.keys():
                fake_dict[word] = 1
            else:
                fake_dict[word] += 1
            if word not in total_dict.keys():
                total_dict[word] = 1
            else:
                total_dict[word] += 1
    pickle.dump(real_dict, open("./data/real_dict.p", "wb"))
    pickle.dump(fake_dict, open("./data/fake_dict.p", "wb"))
    pickle.dump(total_dict, open("./data/total_dict.p", "wb"))
    print "========== Done Constructing Dict =========="
    return real_dict, fake_dict, total_dict


def separate_sets(seed = 0, overwrite = False):
    """
    Separate the two files of headlines into 70:15:15 of training, validation and
    test sets
    :param seed: The random seed
    :type seed: int
    :param overwrite: Flag to overwrite the existing file
    :type overwrite: bool
    :return: A dict containing real and fake training, val, test sets
    :rtype: dict
    """
    np.random.seed(seed)
    sets = {}

    if not overwrite and os.path.isfile("./data/sets.p"):
        sets = pickle.load(open("./data/sets.p", mode = "rb"))
        return sets

    real_lines = open(CLEAN_REAL, mode = 'r').readlines()
    real_lines = [line.strip() for line in real_lines]
    np.random.shuffle(real_lines)
    num_15 = int(len(real_lines) * .15)
    test_set = real_lines[0: num_15]
    test_label = [1] * num_15
    val_set = real_lines[num_15: 2 * num_15]
    val_label = [1] * num_15
    train_set = real_lines[2 * num_15:]
    train_label = [1] * (len(real_lines) - 2 * num_15)

    fake_lines = open(CLEAN_FAKE, mode = 'r').readlines()
    fake_lines = [line.strip() for line in fake_lines]
    np.random.shuffle(fake_lines)
    num_15 = int(len(fake_lines) * .15)
    test_set += fake_lines[0: num_15]
    test_label += [0] * num_15
    val_set += fake_lines[num_15: 2 * num_15]
    val_label += [0] * num_15
    train_set += fake_lines[2 * num_15:]
    train_label += [0] * (len(fake_lines) - 2 * num_15)

    sets[TRAIN_SET] = train_set
    sets[TRAIN_LABEL] = train_label
    sets[VAL_SET] = val_set
    sets[VAL_LABEL] = val_label
    sets[TEST_SET] = test_set
    sets[TEST_LABEL] = test_label

    pickle.dump(sets, open("./data/sets.p", mode = "wb"))

    return sets


def get_set_word_dict(data_set, label):
    """
    Generate the word dictionary for the data_set
    :param data_set: the data_set
    :type data_set: list
    :param label: the labels for training data_set
    :type label: list
    :return: the dictionary for the training data_set
    :rtype: dict
    """
    real_dict, fake_dict = {}, {}

    for i in range(len(data_set)):
        headline = data_set[i]
        if label[i] == 1:
            # use data_set instead of list to avoid over-count
            for word in set(headline.split()):
                if word not in real_dict:
                    real_dict[word] = 1
                else:
                    real_dict[word] += 1
        else:
            for word in set(headline.split()):
                if word not in fake_dict:
                    fake_dict[word] = 1
                else:
                    fake_dict[word] += 1

    return real_dict, fake_dict


# Testing
if __name__ == "__main__":
    sets = separate_sets(seed = 0, overwrite = False)
    train_set = sets[TRAIN_SET]
    train_label = sets[TRAIN_LABEL]
    val_set = sets[VAL_SET]
    val_label = sets[VAL_LABEL]
    test_set = sets[TEST_SET]
    test_label = sets[TEST_LABEL]

    print len(train_set), len(train_label)
    print len(val_set), len(val_label)
    print len(test_set), len(test_label)


