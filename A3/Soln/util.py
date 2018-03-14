import numpy as np
import math
import os
import pickle
import random


# ==================== Constants ====================
CLEAN_REAL = "./clean_real.txt"
CLEAN_FAKE = "./clean_fake.txt"


# ==================== Helper Functions ====================
def construct_word_dict(overwrite = False):
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


def separate_sets(seed = 0):
    """
    Separate the two files of headlines into 70:15:15 of training, validation and
    test sets
    :param seed: The random seed
    :type seed: int
    :return: A tuple containing two tuples of real and fake sets
    :rtype: tuple
    """
    random.seed(seed)

    real_lines = open(CLEAN_REAL, mode = 'r').readlines()
    real_lines = [line.strip() for line in real_lines]
    random.shuffle(real_lines)
    num_15 = int(len(real_lines) * .15)
    test_set = real_lines[0: num_15]
    test_label = [1] * num_15
    val_set = real_lines[num_15: 2 * num_15]
    val_label = [1] * num_15
    train_set = real_lines[2 * num_15:]
    train_label = [1] * (len(real_lines) - 2 * num_15)

    fake_lines = open(CLEAN_FAKE, mode = 'r').readlines()
    fake_lines = [line.strip() for line in fake_lines]
    random.shuffle(fake_lines)
    num_15 = int(len(fake_lines) * .15)
    test_set += fake_lines[0: num_15]
    test_label += [0] * num_15
    val_set += fake_lines[num_15: 2 * num_15]
    val_label += [0] * num_15
    train_set += fake_lines[2 * num_15:]
    train_label += [0] * (len(fake_lines) - 2 * num_15)

    return (train_set, train_label), (val_set, val_label), (test_set, test_label)


def get_train_set_word_dict(train_set, train_label):
    """
    Generate the word dictionary for the training set
    :param train_set: the training set
    :type train_set: list
    :param train_label: the labels for training set
    :type train_label: list
    :return: the dictionary for the training set
    :rtype: dict
    """
    real_dict, fake_dict = {}, {}

    for i in range(len(train_set)):
        headline = train_set[i]
        if train_label[i] == 1:
            # use set instead of list to avoid over-count
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


# ==================== Math ====================
def small_product(num_arr):
    """
    use the fact that a1 x a2 x ... x ak = exp(log(a1) + log(a2) + ... + log(ak)) to
    compute product of small numbers to prevent underflow
    :param num_arr: a list of small numbers
    :type num_arr: list
    :return: the computed result
    :rtype: float
    """
    logged_num = [math.log(n) for n in num_arr]
    return math.exp(sum(logged_num))


def naive_bayes(train_label, real_dict, fake_dict, test_words, m, p_hat):
    """
    The naive bayes classifier
    :param train_label: the training label
    :type train_label: list
    :param test_words: the test list that contains words
    :type test_words: list
    :param m: number of virtual prior
    :type m: int
    :param p_hat: the virtual prior
    :type p_hat: float
    :return: the label 1 (real) or 0 (fake)
    :rtype: int
    """
    real_count = train_label.count(1)
    fake_count = train_label.count(0)
    total_count = len(train_label)

    # Get priors
    p_real = float(real_count) / float(total_count)
    p_fake = float(fake_count) / float(total_count)

    # Get all words probability (count) P(w | c)
    # P(w | c) = count(word, c) / count(c)
    real_probs, fake_probs = [], []
    for word in test_words:
        if word in real_dict.keys():
            word_real_count = real_dict[word]
        else:
            word_real_count = 0
        if word in fake_dict.keys():
            word_fake_count = fake_dict[word]
        else:
            word_fake_count = 0
        p_word_given_real = (float(word_real_count) + m * p_hat) / float(real_count + m)
        p_word_given_fake = (float(word_fake_count) + m * p_hat) / float(fake_count + m)
        real_probs.append(p_word_given_real)
        fake_probs.append(p_word_given_fake)

    # Get the likelihoods and calculate the probability of test being real and fake
    p_real_likelihood = small_product(real_probs)
    p_real_prob = p_real * p_real_likelihood

    p_fake_likelihood = small_product(fake_probs)
    p_fake_prob = p_fake * p_fake_likelihood

    if p_real_prob >= p_fake_prob:
        return 1
    else:
        return 0


# Testing
if __name__ == "__main__":
    sets = separate_sets(seed = 0)
    (train_set, train_label), (val_set, val_label), (val_set, test_label) = sets
    real_dict, fake_dict = get_train_set_word_dict(train_set, train_label)

    m = 1
    p_hat = 0.15
    correct = 0
    # for i in range(len(val_set)):
    #     test_words = val_set[i].split()
    #     result = naive_bayes(train_label, real_dict, fake_dict, test_words, m, p_hat)
    #     if result == val_label[i]:
    #         correct += 1
    # print float(correct) / float(len(val_label))

    test_words = val_set[1].split()
    result = naive_bayes(train_label, real_dict, fake_dict, test_words, m, p_hat)
    print "label: ", val_label[1]
    print "result: ", result

