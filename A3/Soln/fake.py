import operator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from util import *


# ==================== Answers ====================
# Part 1
def part1(print_dict = False):
    """
    Parse the data and return three most frequent words
    :param print_dict: Flag for printing the whole directories
    :type print_dict: bool
    :return: None
    :rtype: None
    """
    real_dict = pickle.load(open("./data/real_dict.p", "rb"))
    fake_dict = pickle.load(open("./data/fake_dict.p", "rb"))
    total_dict = pickle.load(open("./data/total_dict.p", "rb"))

    if print_dict:
        sorted_real = sorted(real_dict.items(), key = operator.itemgetter(1),
                             reverse = True)
        sorted_fake = sorted(fake_dict.items(), key = operator.itemgetter(1),
                             reverse = True)
        sorted_total = sorted(total_dict.items(), key = operator.itemgetter(1),
                              reverse = True)
        print "real: \n" + str(sorted_real)
        print "fake: \n" + str(sorted_fake)
        print "total: \n" + str(sorted_total)

    for word in ["clinton", "election", "president"]:
        print "[{}]:".format(word)
        print "\treal: {}".format(real_dict[word])
        print "\tfake: {}".format(fake_dict[word])
        print "\ttotal: {}".format(total_dict[word])

    return


# Part 2
def part2(tune = False):
    """
    Perform naive bayes and tune the parameters m and p_hat
    :param tune: flag for tuning the parameters
    :type tune: bool
    :return: None
    :rtype: None
    """
    # Generate the sets
    sets = separate_sets(seed = 0)
    (train_set, train_label), (val_set, val_label), (test_set, test_label) = sets
    real_dict, fake_dict = get_train_set_word_dict(train_set, train_label)

    # TODO: Tuning p too high?
    if tune:
        max_val_performance, max_correct_count = 0, 0
        opt_m, opt_p_hat = 0, 0

        print "========== Start Tuning =========="
        for m in range(1, 20, 2):
            for p_hat in np.arange(0.1, 1.0, 0.1):
                print "testing on m = {}, p_hat = {}".format(m, p_hat)
                correct_count = 0
                for i in range(len(val_set)):
                    val_words = val_set[i].split()
                    result = naive_bayes(train_label, real_dict, fake_dict, val_words, m, p_hat)
                    if result == val_label[i]:
                        correct_count += 1
                performance = float(correct_count) / float(len(val_set))
                if performance > max_val_performance:
                    print "new best performance: {} ({})".format(performance, correct_count)
                    opt_m, opt_p_hat = m, p_hat
                    max_val_performance = performance
                    max_correct_count = correct_count
            print "===== Iter {} =====".format(m)
            print "\tm: {}".format(opt_m)
            print "\tp_hat: {}".format(opt_p_hat)
            print "\tperformance: {} ({})".format(max_val_performance, max_correct_count)
        print "========== Done Tuning =========="

        pickle.dump((opt_m, opt_p_hat), open("./data/nb_mp_hat.p", mode = "wb"))

    m, p_hat = pickle.load(open("./data/nb_mp_hat.p", mode = "rb"))
    print "m: {}\np_hat: {}".format(m, p_hat)
    train_performance, val_performance, test_performance = 0, 0, 0
    correct_count = 0
    for i in range(len(train_set)):
        train_words = train_set[i].split()
        result = naive_bayes(train_label, real_dict, fake_dict, train_words, m, p_hat)
        if result == train_label[i]:
            correct_count += 1
        train_performance = float(correct_count) / float(len(train_label))

    correct_count = 0
    for i in range(len(val_set)):
        val_words = val_set[i].split()
        result = naive_bayes(train_label, real_dict, fake_dict, val_words, m, p_hat)
        if result == val_label[i]:
            correct_count += 1
        val_performance = float(correct_count) / float(len(val_label))

    correct_count = 0
    for i in range(len(test_set)):
        test_words = test_set[i].split()
        result = naive_bayes(train_label, real_dict, fake_dict, test_words, m, p_hat)
        if result == test_label[i]:
            correct_count += 1
        test_performance = float(correct_count) / float(len(test_label))

    print "train performance = {}".format(train_performance)
    print "validation performance = {}".format(val_performance)
    print "test performance = {}".format(test_performance)



    return


# Part 3
def part3(save = False):
    """
    Get the top words
    :param save:
    :type save:
    :return:
    :rtype:
    """
    real_dict = pickle.load(open("./data/real_dict.p", "rb"))
    fake_dict = pickle.load(open("./data/fake_dict.p", "rb"))
    real_count = len(real_dict.keys())
    fake_count = len(fake_dict.keys())

    real_influence_dict, fake_influence_dict = {}, {}

    for word, word_count in real_dict.iteritems():
        p_word_given_real = float(word_count) / float(real_count)
        # Register the influence of the word to real news to the dict
        real_influence_dict[word] = p_word_given_real

    for word, word_count in fake_dict.iteritems():
        p_word_given_fake = (float(word_count)) / float(fake_count)
        # Register the influence of the word to fake news to the dict
        fake_influence_dict[word] = p_word_given_fake

    if save:
        pickle.dump(real_influence_dict,
                    open("./data/real_influence_dict.p", mode = "wb"))
        pickle.dump(fake_influence_dict,
                    open("./data/fake_influence_dict.p", mode = "wb"))

    sorted_real = sorted(real_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)
    sorted_fake = sorted(fake_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)

    top_10_important_real = sorted_real[:10]
    top_10_unimportant_real = sorted_real[-1:-11:-1]
    top_10_important_fake = sorted_fake[:10]
    top_10_unimportant_fake = sorted_fake[-1:-11:-1]
    print "Real:"
    print top_10_important_real
    print top_10_unimportant_real
    print "Fake:"
    print top_10_important_fake
    print top_10_unimportant_fake

    sorted_real = [tup for tup in sorted_real if tup[0] not in ENGLISH_STOP_WORDS]
    sorted_fake = [tup for tup in sorted_fake if tup[0] not in ENGLISH_STOP_WORDS]
    top_10_important_real = sorted_real[:10]
    top_10_unimportant_real = sorted_real[-1:-11:-1]
    top_10_important_fake = sorted_fake[:10]
    top_10_unimportant_fake = sorted_fake[-1:-11:-1]
    print "\nReal:"
    print top_10_important_real
    print top_10_unimportant_real
    print "Fake:"
    print top_10_important_fake
    print top_10_unimportant_fake
    return


# Part 4
def part4():
    pass


# Part 5
def part5():
    pass


# Part 6
def part6():
    pass


# Part 7
def part7():
    pass


# Part 8
def part8():
    pass


if __name__ == "__main__":
    # construct_word_dict(overwrite = False)
    # part1(print_dict = False)
    # part2(tune = False)
    part3()
    # part4()
    # part5()
    # part6()
    # part7()
    # part8()
