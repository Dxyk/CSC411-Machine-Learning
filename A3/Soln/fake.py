import operator

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
    (train_set, train_label), (val_set, val_label), (val_set, test_label) = sets

    print len(val_set), len(val_label)
    print val_label.count(1), val_label.count(0)

    # Tuning
    if tune:
        real_dict, fake_dict = get_train_set_word_dict(train_set, train_label)
        max_val_performance, max_correct_count = 0, 0
        opt_m, opt_p_hat = 0, 0

        print "========== Start Tuning =========="
        for m in range(1, 20, 2):
            for p_hat in np.arange(0.05, 1.0, 0.1):
                print "testing on m = {}, p_hat = {}".format(m, p_hat)
                correct_count = 0
                for i in range(len(val_set)):
                    val_words = val_set[i].split()
                    result = naive_bayes(train_set, train_label,real_dict, fake_dict, val_words, m, p_hat)
                    if result == val_label[i]:
                        correct_count += 1
                performance = float(correct_count) / float(len(val_set))
                print "performance: {} ({})".format(performance, correct_count)
                if performance > max_val_performance:
                    opt_m, opt_p_hat = m, p_hat
                    max_val_performance = performance
                    max_correct_count = correct_count
            print "===== Iter {} =====".format(m)
            print "\tm: {}".format(opt_m)
            print "\tp_hat: {}".format(opt_p_hat)
            print "\tperformance: {} ({})".format(max_val_performance, max_correct_count)
        print "========== Done Tuning =========="
        print "m: {}\np_hat: {}".format(opt_m, opt_p_hat)

    return


# Part 3
def part3():
    pass


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
    construct_word_dict()
    # part1(print_dict = False)
    part2(tune = True)
    # part3()
    # part4()
    # part5()
    # part6()
    # part7()
    # part8()
