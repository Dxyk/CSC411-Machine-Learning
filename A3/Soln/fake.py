from util import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import operator
import matplotlib.pyplot as plt


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
# TODO
def part2(tune = True):
    """
    Perform naive bayes, tune the parameters m and p_hat and report the results.
    :param tune: flag for tuning the parameters
    :type tune: bool
    :return: None
    :rtype: None
    """
    # Generate the sets
    sets = separate_sets(seed = 0, overwrite = False)
    train_set = sets[TRAIN_SET]
    train_label = sets[TRAIN_LABEL]
    val_set = sets[VAL_SET]
    val_label = sets[VAL_LABEL]
    test_set = sets[TEST_SET]
    test_label = sets[TEST_LABEL]
    real_dict, fake_dict = get_set_word_dict(train_set, train_label)

    if tune:
        max_val_performance, max_correct_count = 0, 0
        opt_m, opt_p_hat = 0, 0

        print "========== Start Tuning =========="
        for m in range(1, 10):
            for p_hat in np.arange(0.05, 1.0, 0.1):
                print "testing on m = {}, p_hat = {}".format(m, p_hat)
                correct_count = 0
                for i in range(len(val_set)):
                    val_words = val_set[i].split()
                    result = naive_bayes(train_label, real_dict, fake_dict, val_words, m, p_hat)
                    if result == val_label[i]:
                        correct_count += 1
                performance = float(correct_count) / float(len(val_set))
                print "Performance: {} ({})".format(performance, correct_count)
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
def part3(debug = False):
    """
    Get the top important present and absent words

    :param debug: Debugging flag
    :type debug: bool
    :return: None
    :rtype: None
    """
    # Load m and p_hat
    m, p_hat = pickle.load(open("./data/nb_mp_hat.p", mode = "rb"))

    # Load the word count dicts
    real_dict = pickle.load(open("./data/real_dict.p", "rb"))
    fake_dict = pickle.load(open("./data/fake_dict.p", "rb"))
    total_dict = pickle.load(open("./data/total_dict.p", "rb"))
    real_count = len(real_dict.keys())
    real_word_count = sum(real_dict.values())
    fake_count = len(fake_dict.keys())
    fake_word_count = sum(fake_dict.values())
    total_count = real_count + fake_count
    # total_count = len(total_dict.keys())
    total_word_count = sum(total_dict.values())

    # Priors
    p_real = float(real_count) / float(total_count)
    p_fake = float(fake_count) / float(total_count)

    if debug:
        print "total real count:", real_count
        print "total fake count:", fake_count
        print "total count:", total_count
        print "total word count: ", total_word_count
        print "p(real): ", p_real
        print "p(fake): ", p_fake
        print "===================="

    real_presence_influence_dict, fake_presence_influence_dict = {}, {}
    real_absence_influence_dict, fake_absence_influence_dict = {}, {}

    for word in total_dict.keys():
        if word in real_dict.keys():
            word_real_count = real_dict[word]
        else:
            word_real_count = 0
        if word in fake_dict.keys():
            word_fake_count = fake_dict[word]
        else:
            word_fake_count = 0
        # p_word = float(total_dict[word]) / float(total_word_count)
        # TODO: results are not right when we add p(word) to the calculations, why?
        p_word = 1
        # P(real | word) = P(word | real) P(real) / P(word)
        p_word_given_real = float(word_real_count + m * p_hat) / float(real_count + m)
        p_real_given_word = p_word_given_real * p_real / p_word
        real_presence_influence_dict[word] = p_real_given_word

        # P(real | not word) = P(not word | real) P(real) / P(word)
        p_not_word_given_real = float(real_count - word_real_count + m * p_hat) / float(real_count + m)
        p_real_given_not_word = p_not_word_given_real * p_real / p_word
        real_absence_influence_dict[word] = p_real_given_not_word

        # P(fake | word) = P(word | fake) P(fake) / P(word)
        p_word_given_fake = float(word_fake_count + m * p_hat) / float(fake_count + m)
        p_fake_given_word = p_word_given_fake * p_fake / p_word
        fake_presence_influence_dict[word] = p_fake_given_word

        # P(fake | not word) = P(not word | fake) P(fake) / P(word)
        p_not_word_given_fake = float(fake_count - word_fake_count + m * p_hat) / float(fake_count + m)
        p_fake_given_not_word = p_not_word_given_fake * p_fake / p_word
        fake_absence_influence_dict[word] = p_fake_given_not_word

        if debug and p_real_given_word > 1 or p_real_given_word < 0:
            print word
            print "total:", total_dict[word]
            print "real count:", word_real_count
            print "fake count:", word_fake_count
            print "P(word | real):", p_word_given_real
            print "P(real):", p_real
            print "P(word):", p_word
            print "P(real | word):", p_real_given_word
            break

    sorted_real_presence = sorted(real_presence_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)
    sorted_real_absence = sorted(real_absence_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)
    sorted_fake_presence = sorted(fake_presence_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)
    sorted_fake_absence = sorted(fake_absence_influence_dict.items(),
                                   key = operator.itemgetter(1), reverse = True)

    top_10_real_presence = sorted_real_presence[:10]
    top_10_real_absence = sorted_real_absence[:10]
    top_10_fake_presence = sorted_fake_presence[:10]
    top_10_fake_absence = sorted_fake_absence[:10]
    print "a:"
    print "\tReal: "
    print "\ttop 10 important presence:", [t[0] for t in top_10_real_presence]
    print "\ttop 10 important absence:", [t[0] for t in top_10_real_absence]
    print "\tFake:"
    print "\ttop 10 important presence:", [t[0] for t in top_10_fake_presence]
    print "\ttop 10 important absence:", [t[0] for t in top_10_fake_absence]

    sorted_real_presence = [tup for tup in sorted_real_presence if tup[0] not in ENGLISH_STOP_WORDS]
    sorted_fake_presence = [tup for tup in sorted_fake_presence if tup[0] not in ENGLISH_STOP_WORDS]
    top_10_real_presence = sorted_real_presence[:10]
    top_10_fake_presence = sorted_fake_presence[:10]
    print "\nb:"
    print "\tReal: "
    print "\ttop 10 important presence:", [t[0] for t in top_10_real_presence]
    print "\tFake:"
    print "\ttop 10 important presence:", [t[0] for t in top_10_fake_presence]

    return


# Part 4
def part4(max_iter = 10000, check_point_len = 50, tune = False):
    """
    :param max_iter: The maximum iterations
    :type max_iter: int
    :param check_point_len: the check point length
    :type check_point_len: int
    :param tune: the tuning flag
    :type tune: bool
    :return: None
    :rtype: None
    """
    # Load Data
    print "========== Loading Data =========="
    sets = separate_sets(seed = 0, overwrite = False)
    train_set = sets[TRAIN_SET]
    train_label = sets[TRAIN_LABEL]
    val_set = sets[VAL_SET]
    val_label = sets[VAL_LABEL]
    test_set = sets[TEST_SET]
    test_label = sets[TEST_LABEL]
    word_dict = pickle.load(open("./data/total_dict.p", mode = "rb"))
    word_map = sorted(word_dict.keys())
    # Process data to np sets
    x_train, y_train = generate_np_data(word_map, train_set, train_label)
    x_val, y_val = generate_np_data(word_map, val_set, val_label)
    x_test, y_test = generate_np_data(word_map, test_set, test_label)

    # Init Weight matrix
    init_theta = np.random.normal(0, 0.05, (x_train.shape[0],))

    if tune:
        tune_lr_params(x_train, y_train, x_val, y_val, x_test, y_test, init_theta,
                       max_iter, check_point_len)

    opt_alpha, opt_reg_lambda = pickle.load(open("./data/lr_params.p", mode = "rb"))

    res = grad_descent(loss_fn, dlossdw, x_train, y_train, x_val, y_val, x_test,
                       y_test, init_theta, opt_alpha, opt_reg_lambda, max_iter,
                       check_point_len)

    theta, train_res, val_res, test_res, iters = res

    pickle.dump(theta, open("data/trained_lr_weight.p", mode = "wb"))

    plt.plot(iters, train_res, "b", label = "Train")
    plt.plot(iters, val_res, "r", label = "Validation")
    plt.plot(iters, test_res, "g", label = "Test")
    plt.ylabel("performance")
    plt.xlabel("iterations")
    plt.title("Logistic Regression Learning curve")
    plt.legend(loc = "best")
    plt.savefig("./Report/images/4/learning_curve.png")
    plt.show()

    return


# Part 5
def part5():
    pass


# Part 6
def part6():
    theta = pickle.load(open("./data/trained_lr_weight.p", mode = "rb"))
    word_dict = pickle.load(open("./data/total_dict.p", mode = "rb"))
    word_map = sorted(word_dict.keys())
    theta = theta[:-1]
    word_weights = {}
    for idx, word in enumerate(word_map):
        word_weights[word] = theta[idx]
    sorted_word_weights = sorted(word_weights.items(), key = operator.itemgetter(1),
                                 reverse = True)

    top_pos = sorted_word_weights[:10]
    top_neg = sorted_word_weights[-1:-11:-1]
    print "a:"
    print "Positive:"
    print [i[0] for i in top_pos]
    print "Negative:"
    print [i[0] for i in top_neg]

    # b
    sorted_word_weights = [word_weight for word_weight in sorted_word_weights if word_weight[0] not in ENGLISH_STOP_WORDS]

    top_pos = sorted_word_weights[:10]
    top_neg = sorted_word_weights[-1:-11:-1]
    print "b:"
    print "Positive:"
    print [i[0] for i in top_pos]
    print "Negative:"
    print [i[0] for i in top_neg]

    return





# Part 7
def part7():
    pass


# Part 8
def part8():
    pass


if __name__ == "__main__":
    # construct_file_word_dict(overwrite = True)
    # separate_sets(seed = 0, overwrite = True)
    # part1(print_dict = False)
    # part2(tune = True)
    # part3(debug = False)
    # part4(tune = False)
    # part5()
    part6()
    # part7()
    # part8()
