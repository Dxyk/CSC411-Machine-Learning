from util import *


def generate_dt_data(word_map, data_set, label):
    np_data_set = np.zeros((len(data_set), len(word_map)))
    np_label = np.zeros((len(label),))
    for idx, headline in enumerate(data_set):
        for word in headline.strip().split():
            np_data_set[idx, word_map.index(word)] += 1
        np_label[idx] = label[idx]
    return np_data_set, np_label


def get_dt_performance(prediction, label):
    correct_count = 0
    for idx in range(len(prediction)):
        if prediction[idx] == label[idx]:
            correct_count += 1
    return float(correct_count) / float(len(prediction))
