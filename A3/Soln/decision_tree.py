from util import *
from sklearn.tree import *
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from subprocess import call


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


# if __name__ == "__main__":
#     # Declare input and label
#     X = [[0, 0], [1, 1]]
#     Y = [0, 1]
#
#     # Fitting the model
#     clf = DecisionTreeClassifier()
#     clf = clf.fit(X, Y)
#
#     # Predict the class of samples
#     clf.predict([[2., 2.]])
#
#     # Predict probability of each class
#     clf.predict_proba([[2., 2.]])
#
#     iris = load_iris()
#     clf = DecisionTreeClassifier()
#     clf = clf.fit(iris.data, iris.target)
#
#     dot_data = tree.export_graphviz(clf, out_file=open("./data/tree.dot",
#                                                        mode = "wb"),
#                                     feature_names=iris.feature_names,
#                                     class_names=iris.target_names,
#                                     filled=True, rounded=True,
#                                     special_characters=True)
#     graph = graphviz.Source(dot_data)
#
#     # generate graph
#     os.system("dot -Tpng ./data/tree.dot -o ./data/tree.png")
#
#     clf.predict(iris.data[:1, :])
#     clf.predict_proba(iris.data[:1, :])
