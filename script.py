import etl
from PrunedDecisionTreeClassifier import PrunedDecisionTreeClassifier, save_tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

import clf_test_frame
import os
import datetime

# NOTE: Uncomment / comment below lines to switch between data sets
# X_train, X_test, y_train, y_test = etl.load_data('./datasets/star/pulsar_stars.csv')
X_train, X_test, y_train, y_test = etl.load_data('./datasets/gestures/gestures.csv')

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

base_dir = os.path.join(os.getcwd(), 'out', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))


def decision_tree():
    out_dir = base_dir + '/decision_tree_0'
    clf = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=0)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)
    clf.fit(X_train, y_train)
    save_tree(clf, out_dir)


    out_dir = base_dir + '/decision_tree_20'
    clf = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=20)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)
    clf.fit(X_train, y_train)
    save_tree(clf, out_dir)

    out_dir = base_dir + '/decision_tree_200'
    clf = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=200)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)
    clf.fit(X_train, y_train)
    save_tree(clf, out_dir)


def neural_networks():
    out_dir = base_dir + '/neural_network_001'
    clf = MLPClassifier(random_state=1, learning_rate_init=.001, learning_rate='invscaling', early_stopping=True)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)

    out_dir = base_dir + '/neural_network_01'
    clf = MLPClassifier(random_state=1, learning_rate_init=.01, learning_rate='invscaling', early_stopping=True)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)

    out_dir = base_dir + '/neural_network_1'
    clf = MLPClassifier(random_state=1, learning_rate_init=.1, learning_rate='invscaling', early_stopping=True)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)


def knn():
    out_dir = base_dir + '/knn_1'
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)

    out_dir = base_dir + '/knn_10'
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)

    out_dir = base_dir + '/knn_100'
    clf = KNeighborsClassifier(n_neighbors=100, weights='distance')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)


def boosting():
    out_dir = base_dir + '/boosting_10'
    tree = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=200)
    clf = GradientBoostingClassifier(init=tree, n_estimators=10)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)

    out_dir = base_dir + '/boosting_50'
    tree = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=200)
    clf = GradientBoostingClassifier(init=tree, n_estimators=50)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)

    out_dir = base_dir + '/boosting_100'
    tree = PrunedDecisionTreeClassifier(random_state=1, pruning_threshold=200)
    clf = GradientBoostingClassifier(init=tree, n_estimators=100)
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir)


def svc():
    out_dir = base_dir + '/svc_rbf'
    clf = SVC(random_state=1, kernel='rbf', gamma='scale')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)

    out_dir = base_dir + '/svc_sigmoid'
    clf = SVC(random_state=1, kernel='sigmoid', gamma='scale')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)

    out_dir = base_dir + '/svc_poly'
    clf = SVC(random_state=1, kernel='poly', gamma='scale')
    clf_test_frame.analyze(clf, X_train, y_train, X_test, y_test, out_dir=out_dir)


# NOTE: Change the line below to call noe of the above methods to run the analysis on the dataset
svc()
