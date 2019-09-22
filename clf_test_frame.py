from sklearn.model_selection import cross_validate
import time
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np


def _analyze(clf, X_train, y_train, X_test, y_test):
    f = int(len(X_train) / 5)
    train_sizes = [f, 2 * f, 3 * f, 4 * f, 5 * f]
    train_metrics = []
    test_scores = []
    test_times = []

    for size in train_sizes:
        x = X_train[:size, :]
        y = y_train[:size]
        metrics = cross_validate(clf,
                                 X=x,
                                 y=y,
                                 n_jobs=4,
                                 cv=7,
                                 return_train_score=True,
                                 return_estimator=True)
        train_metrics.append(metrics)
        clf = metrics['estimator'][0]

        start_time = time.time()
        test_scores.append(clf.score(X_test, y_test))
        end_time = time.time()
        test_times.append(1000 * (end_time - start_time))

    return train_sizes, train_metrics, test_scores, test_times


def plot_curve(train_sizes, train_data, cross_val_data, test_data, title, xlabel, ylabel, out_dir):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    train_mean = np.mean(train_data, axis=1)
    train_std = np.std(train_data, axis=1)
    cross_val_mean = np.mean(cross_val_data, axis=1)
    cross_val_std = np.std(cross_val_data)

    plt.grid()
    color = 'tab:red'
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.1,
                     color=color)
    plt.plot(train_sizes, train_mean, 'o-', color=color, label="Train")

    color = 'tab:green'
    plt.fill_between(train_sizes,
                     cross_val_mean - cross_val_std,
                     cross_val_mean + cross_val_std,
                     alpha=0.1,
                     color=color)
    plt.plot(train_sizes, cross_val_mean, 'o-', color=color, label='Cross Validation')

    color = 'tab:blue'
    plt.plot(train_sizes, test_data, 'o-', color=color, label='Test')

    plt.legend(loc="best")
    plt.savefig(out_dir)
    plt.clf()


def plot_learning_curve(clf_name, train_sizes, train_scores, cross_val_scores, test_scores, out_dir):
    out_dir = out_dir + '/learning_curve_' + clf_name + '.png'
    title = 'Learning Curve (' + clf_name + ')'
    xlabel = 'Training Samples'
    ylabel = 'Score'
    plot_curve(train_sizes, train_scores, cross_val_scores, test_scores, title, xlabel, ylabel, out_dir)


def plot_times(clf_name, train_sizes, fit_times, cross_val_times, test_times, out_dir):
    out_dir = out_dir + '/times_' + clf_name + '.png'
    title = 'Time Performance (' + clf_name + ')'
    xlabel = 'Training Samples'
    ylabel = 'Time (ms)'
    plot_curve(train_sizes, fit_times, cross_val_times, test_times, title, xlabel, ylabel, out_dir)


def analyze(clf, X_train, y_train, X_test, y_test, out_dir):
    # Make output directory
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'out', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    os.makedirs(out_dir)

    # Given a classifier, train and get training times/scores
    train_sizes, train_metrics, test_scores, test_times = _analyze(clf, X_train, y_train, X_test, y_test)

    train_scores = []
    fit_times = []
    cross_val_scores = []
    cross_val_times = []

    for m in train_metrics:
        train_scores.append(m['train_score'])
        fit_times.append(m['fit_time'] * 1000)

        cross_val_scores.append(m['test_score'])
        cross_val_times.append(m['score_time'] * 1000)

    clf_name = type(clf).__name__
    plot_learning_curve(clf_name, train_sizes, train_scores, cross_val_scores, test_scores, out_dir)
    plot_times(clf_name, train_sizes, fit_times, cross_val_times, test_times, out_dir)
