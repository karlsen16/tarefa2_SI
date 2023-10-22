import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score, confusion_matrix, \
                            mean_squared_error, mean_absolute_error, max_error

from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.externals import joblib


def MLP_reg():
    X = df[['qPA', 'pulso', 'respiracao']]
    X = scaler.fit_transform(X)
    Y = df['gravidade']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MLPRegressor(solver='lbfgs', learning_rate='adaptive', learning_rate_init=0.01, alpha=1e-5,
                       hidden_layer_sizes=(10, 8, 7), max_iter=10000)
    clf = clf.fit(X_train, Y_train)

    print("MLP - REG")
    print_results(clf, X_test, Y_test, False, False)


def MLP_class():
    X = df[['qPA', 'pulso', 'respiracao']]
    X = scaler.fit_transform(X)
    Y = df['saida']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MLPClassifier(solver='lbfgs', learning_rate='adaptive', learning_rate_init=0.01, alpha=1e-5,
                        hidden_layer_sizes=(10, 8, 7), max_iter=10000)
    clf = clf.fit(X_train, Y_train)

    print("MLP - CLASS")
    print_results(clf, X_test, Y_test, True, False)


def random_tree():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=6, max_features=2, criterion='entropy', max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print("RANDOM FOREST - CLASS")
    print_results(clf, X_test, Y_test, True, False)


def random_tree_reg():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=10, max_features=2)
    clf = clf.fit(X_train, Y_train)

    print("RANDOM FOREST - REG")
    print_results(clf, X_test, Y_test, False, False)


def cart_classifier():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print("CART- CLASS")
    print_results(clf, X_test, Y_test, True, True)


def cart_regression():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print("CART- REG")
    print_results(clf, X_test, Y_test, False, True)


def print_results(par_clf, par_X_test, par_Y_test, classi, tree_):

    if tree_:
        dot_data = tree.export_graphviz(par_clf, out_file=None,
                                        feature_names=['qPA', 'pulso', 'respiracao'],
                                        class_names=['critico', 'instavel', 'potencial estavel', 'estavel'],
                                        filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Result")

    plt.show()
    par_clf = par_clf.predict(par_X_test)

    print("Mean Absolute Error: ", mean_absolute_error(par_Y_test, par_clf))
    print("Mean Squared Error: ", mean_squared_error(par_Y_test, par_clf))
    print("Max Error: ", max_error(par_Y_test, par_clf))

    if classi:
        print("Prec: ", precision_score(par_Y_test, par_clf, average=None, zero_division=np.nan))
        print("Recall: ", recall_score(par_Y_test, par_clf, average=None))
        print("F-Measure: ", f1_score(par_Y_test, par_clf, average=None))
        print("Accu: ", accuracy_score(par_clf, par_Y_test))
        print("Conf. Matrix: \n", confusion_matrix(par_clf, par_Y_test))
    # df_saida = pd.DataFrame({'ALG': par_clf, 'Y_test': par_Y_test})
    # print(df_saida)


df = pd.read_csv('com_label.txt', names=['ID', 'pSist', 'pDiast',
                                         'qPA', 'pulso', 'respiracao', 'gravidade',
                                         'saida']).iloc[:, 3:]

scaler = preprocessing.StandardScaler()

MLP_class()
# MLP_reg()
random_tree()
# random_tree_reg()
cart_classifier()
# cart_regression()

# # saving
# # filename = 'model.sav'
# # joblib.dump(clf, filename)
