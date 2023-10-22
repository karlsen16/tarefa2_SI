import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.externals import joblib


def MLP_reg():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=1000)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, False, False)


def MLP_class():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1, max_iter=1000)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, False, False)


def random_tree():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=6, max_features=2, criterion='entropy', max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, True, False)


def random_tree_reg():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestRegressor(n_estimators=10, max_features=2)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, False, False)


def cart_classifier():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, True, True)


def cart_regression():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf = clf.fit(X_train, Y_train)

    print_results(clf, X_test, Y_test, False, True)


def print_results(par_clf, par_X_test, par_Y_test, accu, tree_):

    if tree_:
        dot_data = tree.export_graphviz(par_clf, out_file=None,
                                        feature_names=['qPA', 'pulso', 'respiracao'],
                                        class_names=['critico', 'instavel', 'potencial estavel', 'estavel'],
                                        filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("Result")

    plt.show()
    par_clf = par_clf.predict(par_X_test)
    print("Mean: ", mean_absolute_error(par_clf, par_Y_test))
    if accu:
        print("Accu: ", accuracy_score(par_clf, par_Y_test))
    df_saida = pd.DataFrame({'ALG': par_clf, 'Y_test': par_Y_test})
    print(df_saida)


df = pd.read_csv('com_label.txt', names=['ID', 'pSist', 'pDiast',
                                         'qPA', 'pulso', 'respiracao', 'gravidade',
                                         'saida']).iloc[:, 1:]


# MLP_reg()
# MLP_class()
# random_tree()
# random_tree_reg()
# cart_classifier()
# cart_regression()

# # saving
# # filename = 'model.sav'
# # joblib.dump(clf, filename)
