import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree, preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
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

    clf = MLPRegressor(activation='tanh', alpha=0.001, early_stopping=False,
                       hidden_layer_sizes=(100,), learning_rate='adaptive',
                       max_iter=8000, solver='lbfgs')

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_MLP_reg, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("MLP - REG")
    print_results(clf, X_test, Y_test, False, False)
    # print("Melhores parâmetros:", grid_search.best_params_)


def MLP_class():
    X = df[['qPA', 'pulso', 'respiracao']]
    X = scaler.fit_transform(X)
    Y = df['saida']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = MLPClassifier(activation='logistic', alpha=0.001, early_stopping=False,
                        hidden_layer_sizes=(50,), learning_rate='adaptive',
                        max_iter=8000, solver='lbfgs')

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_MLP_class, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("MLP - CLASS")
    print_results(clf, X_test, Y_test, True, False)
    # print("Melhores parâmetros:", grid_search.best_params_)


def random_tree():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=30,
                                 max_features='sqrt', min_samples_leaf=1, min_samples_split=2,
                                 n_estimators=30)

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_RF_class, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("RANDOM FOREST - CLASS")
    print_results(clf, X_test, Y_test, True, False)
    # print("Melhores parâmetros:", grid_search.best_params_)


def random_tree_reg():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = RandomForestRegressor(bootstrap=False, max_depth=20, max_features='sqrt',
                                min_samples_leaf=1, min_samples_split=2, n_estimators=30)

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_RF_reg, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("RANDOM FOREST - REG")
    print_results(clf, X_test, Y_test, False, False)
    # print("Melhores parâmetros:", grid_search.best_params_)


def cart_classifier():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['saida'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=20, max_features='sqrt',
                                      min_samples_leaf=1, min_samples_split=5)

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_CART_class, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("CART- CLASS")
    print_results(clf, X_test, Y_test, True, True)
    # print("Melhores parâmetros:", grid_search.best_params_)


def cart_regression():
    X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
    Y = df['gravidade'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = tree.DecisionTreeRegressor(criterion='squared_error', max_depth=None, max_features='log2',
                                     min_samples_leaf=2, min_samples_split=2)

    clf = clf.fit(X_train, Y_train)
    # grid_search = GridSearchCV(clf, param_grid_CART_reg, cv=5)
    # clf = grid_search.fit(X_train, Y_train)

    print("CART- REG")
    print_results(clf, X_test, Y_test, False, True)
    # print("Melhores parâmetros:", grid_search.best_params_)


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


df = pd.read_csv('com_label.txt', names=['ID', 'pSist', 'pDiast',
                                         'qPA', 'pulso', 'respiracao', 'gravidade',
                                         'saida']).iloc[:, 3:]

scaler = preprocessing.StandardScaler()

param_grid_CART_class = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

param_grid_CART_reg = {
    'criterion': ['squared_error', 'poisson', 'friedman_mse', 'absolute_error'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

param_grid_RF_reg = {
    'n_estimators': [5, 10, 30],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_grid_RF_class = {
    'n_estimators': [5, 10, 30],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

param_grid_MLP_reg = {
    'hidden_layer_sizes': [(100, 50), (50, 25), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['lbfgs'],
    'alpha': [0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [300],
    'early_stopping': [False]
}

param_grid_MLP_class = {
    'hidden_layer_sizes': [(100, 50), (50, 25), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['lbfgs'],
    'alpha': [0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [300],
    'early_stopping': [False]
}

# MLP_class()
MLP_reg()
# random_tree()
# random_tree_reg()
# cart_classifier()
# cart_regression()

# # saving
# # filename = 'model.sav'
# # joblib.dump(clf, filename)
