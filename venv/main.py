import csv
import matplotlib.pyplot as plt
from sklearn import tree
# from sklearn.datasets import load_iris
import numpy as np

with open('test.txt', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# print(data)

# data = [[0, 0], [1, 1]]
Y = np.arange(1, 11, 1)
print(Y)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(data, Y)
clf.predict([[2., 2., 2., 2., 2., 2., 2., 2.]])
# clf.predict_proba([[2., 2., 2., 2., 2., 2., 2., 2.]])

# iris = load_iris()
# data, y = iris.data, iris.target
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(data, y)

tree.plot_tree(clf)
plt.show()
