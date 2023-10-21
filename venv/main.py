import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib

df = pd.read_csv('com_label.txt', names=['ID', 'pSist', 'pDiast',
                                         'qPA', 'pulso', 'respiracao', 'gravidade',
                                         'saida']).iloc[:, 1:]
# print(df.head())

# X = df[['qPA', 'pulso', 'respiracao']].to_numpy()
X = df[['qPA', 'pulso', 'respiracao', 'gravidade']].to_numpy()
# X = df[['gravidade']].to_numpy()
Y = df['saida'].to_numpy()
# Y = df['gravidade'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = tree.DecisionTreeRegressor(max_depth=3)
clf = clf.fit(X_train, Y_train)
tree.plot_tree(clf, feature_names=['qPA', 'pulso', 'respiracao', 'gravidade'],
               class_names=['critico', 'instavel', 'potencial estavel', 'estavel'])
# tree.plot_tree(clf, feature_names=['qPA', 'pulso', 'respiracao'],
#                class_names=['critico', 'instavel', 'potencial estavel', 'estavel'])
plt.show()

clf = clf.predict(X_test)

df_saida = pd.DataFrame({'ID3': clf, 'Y_test': Y_test})
print(df_saida)
# # saving
# # filename = 'model.sav'
# # joblib.dump(clf, filename)
