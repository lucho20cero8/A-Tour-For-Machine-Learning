# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:50:03 2020

@author: lucho
"""


from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')
    
#_____________________________________________________________________________   
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

#_____________________________________________________________________________

df = pd.read_csv('http://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/glass/glass.data', header=None)
df.tail()

df = pd.read_csv('glass.data', header=None)
df.tail()
#_____________________________________________________________________________

y1 = df.iloc[0:29, 10].values
y2 = df.iloc[163:175, 10].values
y3 = df.iloc[185:214, 10].values
y3 = np.where(y3 == 7, 2, 0)
y = np.concatenate((y1, y2), axis=0)
y = np.where(y == 1, 0, 1)
y = np.concatenate((y, y3), axis=0)


## 2, 4
X1 = df.iloc[0:29, [2, 4]].values
X2 = df.iloc[163:175, [2, 4]].values      #//Grafica final
X3 = df.iloc[185:214, [2, 4]].values 
X  = np.concatenate((X1, X2), axis=0)
X  = np.concatenate((X, X3), axis=0)

# X_std = np.copy(X)
# X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#_____________________________________________________________________________


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)  
  
# Estandarizacion
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
#_____________________________________________________________________________   
#PERCEPTRON
ppn = Perceptron(max_iter=10, eta0=0.01, random_state=1)
ppn.fit(X_train_std, y_train)

    #Errores
y_pred = ppn.predict(X_test_std)
print('Misclassified samples of PPN: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))
    
    #regiones de decisión
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()    
#_____________________________________________________________________________   
#PERCEPTRON + LOGISTIC REGRESSION
lr = LogisticRegression(C=10.0, random_state=1)
lr.fit(X_train_std, y_train)
    #Errores
y_pred = lr.predict(X_test_std)
print('Misclassified samples of PPN+LR: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % lr.score(X_test_std, y_test))
    #regiones de decisión
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()

#_____________________________________________________________________________
#SUPPORT VECTOR MACHINE 
svm = SVC(kernel='linear', C=100.0, random_state=1)
svm.fit(X_train_std, y_train)
  #Errores
y_pred = svm.predict(X_test_std)
print('Misclassified samples of SVM: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % svm.score(X_test_std, y_test))
    #regiones de decisión
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()
#_____________________________________________________________________________
#SUPPORT VECTOR MACHINES + RBF
svm = SVC(kernel='rbf', random_state=1, gamma=0.5, C=100.0)
svm.fit(X_train_std, y_train)
  #Errores
y_pred = svm.predict(X_test_std)
print('Misclassified samples of SVM + RBF: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % svm.score(X_test_std, y_test))
    #regiones de decisión

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_16.png', dpi=300)
plt.show()

#_____________________________________________________________________________   
#DECISION TREE  
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=2, 
                              random_state=1)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
  #Errores
y_pred = tree.predict(X_test)
print('Misclassified samples of DT: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % tree.score(X_test_std, y_test))
    #regiones de decisión
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()  
#_____________________________________________________________________________   
#RANDOM FORESTS
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=10, 
                                random_state=1,
                                n_jobs=4)
forest.fit(X_train, y_train)
  #Errores
y_pred = forest.predict(X_test)
print('Misclassified samples of RF: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % forest.score(X_test_std, y_test))
    #Regiones de decisión
plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(48, 69))
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
plt.show()
