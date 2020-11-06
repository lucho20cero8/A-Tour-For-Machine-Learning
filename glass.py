# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:32:01 2020

@author: lucho
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#___________________________________________________________________________
class Perceptron(object):
   
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#_____________________________________________________________________________
class AdalineGD(object):
    


    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
     
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

#_____________________________________________________________________________
class AdalineSGD(object):
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
       
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

#_____________________________________________________________________________
def plot_decision_regions(X, y, classifier, resolution=0.02):

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

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
#___________________________________________________________________________

df = pd.read_csv('http://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/glass/glass.data', header=None)
df.tail()

df = pd.read_csv('glass.data', header=None)
df.tail()
#___________________________________________________________________________

y1 = df.iloc[0:29, 10].values
y2 = df.iloc[185:214, 10].values
y = np.concatenate((y1, y2), axis=0)
y = np.where(y == 1, -1, 1)
y  = np.delete(y,(45,31),axis=0)
## 2, 4
X1 = df.iloc[0:29, [2, 4]].values
X2 = df.iloc[185:214, [2, 4]].values      #//Grafica final
X  = np.concatenate((X1, X2), axis=0)
X  = np.delete(X,(45,31),axis=0)

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()




#___________________________________________________________________________
#plot data
plt.scatter(X_std[:29, 0], X_std[:29, 1],
            color='red', marker='o', label='Float glass process')
plt.scatter(X_std[29:54, 0], X_std[29:54, 1],
            color='blue', marker='x', label='Headlamps')

plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')
plt.show()
#____________________________________________________________________________
# X = df.iloc[0:211, [9, 10]].values

# # plot data
# plt.scatter(X[:69, 0], X[:69, 1],
#             marker='1', label='1')
# plt.scatter(X[70:145, 0], X[70:145, 1],
#             marker='2', label='2')
# plt.scatter(X[146:162, 0], X[146:162, 1],
#             marker='3', label='3')
# plt.scatter(X[163:175, 0], X[163:175, 1],
#             marker='4', label='5')
# plt.scatter(X[176:184, 0], X[176:184, 1],
#             marker='+', label='6')
# plt.scatter(X[185:213, 0], X[185:213, 1],
#             marker='x', label='7')


# plt.xlabel('Fe [%]')
# plt.ylabel('Tipo de vidrio')
# plt.legend(loc='upper left')


# plt.savefig('Definitiva.png', dpi=300)
# plt.show()
#____________________________________________________________________________

ppn = Perceptron(eta=0.01, n_iter=10)

ppn.fit(X_std, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()



plot_decision_regions(X_std, y, classifier=ppn)
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')

# plt.savefig('images/02_08.png', dpi=300)
plt.show()
#___________________________________________________________________________


ada = AdalineGD(eta=0.001, n_iter=10)

ada.fit(X_std, y)

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()


plot_decision_regions(X_std, y, classifier=ada)
plt.xlabel('Porcentaje sodio [%]')
plt.ylabel('Porcentaje aluminio [%]')
plt.legend(loc='upper left')

# plt.savefig('images/02_08.png', dpi=300)
plt.show()
