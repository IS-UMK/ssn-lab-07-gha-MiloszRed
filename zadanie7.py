import numpy as np
from sklearn.datasets import fetch_openml
from utils import plot_gallery
from gha import GHA
import matplotlib.pyplot as plt

class HebbGHA(GHA):
    
    def init(self, X):
        n_features = X.shape[1]
        self.W = np.random.randn(self.n_components, n_features)
        return self

    def fit(self, X):
        self.init(X)

        for epoch in range(self.n_epochs):
            for x in np.random.permutation(X):
                x = x.reshape(-1, 1)
                y = self.W @ x
                
                self.W += self.eta * (np.outer(y, x.T) - np.tril(np.outer(y, y.T)) @ self.W)
        return self


mnist = fetch_openml('mnist_784', data_home='./dane/', parser='auto')
n = 1000
ind = np.random.permutation(mnist.data.shape[0])

X = mnist.data.values[ind[:n]]
X = 2 * X / 255.0 - 1.0
target =  mnist.target.values[ind[:n]]
w, h = 28, 28


n_components = 40
eta = 0.0001
n_epochs = 20

model = HebbGHA(n_components=n_components, eta=eta, n_epochs=n_epochs)
model.fit(X)

# Wizualizacja wektorów własnych cyfr jako obrazków 28x28
plot_gallery(model.W, h=h, w=w, titles=[f'PC {i+1}' for i in range(n_components)])

Z = model.transform(X)
for c in np.unique(target):
    plt.scatter(Z[target==c, 0], Z[target==c, 1], label=c)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
X_reconstructed = model.inverse_transform(Z)

plot_gallery(X, h=h, w=w, titles=target)
plot_gallery(X_reconstructed, h=h, w=w, titles=target)