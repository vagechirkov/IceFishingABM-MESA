import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# toy data: two points close together, one far away
X = np.array([[0.0], [0.1], [2.0]])   # training locations
n = X.shape[0]

# evaluation grid
xs = np.linspace(-1, 3, 400).reshape(-1, 1)

# bandwidth / length scale
h = 0.2
kernel = RBF(length_scale=h)

# KDE: sum of kernels centered at each point
def kde_eval(xs, X, h):
    D = np.abs(xs - X.T)     # pairwise distances
    K = np.exp(-(D**2) / (2*h**2))
    return K.sum(axis=1)     # sum over training points

kde_vals = kde_eval(xs, X, h)

# GP with y=1
gp1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None)
gp1.fit(X, np.ones(n))
gp1_vals, _ = gp1.predict(xs, return_std=True)

# GP with KDE-style targets y = K @ 1
Dtrain = np.abs(X - X.T)
Ktrain = np.exp(-(Dtrain**2) / (2*h**2))
y_kde_targets = Ktrain.sum(axis=1)
gp2 = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None)
gp2.fit(X, y_kde_targets)
gp2_vals, _ = gp2.predict(xs, return_std=True)


plt.figure(figsize=(8,4))
plt.plot(xs, gp2_vals, 'r', linewidth=3, label="GP with KDE targets")
plt.plot(xs, kde_vals, 'k--', linewidth=2, label="KDE (sum of kernels)")
plt.plot(xs, gp1_vals, 'b', linewidth=1, label="GP with y=1")

plt.scatter(X, np.zeros_like(X), c='k', marker='x', label="Points")
plt.legend()
plt.title("KDE vs GP outputs")
plt.xlabel("x")
plt.ylabel("feature value")
plt.show()
