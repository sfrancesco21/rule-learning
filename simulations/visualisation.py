import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def plot_feature_space(agent):
    # Define the grid for plotting
    x = np.linspace(1, 6, 100)
    y = np.linspace(1, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Define means, covariance matrices, and weights
    means0 = []
    covariances0 = []
    alphas0 = []
    means1 = []
    covariances1 = []
    alphas1 = []

    for m, s, l, a in zip(agent.mu, agent.Sigma, agent.label, agent.alpha):
        if l == 0:
            means0.append(m)
            covariances0.append(s)
            alphas0.append(a)
        else:
            means1.append(m)
            covariances1.append(s)
            alphas1.append(a)

    # Initialize the density grid
    density = np.zeros(X.shape)

    # Compute the mixture density
    for mean, cov, weight in zip(means0, covariances0, alphas0):
        rv = multivariate_normal(mean, cov)
        density += weight * rv.pdf(pos)

    for mean, cov, weight in zip(means1, covariances1, alphas1):
        rv = multivariate_normal(mean, cov)
        density -= weight * rv.pdf(pos)

    # Normalize the density
    density /= np.sum(alphas0 + alphas1)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, density, levels=50, cmap='viridis')
    plt.colorbar(label='Density')
    contour = plt.contour(X, Y, density, levels=[0.0], colors='red')
    plt.clabel(contour, inline=True, fontsize=8, fmt='decision boundary')
    plt.title('Feature space representation')
    plt.xticks(ticks=np.arange(1, 7), labels=np.arange(1, 7))
    plt.yticks(ticks=np.arange(1, 7), labels=np.arange(1, 7))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

