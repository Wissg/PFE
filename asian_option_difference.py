import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import statistics
from scipy import integrate as intg


def euler(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 2)
    t = np.linspace(0, T, M + 2)
    deltat = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 2, N + 2))

    for n in range(0, M + 1):
        F[n, 0] = 1 / (r * T) * (1 - np.exp(-r * (T - n * deltat)))
        F[n, N + 1] = 0

    for i in range(N + 2):
        F[M + 1, i] = np.maximum(x[i], 0)

    for n in range(M + 1, -1, -1):
        for i in range(1, N + 1):
            F[n - 1, i] = F[n, i] - deltat * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * deltat) + (
                        sigma ** 2 * (x[i]) ** 2 * deltat) / (2 * delta_x ** 2) * (
                                      F[n, i + 1] + F[n, i - 1] - 2 * F[n, i])


    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis')
    ax.set_xlabel('Variable x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Function f')
    ax.set_title('Surface Plot')
    plt.savefig("Graph/euler.png")
    plt.show()


if __name__ == '__main__':
    x_max = 1
    x_min = 0
    S0 = 10
    r = 0.4
    sigma = 0.3
    T = 1
    K = 10
    N = 100
    M = 999
    euler(S0, r, sigma, T, K, N, M, x_max, x_min)
