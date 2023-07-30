import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def euler(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 2)
    t = np.linspace(0, T, M + 2)
    delta_t = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 2, N + 2))

    # Set initial and boundary conditions
    F[:, 0] = 1 / (r * T) * (1 - np.exp(-r * (T - t)))
    F[:, N + 1] = 0
    F[M + 1, :] = np.maximum(-x, 0)

    for n in range(M, 0, -1):
        for i in range(1, N + 1):
            F[n - 1, i] = F[n, i] - delta_t * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * delta_x) + \
                          sigma ** 2 * x[i] ** 2 * delta_t / (2 * delta_x ** 2) * (
                                      F[n, i + 1] - 2 * F[n, i] + F[n, i - 1])

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis', edgecolor='none')
    ax.view_init(elev=10, azim=1)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(t, x)')
    ax.set_title('Numerical Solution using Explicit Euler Method')
    plt.savefig('euler_explicit.png')
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
