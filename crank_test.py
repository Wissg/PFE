import numpy as np
import matplotlib.pyplot as plt


def Kronecker(j, i):
    if j == i:
        return 1
    else:
        return 0


# Crank_Nicolson

def Crank_Nicolson(S0, r, T, N, M, x_max, x_min, sigma):
    x = np.linspace(x_min, x_max, N + 2)
    t = np.linspace(0, T, M + 2)
    deltat = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    V = np.zeros(shape=(M + 2, N + 2))
    C = np.zeros(shape=(M + 2, N + 2))
    K = np.zeros((M + 2, N + 2))
    K2 = np.zeros(shape=(M + 2, N + 2))

    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    D = np.zeros(N + 1)
    D2 = np.zeros(N + 1)

    for n in range(0, M + 2):
        V[n, 0] = 1 / (r * T) * (1 - np.exp(-r * (T - n * deltat))) - x_min * np.exp(-r * (T - t[n]))
        V[n, N + 1] = 0

    for i in range(N + 2):
        V[M + 1, i] = np.maximum(-i * delta_x, 0)

    # for n in range(0, M + 1):
    for n in range(1, M + 1):
        for i in range(1, N + 1):
            A[i] = - deltat * 0.25 * (-(i * r + 1/(T * delta_x)) - (sigma**2) * (i**2))
            B[i] = deltat * 0.25 * (-(i * r + 1/(T * delta_x)) + (sigma**2) * (i**2))
            D[i] = 1 + deltat * 0.5 * (sigma**2) * (i**2)
            h = 1 / (r * T) * (1 - np.exp(-r * (T - (n - 1) * deltat)))
            # K[n, i] = A[i] * V[n - 1, i + 1] + B[i] * V[n - 1, i - 1] + D[i] * V[n - 1, i] - Kronecker(1, i) * B[1] * h
            # K[n, i] = A[i] * V[n, i - 1] + B[i] * V[n, i + 1] + D[i] * V[n, i] - Kronecker(1, i) * B[1] * h
            K[n, i] = B[i] * V[n, i - 1] + A[i] * V[n, i + 1] + D[i] * V[n, i] - Kronecker(1, i) * B[1] * h


        D2[1] = D[1]
        K2[n, 1] = K[n, 1]
        for i in range(2, N + 1):
            D2[i] = D[i] - (B[i] * A[i - 1]) / D2[i - 1]
            K2[n, i] = K[n, i] - (B[i] * K2[n, i - 1]) / D2[i - 1]
        V[n + 1, N] = K2[n, N] / D2[N]
        for i in range(N - 1, 0, -1):
            V[n + 1, i] = (K2[n, i] - A[i] * V[n + 1, i + 1]) / D2[i]

    print(V)
    # plt.plot(x, V[0, :], label="t = 0")
    # plt.plot(x, V[M // 2, :], label="t = T/2")
    # plt.xlabel('Variable x')
    # plt.ylabel('Fonction f')
    # plt.title('Fonction f en 2 dimensions en t=0 et t=T/2')
    # plt.legend()
    # plt.show()

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface at t = 0
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, T, V, cmap='viridis')
    ax.set_xlabel('Variable x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Function f')
    ax.set_title('2D Surface Plot at t = 0')
    plt.savefig("Graph/crank.png")
    plt.show()
    return V

# Paramètres

Xmin = 0
Xmax = 2
S0 = 100
K = 100
T = 1
r = 0.02
sigma = 0.3
N = 99
M = 999

S = np.linspace(1,50,50)
V = np.zeros(len(S))
x = np.linspace(Xmin, Xmax, N + 2)

# Appel de la fonction Crank_Nicolson

Crank_Nicolson(S0, r, T, N, M, Xmax, Xmin, sigma)

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')

