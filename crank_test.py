import numpy as np
import matplotlib.pyplot as plt

def Kronecker(j, i):
    if j == i:
        return 1
    else:
        return 0

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



    for n in range(0, M+2):
        K[n, 0] = 1 / (r * T) * (1 - np.exp(-r * (T-n*deltat))) - x_min * np.exp(-r*(T-t[n]))
        K[n, N + 1] = 0

    for i in range(N + 2):
        K[M + 1, i] = np.maximum(-i * delta_x, 0)

    for n in range(0, M + 1):
        for i in range(1, N + 1):
            A[i] = - deltat * 0.25 * (-(i * r + 1/(T * delta_x)) - (sigma**2) * (i**2))
            B[i] = deltat * 0.25 * (-(i * r + 1/(T * delta_x)) + (sigma**2) * (i**2))
            D[i] = 1 + deltat * 0.5 * (sigma**2) * (i**2)
            K[n, i] = A[i] * K[n, i - 1] + B[i] * K[n, i + 1] + D[i] * K[n, i] - A[i] * K[n, i + 1] - Kronecker(1, i) * B[1] * S0

        D2[1] = D[1]
        K2[n, 1] = K[n, 1]
        for i in range(2, N + 1):
            D2[i] = D[i] - (B[i] * A[i - 1]) / D2[i - 1]
            K2[n, i] = K[n, i] - (B[i] * K2[n, i - 1]) / D2[i - 1]
        V[n + 1, N] = K2[n, N] / D2[N]
        for i in range(N - 1, 0, -1):
            V[n + 1, i] = (K2[n, i] - A[i] * V[n + 1, i + 1]) / D2[i]

    print(V)
    plt.plot(x, V[0, :], label="t = 0")
    plt.plot(x, V[M // 2, :], label="t = T/2")
    plt.xlabel('Variable x')
    plt.ylabel('Fonction f')
    plt.title('Fonction f en 2 dimensions en t=0 et t=T/2')
    plt.legend()
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

# Appel de la fonction Crank_Nicolson
V = Crank_Nicolson(S0, r, T, N, M, Xmax, Xmin, sigma)
