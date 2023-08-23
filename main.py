import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import statistics
from scipy import integrate as intg
from mpl_toolkits import mplot3d


def asian_option_price_call(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(S_avg[i]/T - K, 0)
        payoffs_geo[i] = np.maximum(S_avg_geo[i]/T - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo

S0 = np.linspace(10,200,100)
K = 20
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000

price_arith = np.zeros(len(S0))
price_geo = np.zeros(len(S0))

for i in range(len(S0)):
    price_arith[i], price_geo[i] = asian_option_price_call(S0[i], r, K, T, N, sigma, Nmc)

plt.plot(S0, price_arith, label="Prix Call arithmétique")
plt.plot(S0, price_geo, label="Prix Call géométrique")
plt.legend()
plt.show()


def calculate_floating_strike(S):
    K = S[0]  # Initialisation du prix d'exercice flottant à S0 (prix initial de l'actif sous-jacent)
    for j in range(1, len(S)):
        K = np.maximum(K, S[j])  # Mise à jour du prix d'exercice flottant en prenant le maximum entre K et S[j]
    return K


def asian_option_price_call_flottant(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(S[N] - S_avg[i]/T, 0)  # Utilisation du dernier prix d'exercice flottant
        payoffs_geo[i] = np.maximum(S[N] - S_avg_geo[i]/T, 0)  # Utilisation du dernier prix d'exercice flottant

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo

S0 = np.linspace(10,200,100)
K = 20
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000

price_arith = np.zeros(len(S0))
price_geo = np.zeros(len(S0))

for i in range(len(S0)):
    price_arith[i], price_geo[i] = asian_option_price_call_flottant(S0[i], r, K, T, N, sigma, Nmc)

plt.plot(S0, price_arith, label="Prix Call arithmétique")
plt.plot(S0, price_geo, label="Prix Call géométrique")
plt.legend()
plt.show()


def asian_option_price_put_flottant(S0, r, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        K = S0  # Initialisation du premier prix d'exercice flottant à S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        K = S[N]  # Mise à jour du prix d'exercice flottant
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(S_avg[i]/T - S[N], 0)  # Utilisation du dernier prix d'exercice flottant
        payoffs_geo[i] = np.maximum(S_avg_geo[i]/T - S[N], 0)  # Utilisation du dernier prix d'exercice flottant

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo

S0 = np.linspace(10,200,100)
K = 20
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000

price_arith = np.zeros(len(S0))
price_geo = np.zeros(len(S0))

for i in range(len(S0)):
    price_arith[i], price_geo[i] = asian_option_price_put_flottant(S0[i], r, T, N, sigma, Nmc)

plt.plot(S0, price_arith, label="Prix Put arithmétique")
plt.plot(S0, price_geo, label="Prix Put géométrique")
plt.legend()
plt.show()


def asian_option_price_put(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(K - S_avg[i]/T, 0)
        payoffs_geo[i] = np.maximum(K - S_avg_geo[i]/T, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo

S0 = np.linspace(10,200,100)
K = 100
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000

price_arith = np.zeros(len(S0))
price_geo = np.zeros(len(S0))

for i in range(len(S0)):
    price_arith[i], price_geo[i] = asian_option_price_put(S0[i], r, K, T, N, sigma, Nmc)

plt.plot(S0, price_arith, label="Prix Put arithmétique")
plt.plot(S0, price_geo, label="Prix Put géométrique")
plt.legend()
plt.show()

def asian_option_price_call_t(St, At, t, r, K, T, N, sigma, Nmc):
    dt = (T - t) / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = St
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum((At*t + S_avg[i])/T - K, 0)
        payoffs_geo[i] = np.maximum((At*t + S_avg_geo[i])/T - K, 0)

    option_price = np.exp(-r * (T - t)) * np.mean(payoffs)
    option_price_geo = np.exp(-r * (T - t)) * np.mean(payoffs_geo)

    return option_price, option_price_geo

St = np.linspace(10,200,50)
K = 20
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000
At = 100
t = T/2

price_arith = np.zeros(len(St))
price_geo = np.zeros(len(St))

for i in range(len(St)):
    price_arith[i], price_geo[i] = asian_option_price_call_t(St[i], At, t, r, K, T, N, sigma, Nmc)

plt.plot(St, price_arith, label="Prix Call arith")
plt.plot(St, price_geo, label="Prix Call geo")
plt.legend()
plt.show()

St = np.linspace(10,200,50)
At = np.linspace(10,200,50)
price_arith = np.zeros(shape=(len(St),len(At)))
price_geo = np.zeros(shape=(len(St),len(At)))

for i in range(len(St)):
    for j in range(len(At)):
        price_arith[i, j], price_geo[i, j] = asian_option_price_call_t(St[i], At[j], t, r, K, T, N, sigma, Nmc)

A, S = np.meshgrid(At, St)
ax = plt.axes(projection= '3d')
ax.plot_surface(S, A, price_arith, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=1)
ax.set_xlabel('Variable St')
ax.set_ylabel('Variable At')
ax.set_zlabel('Price')
plt.show()

def asian_option_price_put_t(St, At, t, r, K, T, N, sigma, Nmc):
    dt = (T - t) / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = St
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(K - (At*t + S_avg[i])/T, 0)
        payoffs_geo[i] = np.maximum(K - (At*t + S_avg_geo[i])/T, 0)

    option_price = np.exp(-r * (T - t)) * np.mean(payoffs)
    option_price_geo = np.exp(-r * (T - t)) * np.mean(payoffs_geo)

    return option_price, option_price_geo

St = np.linspace(10,200,50)
K = 200
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000
At = 100
t = T/2

price_arith = np.zeros(len(St))
price_geo = np.zeros(len(St))

for i in range(len(St)):
    price_arith[i], price_geo[i] = asian_option_price_put_t(St[i], At, t, r, K, T, N, sigma, Nmc)

plt.plot(St, price_arith, label="Prix Put arith")
plt.plot(St, price_geo, label="Prix Put geo")
plt.legend()
plt.show()

St = np.linspace(10,200,50)
At = np.linspace(10,200,50)
price_arith = np.zeros(shape=(len(St),len(At)))
price_geo = np.zeros(shape=(len(St),len(At)))

for i in range(len(St)):
    for j in range(len(At)):
        price_arith[i, j], price_geo[i, j] = asian_option_price_put_t(St[i], At[j], t, r, K, T, N, sigma, Nmc)

A, S = np.meshgrid(At, St)
ax = plt.axes(projection= '3d')
ax.plot_surface(S, A, price_arith, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=1)
ax.set_xlabel('Variable St')
ax.set_ylabel('Variable At')
ax.set_zlabel('Price')
plt.show()

def asian_option_price_call_flottant_t(St, At, t, r, K, T, N, sigma, Nmc):
    dt = (T - t) / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = St
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum(S[N] - (At*t + S_avg[i])/T, 0)
        payoffs_geo[i] = np.maximum(S[N] - (At*t + S_avg_geo[i])/T, 0)

    option_price = np.exp(-r * (T - t)) * np.mean(payoffs)
    option_price_geo = np.exp(-r * (T - t)) * np.mean(payoffs_geo)

    return option_price, option_price_geo

St = np.linspace(10,200,50)
K = 100
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000
At = 100
t = T/2

price_arith = np.zeros(len(St))
price_geo = np.zeros(len(St))

for i in range(len(St)):
    price_arith[i], price_geo[i] = asian_option_price_call_flottant_t(St[i], At, t, r, K, T, N, sigma, Nmc)

plt.plot(St, price_arith, label="Prix Call arith")
plt.plot(St, price_geo, label="Prix Call geo")
plt.legend()
plt.show()

St = np.linspace(10,200,50)
At = np.linspace(10,200,50)
price_arith = np.zeros(shape=(len(St),len(At)))
price_geo = np.zeros(shape=(len(St),len(At)))

for i in range(len(St)):
    for j in range(len(At)):
        price_arith[i, j], price_geo[i, j] = asian_option_price_call_flottant_t(St[i], At[j], t, r, K, T, N, sigma, Nmc)

A, S = np.meshgrid(At, St)
ax = plt.axes(projection= '3d')
ax.plot_surface(S, A, price_arith, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=1)
ax.set_xlabel('Variable St')
ax.set_ylabel('Variable At')
ax.set_zlabel('Price')
plt.show()

def asian_option_price_put_flottant_t(St, At, t, r, K, T, N, sigma, Nmc):
    dt = (T - t) / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = St
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = np.maximum((At*t + S_avg[i])/T - S[N], 0)
        payoffs_geo[i] = np.maximum((At*t + S_avg_geo[i])/T - S[N], 0)

    option_price = np.exp(-r * (T - t)) * np.mean(payoffs)
    option_price_geo = np.exp(-r * (T - t)) * np.mean(payoffs_geo)

    return option_price, option_price_geo

St = np.linspace(10,200,50)
K = 20
r = 0.05
sigma = 0.5
T = 1
N = 100
Nmc = 1000
At = 100
t = T/2

price_arith = np.zeros(len(St))
price_geo = np.zeros(len(St))

for i in range(len(St)):
    price_arith[i], price_geo[i] = asian_option_price_put_flottant_t(St[i], At, t, r, K, T, N, sigma, Nmc)

plt.plot(St, price_arith, label="Prix Put arith")
plt.plot(St, price_geo, label="Prix Put geo")
plt.legend()
plt.show()

St = np.linspace(10,200,50)
At = np.linspace(10,200,50)
price_arith = np.zeros(shape=(len(St),len(At)))
price_geo = np.zeros(shape=(len(St),len(At)))

for i in range(len(St)):
    for j in range(len(At)):
        price_arith[i, j], price_geo[i, j] = asian_option_price_put_flottant_t(St[i], At[j], t, r, K, T, N, sigma, Nmc)

A, S = np.meshgrid(At, St)
ax = plt.axes(projection= '3d')
ax.plot_surface(S, A, price_arith, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=1)
ax.set_xlabel('Variable St')
ax.set_ylabel('Variable At')
ax.set_zlabel('Price')
plt.show()

def delta(S0, r, K, T, sigma, N, Nmc, h):
    S = np.zeros(N + 1)
    S[0] = S0
    dt = T / N
    payoff_down = np.zeros(Nmc)
    payoff_up = np.zeros(Nmc)
    for i in range(Nmc):
        S[0] = S0
        for t in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)

        # Calculate the delta using finite difference
        A_up = np.mean((S[1:] * (1 + h))[1:], axis=0)
        A_down = np.mean((S[1:] * (1 - h))[1:], axis=0)
        payoff_up[i] = np.maximum(A_up - K, 0)
        payoff_down[i] = np.maximum(A_down - K, 0)
    C_up = np.exp(-r * T) * np.mean(payoff_up)
    C_down = np.exp(-r * T) * np.mean(payoff_down)
    return (C_up - C_down) / (2 * S0 * h)

def gamma(S0, r, K, T, sigma, N, Nmc, h):
    S = np.zeros(N + 1)
    S[0] = S0
    dt = T / N
    payoff_down = np.zeros(Nmc)
    payoff_up = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for t in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)

        # Calculate the delta using finite difference
        A = np.mean(S[1:], axis=0)
        A_up = np.mean((S[1:] * (1 + h))[1:], axis=0)
        A_down = np.mean((S[1:] * (1 - h))[1:], axis=0)
        payoff = np.maximum(A - K, 0)
        payoff_up[i] = np.maximum(A_up - K, 0)
        payoff_down[i] = np.maximum(A_down - K, 0)

    C = np.exp(-r * T) * np.mean(payoff)
    C_up = np.exp(-r * T) * np.mean(payoff_up)
    C_down = np.exp(-r * T) * np.mean(payoff_down)

    gamma = ((C_up - C) / (S0 * h) - (C - C_down) / (S0 * h)) / (2 * S0 * h)
    #gamma = (C_up - 2*C + C_down)/h**2
    return gamma

def gamma_put(S0, r, K, T, sigma, N, Nmc, h):
    S = np.zeros(N + 1)
    S[0] = S0
    dt = T / N
    payoff_down = np.zeros(Nmc)
    payoff_up = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for t in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)

        # Calculate the delta using finite difference
        A = np.mean(S[1:], axis=0)
        A_up = np.mean((S[1:] * (1 + h))[1:], axis=0)
        A_down = np.mean((S[1:] * (1 - h))[1:], axis=0)
        payoff = np.maximum(K - A, 0)
        payoff_up[i] = np.maximum(K - A_up, 0)
        payoff_down[i] = np.maximum(K - A_down, 0)

    C = np.exp(-r * T) * np.mean(payoff)
    C_up = np.exp(-r * T) * np.mean(payoff_up)
    C_down = np.exp(-r * T) * np.mean(payoff_down)

    gamma = ((C_up - C) / (S0 * h) - (C - C_down) / (S0 * h)) / (2 * S0 * h)
    #gamma = (C_up - 2*C + C_down)/h**2
    return gamma

def phi(t, x, T, S, Nmc, N, dt):
    I2 = np.zeros(Nmc)
    for j in range(Nmc):
        I = 0
        for i in range(t, N):
            I += S[i] * dt
        I2[j] = I
    return np.maximum(np.mean(I2) - x, 0)


def EDP_2Dim_fixe(S0, r, K, T, N, sigma, Nmc):
    t = 0
    dt = T / N
    S = np.zeros(N + 1)
    rho = 1 / T
    epsilon = np.zeros(N + 1)
    epsilon[0] = S0
    S[0] = S0
    ph = np.zeros(Nmc + 1)
    for i in range(Nmc):
        for j in range(0, N):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j + 1] = S[j] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
            epsilon[j + 1] = epsilon[j] * (1 - sigma * dW - r * dt + sigma ** 2 * dt) - rho * dt
        ph[i] = phi(t, K / S0, T, S, Nmc, N, dt)
    return np.exp(-r * T) * S0 * np.mean(ph)


def Log_return_symetrique(Nmc, T, N, K, So, r, sigma):
    dt = T / N
    S_sym = np.zeros(N + 1)
    S = np.zeros(N + 1)
    S[0] = So
    S_sym[0] = So
    S_avg = np.zeros(Nmc + 1)
    S_geo = np.zeros(Nmc + 1)
    S_sym_avg = np.zeros(Nmc + 1)
    S_sym_geo = np.zeros(Nmc + 1)
    Res_geo = np.zeros(Nmc + 1)
    Res = np.zeros(Nmc + 1)
    payoffs = np.zeros(Nmc + 1)
    payoffs_geo = np.zeros(Nmc + 1)
    payoffs_sym = np.zeros(Nmc + 1)
    payoffs_sym_geo = np.zeros(Nmc + 1)

    for i in range(Nmc):
        for j in range(0, N):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j + 1] = S[j] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
            S_sym[j + 1] = S[j] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * -dW)
        S_avg[i] = np.mean(S)

        S_geo[i] = statistics.geometric_mean(S)
        S_sym_avg[i] = np.mean(S_sym)
        S_sym_geo[i] = statistics.geometric_mean(S_sym)
        payoffs[i] = np.maximum(S_avg[i] - K, 0)
        payoffs_geo[i] = np.maximum(S_geo[i] - K, 0)
        payoffs_sym[i] = np.maximum(S_sym_avg[i] - K, 0)
        payoffs_sym_geo[i] = np.maximum(S_sym_geo[i] - K, 0)
        Res_geo[i] = payoffs_geo[i] + payoffs_sym_geo[i]
        Res[i] = payoffs[i] + payoffs_sym[i]

    option_price = np.mean(payoffs)
    option_price_geo = np.mean(payoffs_geo)
    option_price_red = np.mean(Res) * 0.5
    option_price_red_geo = np.mean(Res_geo) * 0.5
    return option_price, option_price_geo, option_price_red, option_price_red_geo


# def Asian_solver(m,ns,nI,Smax,Imax,r,D,T,sigma,p,phi1,phi2):
#     dx=Smax/N
#     dt=T/m
#     f=np.zeros(shape=(N+1,m+1))
#     x=np.linspace(0,Smax,ns+1)
#     t=np.linspace(0,T,m+1)
#     for i in range(N+1):
#
#         f[i,m]=np.maximum(-i*dx,0)
#
#     i=np.arange(0,N)
#     M=np.zeros(shape=(N+1,m+1))
#
#     for n in range(0, M + 1):
#         for i in range(1, N + 1):
#
#     for k in range(N-1,m-1):
#         for p in range(N):
#             a[i]=-dt/4 * (-(i*r + 1/(T*delta_x) + sigma**2*i**2))
#             b[i]=-dt/4 * (-(i*r + 1/(T*delta_x) - sigma**2*i**2))
#             c[i]= 1 + dt*0.5*sigma**2*i**2
#             M[i,i-1]=a[i]
#             M[i,i]=b[i]
#             M[i,i+1]=c[i]
#             x[:,nI,k]=phi1(x,Imax,t[k])
#             for j in range(nI-1,-1,-1):
#                 F[0]=p(0,I[j]/T)*np.exp(-r(T-t[k]))
#                 F[i]=x[i]*dt/dI*x[i,j+1,k]+x[i,j,k+1]
#                 F[ns]=phi2(Smax,I[j],t[k])
#                 x[:,j,k]= np.linalg.solve(M,F)
#     return x

def Crank_Nicolson(r, T, N, M, x_max, x_min, sigma):
    x = np.linspace(x_min, x_max, N + 2)
    t = np.linspace(0, T, M + 2)
    deltat = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    V = np.zeros(shape=(N + 2, M + 2))
    A = np.zeros(N + 2)
    B = np.zeros(N + 2)
    C = np.zeros(N + 2)

    # V[:, 0] = np.maximum()

    for n in range(0, M+1):
        V[0, n] = 1 / (r * T) * (1 - np.exp(-r * (T-n*deltat))) - x_min*np.exp(-r*(T-t[n]))
        V[N + 1, n] = 0

    for i in range(0,N+2):
        V[i, M+1] = np.maximum(-i*delta_x, 0)

    for n in range(0, M + 1):
        for i in range(1, N + 1):
            A[i] = - deltat*0.25*(-(i*r + 1/(T*delta_x)) - (sigma**2)*(i**2))
            B[i] = deltat*0.25*(-(i*r + 1/(T*delta_x)) + (sigma**2)*(i**2))
            C[i] = 1 + deltat*0.5*(sigma**2)*(i**2)

            # Solve tridiagonal system of equations
            P = np.zeros(N + 2)
            Q = np.zeros(N + 2)

            P[1] = A[1] / C[1]
            Q[1] = V[1, n] / C[1]

            for i in range(2, N + 1):
                P[i] = A[i] / (C[i] - B[i] * P[i - 1])
                Q[i] = (V[i, n] - B[i] * Q[i - 1]) / (C[i] - B[i] * P[i - 1])

            for i in range(N, 1, -1):
                V[i, n + 1] = P[i] * V[i + 1, n + 1] + Q[i]

    plt.plot(x, V[:, M], label="option_price")
    plt.xlabel('Spot Price')
    plt.ylabel('Option Price')
    plt.title('Asian Option Pricing')
    plt.legend()
    plt.show()
    return V


def Crank_Nicolson2(r, T, N, M, x_max, x_min, sigma):
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    deltat = T / M
    delta_x = (x_max - x_min) / N
    
    K = np.zeros(shape = (N+1,N+1))
    U = np.zeros(shape = (N+1, M+1))
    F = np.zeros(N+1)
    i = np.arange(1,N)
    for k in range(M-1, -1, -1):
        a_ik = deltat*0.25*(-(i*r + 1/(T*delta_x)) - (sigma**2)*(i**2))
        b_ik = deltat*0.25*(-(i*r + 1/(T*delta_x)) + (sigma**2)*(i**2))
        c_ik = 1 + deltat*0.5*(sigma**2)*(i**2)
        M[i, i-1] = b_ik
        M[i, i] = c_ik
        M[i,i+1] = a_ik
        F[0] = 1
        F[i] = U[i, k+1]
        F[N] = 1
        U[:,k] = linalg.solve(M,F)
        return U
        
def Crank_Nicolson3(r, T, N, M, x_max, x_min, sigma): 
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    deltat = T / M
    delta_x = (x_max - x_min) / N
    A = np.zeros(N+1)
    B = np.zeros(N+1)
    C = np.zeros(N+1)

    U = np.zeros(shape=(N+1,M+1))
    for i in range(0, N+1):
        U[i,N] = np.maximum(x[i],0)
        
    for k in range(0, M+1):
        U[0, k] = 1 / (r * T) * (1 - np.exp(-r * (T-t[k]))) - x_min*np.exp(-r*(T-t[k]))
        U[N, k] = 0
      
    for i in range(1, N + 1): 
        A[i] = - deltat*0.25*(-(x[i]*r + 1/(T*delta_x)) - (sigma**2)*(x[i]**2))
        B[i] = deltat*0.25*(-(x[i]*r + 1/(T*delta_x)) + (sigma**2)*(x[i]**2))
        C[i] = 1 + deltat*0.5*(sigma**2)*(x[i]**2)
    
    i = np.arange(1,N)
    for k in range(0, M):
        U[i, k+1] = B[i]*U[i-1,k] + C[i]*U[i,k] + A[i]*U[i+1,k]
        
    plt.plot(U[:,0], x)  
    plt.show()
    return U

U = Crank_Nicolson3(0.02, 1, 99, 999, 2, 0, 0.3)       
        
        

if __name__ == '__main__':

    K = 50
    p = 5

    esp = np.linspace(0, 100, 1000)
    x = np.zeros(len(esp))
    x1 = np.zeros(len(esp))
    # for i in range(len(esp)):
    #     x[i] = np.maximum(esp[i]-K,0) - p
    #     x1[i] = np.maximum(K - esp[i], 0) - p
    #
    #
    # plt.plot(esp, x,label="Payoff Call")
    # plt.plot(esp, x1,label="Payoff Put")
    #
    # plt.axhline(y=0, color='black')
    # plt.axvline(x=K, color='r', linestyle='--',label='K = '+str(K))
    # plt.xlabel('Esperance[S]')
    # plt.ylabel('payoff')
    # plt.title("Payoff (K = "+str(K)+", prime = "+str(p)+")")
    # plt.legend()
    # plt.savefig('Graph\payoff.png')
    # plt.show()

    S0 = 1
    K = 10
    r = 0.04
    sigma = 0.5
    T = 1
    N = 100
    Nmc = 1000
    S = np.linspace(S0, 20, 40)
    h = 1

    calls = np.zeros(len(S))
    callsFloat = np.zeros(len(S))
    puts = np.zeros(len(S))
    calls_geo = np.zeros(len(S))
    calls_geoFloat = np.zeros(len(S))
    puts_geo = np.zeros(len(S))
    delt = np.zeros(len(S))
    gamm = np.zeros(len(S))
    gamm_put = np.zeros(len(S))
    edp = np.zeros(len(S))

    for i in range(len(S)):
    #     # calls[i], calls_geo[i] = asian_option_price_call(S[i], r, K, T, N, sigma, Nmc)
    #     # puts[i], puts_geo[i] = asian_option_price_put(S[i], r, K, T, N, sigma, Nmc)
    #     # delt[i] = delta(S[i], r, K, T, sigma, N, Nmc, h)
           gamm[i] = gamma(S[i], r, K, T, sigma, N, Nmc, h)
           gamm_put[i] = gamma_put(S[i], r, K, T, sigma, N, Nmc, h)
    #     # print("Asian call option price:", calls[i])
    #     # print("Asian put option price:", puts[i])
    #     # edp[i] = EDP_2Dim_fixe(S0, r, K, T, N, sigma, Nmc)
    #     # callsFloat[i], calls_geoFloat[i] = asian_option_price_call_flottant(S[i], r, K, T, N, sigma, Nmc)
    #
    # plt.plot(S, callsFloat, label='Asian Call arithmétique flottant')
    # plt.plot(S, calls_geoFloat, label='Asian Call géométrique flottant')
    #
    # plt.plot(S, calls, label='Asian Call arithmétique')
    # plt.plot(S, edp, label='edp')
    # plt.plot(S, puts, label='Asian Put arithmétique')
    # plt.plot(S, calls_geo, label='Asian Call géométrique')
    # plt.plot(S, puts_geo, label='Asian Put géométrique')
    # plt.xlabel('$S_0$')
    # plt.ylabel('Price Value')
    # plt.legend()
    # # plt.savefig('Graph\AsianOptionPrice.png')
    # plt.show()
    #
    # plt.plot(S, delt, label='delta call')
    # plt.xlabel('$S_0$')
    # plt.ylabel('Price Value')
    # plt.legend()
    # plt.savefig('Graph\delta.png')
    # plt.show()

    plt.plot(S, gamm, label='gamma call')
    plt.xlabel('$S_0$')
    plt.ylabel('Gamma')
    plt.legend()
    #plt.savefig('Graph\gamma.png')
    plt.show()
    
    plt.plot(S, gamm, label='gamma put')
    plt.xlabel('$S_0$')
    plt.ylabel('Gamma')
    plt.legend()
    #plt.savefig('Graph\gamma.png')
    plt.show()
    #
    # S0 = 10
    # K = 1
    # r = 0.1
    # sigma = 0.5
    # T = 1
    # N = np.linspace(10, 100, 100)
    # Nmc = 1000
    #
    # option_price = np.zeros(len(N))
    # option_price_geo = np.zeros(len(N))
    # option_price_red = np.zeros(len(N))
    # option_price_red_geo = np.zeros(len(N))
    #
    # for i in range(len(N)):
    #     option_price[i], option_price_geo[i], option_price_red[i], option_price_red_geo[i] = Log_return_symetrique(Nmc,
    #                                                                                                                T,
    #                                                                                                                int(
    #                                                                                                                    N[
    #                                                                                                                        i]),
    #                                                                                                                K,
    #                                                                                                                S0,
    #                                                                                                                r,
    #                                                                                                                sigma)
    #
    # # plt.plot(S, Valeur1, label="estime1")
    # plt.plot(N, option_price, label="option_price")
    # plt.plot(N, option_price_geo, label="estoption_price_geo")
    # plt.plot(N, option_price_red, label="option_price_red")
    # plt.plot(N, option_price_red_geo, label="option_price_red_geo")
    #
    # plt.legend()
    # plt.show()

    x_max = 2
    x_min = 0
    M = 999
    N = 99
    r = 0.002
    sigma = 0.3
    T =1
    K = 100
    S0 = 100

    V = Crank_Nicolson(r, T, N, M, x_max, x_min, sigma)
    
def Payoff_Asiatique (St, A, t, N, sigma, r, T):
    delta_t = (T-t)/N
    somme = 0
    S = np.zeros(N+1)
    S[0] = St
    for n in range(0,N):
        S[n+1] = S[n]*np.exp(r - sigma**2)*delta_t + sigma*np.random.randn(1)*np.sqrt(delta_t)
        somme = somme + S[n]*delta_t
    result = np.maximum((A*t + somme)/T,0)
    return result

def Prix_Asiatique(St, A, t, Nmc, N, sigma, r, T):
    somme = 0
    for k in range(1,Nmc+1):
        somme = somme + Payoff_Asiatique(St, A, t, N, sigma, r, T)
    result = np.exp(-r*(T-t))*somme/Nmc
    return result

def Prix_Asiatique_Final(t, Nmc, N, sigma, r, T):
    St = np.zeros(201)
    A = np.zeros(201)
    prix_option = np.zeros(shape=(201, 201))
    for k in range(1, 201):
        for j in range(1,201):
            St[k] = k - 1
            A[j] = j -1 
            prix_option[k, j] = Prix_Asiatique(St[k], A[j], t, Nmc, N, sigma, r, T)
    plt.plot(St, A, prix_option)
    plt.show()
    
T=1
r=0.05
sigma = 0.2
Nmc = 200
N = 100
t = T/2

Prix_Asiatique_Final(t, Nmc, N, sigma, r, T)
        
        
        
    

