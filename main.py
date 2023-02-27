import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import statistics
from scipy import integrate as intg


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
        payoffs[i] = max(S_avg[i] - K, 0)
        payoffs_geo[i] = max(S_avg_geo[i] - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo


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
        payoffs[i] = max(K - S_avg[i], 0)
        payoffs_geo[i] = max(K - S_avg_geo[i], 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price, option_price_geo


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


def phi(t,x, T, S):
    I = 0
    for i in range(t,N):
        I += S[i]*dt

    return np.mean(np.maximum((S[T]- S[t])/T - x,0))


def EDP_2Dim_fixe(S0, r, K, T, N, sigma, Nmc):
    t=0
    dt = T / N
    S = np.zeros(N+1)
    rho = 1 / T
    epsilon = np.zeros(N+1)
    epsilon[0] = S0
    S[0] = S0
    ph = np.zeros(Nmc+1)
    for i in range(Nmc):
        for j in range(0, N):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j + 1] = S[j] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dW)
            epsilon[j + 1] = epsilon[j] * (1 - sigma * dW - r * dt + sigma ** 2 * dt) - rho * dt
        ph[i] = phi(t,np.mean(epsilon),T,S)

    return np.exp(-r * T) * S0 * np.mean(ph)


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
    r = 0.1
    sigma = 0.5
    T = 1
    N = 100
    Nmc = 100
    S = np.linspace(S0, 20, 20)
    h = 0.000001

    calls = np.zeros(len(S))
    puts = np.zeros(len(S))
    calls_geo = np.zeros(len(S))
    puts_geo = np.zeros(len(S))
    delt = np.zeros(len(S))
    edp = np.zeros(len(S))

    for i in range(len(S)):
        calls[i] , calls_geo[i] = asian_option_price_call(S[i], r, K, T, N, sigma, Nmc)
        # puts[i] , puts_geo[i] = asian_option_price_put(S[i], r, K, T, N, sigma, Nmc)
        # delt[i] = delta(S[i], r, K, T, sigma, N, Nmc, h)
        # print("Asian call option price:", calls[i])
        # print("Asian put option price:", puts[i])
        edp[i] = EDP_2Dim_fixe(S0, r, K, T, N, sigma, Nmc)
    plt.plot(S, calls, label='Asian Call arithmétique')
    plt.plot(S, edp, label='edp')
    # plt.plot(S, puts, label='Asian Put arithmétique')
    # plt.plot(S, calls_geo, label='Asian Call géométrique')
    # plt.plot(S, puts_geo, label='Asian Put géométrique')
    plt.xlabel('$S_0$')
    plt.ylabel('Price Value')
    plt.legend()
    # plt.savefig('Graph\AsianOptionPrice.png')
    plt.show()
    #
    # plt.plot(S, delt, label='delta call')
    # plt.xlabel('$S_0$')
    # plt.ylabel('Price Value')
    # plt.legend()
    # plt.savefig('Graph\delta.png')
    # plt.show()

