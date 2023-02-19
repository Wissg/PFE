import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import statistics

def asian_option_price_call(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N+1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = max(S_avg[i] - K, 0)
        payoffs_geo[i] = max(S_avg_geo[i] - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price,option_price_geo

def asian_option_price_put(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N+1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)

    for i in range(Nmc):
        S[0] = S0
        for j in range(1, N + 1):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[j] = S[j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        S_avg[i] = np.mean(S)
        S_avg_geo[i] = statistics.geometric_mean(S)
        payoffs[i] = max(K - S_avg[i], 0)
        payoffs_geo[i] = max(K - S_avg_geo[i], 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    option_price_geo = np.exp(-r * T) * np.mean(payoffs_geo)

    return option_price,option_price_geo

if __name__ == '__main__':

    K = 50
    p = 5

    esp = np.linspace(0,100,1000)
    x = np.zeros(len(esp))
    x1 = np.zeros(len(esp))
    for i in range(len(esp)):
        x[i] = np.maximum(esp[i]-K,0) - p
        x1[i] = np.maximum(K - esp[i], 0) - p


    plt.plot(esp, x,label="Payoff Call")
    plt.plot(esp, x1,label="Payoff Put")

    plt.axhline(y=0, color='black')
    plt.axvline(x=K, color='r', linestyle='--',label='K = '+str(K))
    plt.xlabel('Esperance[S]')
    plt.ylabel('payoff')
    plt.title("Payoff (K = "+str(K)+", prime = "+str(p)+")")
    plt.legend()
    plt.savefig('Graph\payoff.png')
    plt.show()

    S0 = 1
    K = 10
    r = 0.1
    sigma = 0.5
    T = 1
    N = 100
    Nmc = 1000
    S = np.linspace(S0, 20, 20)

    calls = np.zeros(len(S))
    puts = np.zeros(len(S))
    calls_geo = np.zeros(len(S))
    puts_geo = np.zeros(len(S))

    for i in range(len(S)):
        calls[i] , calls_geo[i] = asian_option_price_call(S[i], r, K, T, N, sigma, Nmc)
        puts[i] , puts_geo[i] = asian_option_price_put(S[i], r, K, T, N, sigma, Nmc)
        # print("Asian call option price:", calls[i])
        # print("Asian put option price:", puts[i])

    plt.plot(S, calls, label='Asian Call arithmétique')
    plt.plot(S, puts, label='Asian Put arithmétique')
    plt.plot(S, calls_geo, label='Asian Call géométrique')
    plt.plot(S, puts_geo, label='Asian Put géométrique')
    plt.xlabel('$S_0$')
    plt.ylabel('Price Value')
    plt.legend()
    plt.savefig('Graph\AsianOptionPrice.png')
    plt.show()
