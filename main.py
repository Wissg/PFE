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
        payoffs[i] = np.maximum(S_avg[i] - K, 0)
        payoffs_geo[i] = np.maximum(S_avg_geo[i] - K, 0)

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
        payoffs[i] = np.maximum(K - S_avg[i], 0)
        payoffs_geo[i] = np.maximum(K - S_avg_geo[i], 0)

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

def Asian_solver(m,ns,nI,Smax,Imax,r,D,T,sigma,p,phi1,phi2):
    dx=Smax/N
    dt=T/m
    f=np.zeros(shape=(N+1,m+1))
    x=np.linspace(0,Smax,ns+1)
    t=np.linspace(0,T,m+1)
    for i in range(N+1):

        f[i,m]=np.maximum(-i*dx,0)

    i=np.arange(0,N)
    M=np.zeros(shape=(N+1,m+1))

    for n in range(0, M + 1):
        for i in range(1, N + 1):

    for k in range(N-1,m-1):
        for p in range()
        a[i]=-dt/4 * (-(i*r + 1/(T*delta_x) + sigma**2*i**2))
        b[i]=-dt/4 * (-(i*r + 1/(T*delta_x) - sigma**2*i**2))
        c[i]= 1 + dt*0.5*sigma**2*i**2
        M[i,i-1]=a[i]
        M[i,i]=b[i]
        M[i,i+1]=c[i]
        x[:,nI,k]=phi1(x,Imax,t[k])
        for j in range(nI-1,-1,-1):
            F[0]=p(0,I[j]/T)*np.exp(-r(T-t[k]))
            F[i]=x[i]*dt/dI*x[i,j+1,k]+x[i,j,k+1]
            F[ns]=phi2(Smax,I[j],t[k])
            x[:,j,k]= np.linalg.solve(M,F)
    return x

def Crank_Nicolson(Kmax, S0, r, Tmax, N, M, Beta1, Beta2, sigma):
    K = np.linspace(0, Kmax, N + 2)
    deltaK = Kmax / (N + 1)
    T = np.linspace(0, Tmax, M + 2)
    deltat = Tmax / (M + 1)

    V = np.zeros(shape=(M + 2, N + 2))
    C = np.zeros(shape=(M + 2, N + 2))
    C2 = np.zeros(shape=(M + 2, N + 2))
    A = np.zeros(N + 1)
    B = np.zeros(N + 1)
    D = np.zeros(N + 1)
    D2 = np.zeros(N + 1)

    for i in range(N + 2):
        V[0, i] = np.maximum(S0 - K[i], 0)

    for n in range(1, M + 2):
        V[n, 0] = S0
        V[n, N + 1] = 0

    for n in range(0, M + 1):
        for i in range(1, N + 1):
            A[i] = -deltat/4 * (-(i*r + 1/(T*delta_x) + sigma**2*i**2))
            B[i] = -deltat/4 * (-(i*r + 1/(T*delta_x) - sigma**2*i**2))
            D[i] = 1 + dt*0.5*sigma**2*i**2


    return V

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
    N = 1000
    Nmc = 100
    S = np.linspace(S0, 20, 20)
    h = 0.000001

    calls = np.zeros(len(S))
    puts = np.zeros(len(S))
    calls_geo = np.zeros(len(S))
    puts_geo = np.zeros(len(S))
    delt = np.zeros(len(S))
    edp = np.zeros(len(S))

    # for i in range(len(S)):
    #     calls[i], calls_geo[i] = asian_option_price_call(S[i], r, K, T, N, sigma, Nmc)
    #     # puts[i] , puts_geo[i] = asian_option_price_put(S[i], r, K, T, N, sigma, Nmc)
    #     # delt[i] = delta(S[i], r, K, T, sigma, N, Nmc, h)
    #     # print("Asian call option price:", calls[i])
    #     # print("Asian put option price:", puts[i])
    #     edp[i] = EDP_2Dim_fixe(S0, r, K, T, N, sigma, Nmc)
    # plt.plot(S, calls, label='Asian Call arithmétique')
    # plt.plot(S, edp, label='edp')
    # plt.plot(S, puts, label='Asian Put arithmétique')
    # plt.plot(S, calls_geo, label='Asian Call géométrique')
    # plt.plot(S, puts_geo, label='Asian Put géométrique')
    # plt.xlabel('$S_0$')
    # plt.ylabel('Price Value')
    # plt.legend()
    # plt.savefig('Graph\AsianOptionPrice.png')
    # plt.show()
    #
    # plt.plot(S, delt, label='delta call')
    # plt.xlabel('$S_0$')
    # plt.ylabel('Price Value')
    # plt.legend()
    # plt.savefig('Graph\delta.png')
    # plt.show()

    S0 = 10
    K = 1
    r = 0.1
    sigma = 0.5
    T = 1
    N = np.linspace(10, 100, 100)
    Nmc = 1000

    option_price = np.zeros(len(N))
    option_price_geo = np.zeros(len(N))
    option_price_red = np.zeros(len(N))
    option_price_red_geo = np.zeros(len(N))

    for i in range(len(N)):
        option_price[i], option_price_geo[i], option_price_red[i], option_price_red_geo[i] = Log_return_symetrique(Nmc,
                                                                                                                   T, int(N[i]),
                                                                                                                   K,
                                                                                                                   S0,
                                                                                                                   r,
                                                                                                                   sigma)

    # plt.plot(S, Valeur1, label="estime1")
    plt.plot(N, option_price, label="option_price")
    plt.plot(N, option_price_geo, label="estoption_price_geo")
    plt.plot(N, option_price_red, label="option_price_red")
    plt.plot(N, option_price_red_geo, label="option_price_red_geo")

    plt.legend()
    plt.show()
