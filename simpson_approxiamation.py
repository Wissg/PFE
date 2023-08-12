import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import statistics
from scipy import integrate as intg

def geometric_brownian_motion(t, S0, mu, sigma, W):
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)


def simpson_rule(f, a, b, n):
    pas = (b - a) / n
    somme = (f(a) + f(b)) / 2 + 2 * f(a + pas / 2)  # On initialise la somme
    x = a + pas           # La somme commence à x_1
    for i in range(1, n): # On calcule la somme
        somme += f(x) + 2 * f(x + pas / 2)
        x += pas
    return somme * pas / 3   # On retourne cette somme fois le pas / 3

def asian_option_price_call(S0, r, K, T, N, sigma, Nmc):
    dt = T / N
    S_avg = np.zeros(Nmc)
    S_avg_geo = np.zeros(Nmc)
    S = np.zeros(N + 1)
    payoffs = np.zeros(Nmc)
    payoffs_geo = np.zeros(Nmc)
    n_segments = 100  # Nombre de segments pour la méthode de Simpson

    for i in range(Nmc):
        S[0] = S0
        np.random.seed(42)
        W = np.random.randn(N + 1) * np.sqrt(N / n_segments)
        integral_approximation = simpson_rule(lambda t: geometric_brownian_motion(t, S0, r, sigma, W), 0, N,
                                              n_segments)
        S_avg[i] = np.mean(integral_approximation)
        payoffs[i] = np.maximum(S_avg[i] - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price

if __name__ == '__main__':
    # Paramètres
    S0 = 50
    r = 0.4
    K = 20
    T = 1
    N = 100
    sigmas = np.linspace(0.4, 0.5, 50)  # Variations de la volatilité
    Nmc = 1000

    # Calcul des prix des options asiatiques pour chaque valeur de volatilité
    option_prices = [asian_option_price_call(S0, r, K, T, N, sigma, Nmc) for sigma in sigmas]

    # Création du graphique
    plt.figure(figsize=(8, 6))
    plt.plot(sigmas, option_prices, label="Prix de l'option asiatique")
    plt.xlabel("Volatilité")
    plt.ylabel("Prix de l'option")
    plt.title("Prix de l'option asiatique en fonction de la volatilité")
    plt.legend()
    plt.grid(True)
    plt.show()