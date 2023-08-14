import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm  # Importing tqdm for the progress bar
import csv


# Function to perform Monte Carlo simulation for option pricing using Heston model
def monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0):
    dt = T / N

    s_paths = np.zeros((simulations, N + 1))
    v_paths = np.zeros((simulations, N + 1))
    s_paths[:, 0] = s0
    v_paths[:, 0] = v0

    for i in range(1, N + 1):
        dW_V = np.sqrt(dt) * np.random.randn(simulations)
        dW_W = rho * dW_V + np.sqrt(1 - rho ** 2) * np.sqrt(dt) * np.random.randn(simulations)
        v_paths[:, i] = v_paths[:, i - 1] + kappa * (theta - v_paths[:, i - 1]) * dt + sigma * np.sqrt(
            v_paths[:, i - 1] * dt) * dW_V
        s_paths[:, i] = s_paths[:, i - 1] * np.exp(
            (mu - v_paths[:, i - 1] / 2) * dt + np.sqrt(v_paths[:, i - 1] * dt) * dW_W)

    P = np.mean(s_paths, axis=1)
    sample_payoff = np.maximum(P - K, 0)
    discounted_payoff = sample_payoff * np.exp(-mu * T)

    return np.mean(discounted_payoff)

def calculate_vega(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.01):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_sigma_plus_h = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta,
                                                                 sigma + h, v0)
    vega = (option_price_at_sigma_plus_h - option_price_base) / h
    return vega


def calculate_gamma(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.01):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_plus_h = monte_carlo_simulation_heston(s0 + h, K, mu, T, simulations, N, rho, kappa, theta, sigma,
                                                             v0)
    option_price_at_s_minus_h = monte_carlo_simulation_heston(s0 - h, K, mu, T, simulations, N, rho, kappa, theta,
                                                              sigma, v0)
    gamma = (option_price_at_s_plus_h - 2 * option_price_base + option_price_at_s_minus_h) / h ** 2
    return gamma


def calculate_delta(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.01):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_plus_h = monte_carlo_simulation_heston(s0 + h, K, mu, T, simulations, N, rho, kappa, theta, sigma,
                                                             v0)
    delta = (option_price_at_s_plus_h - option_price_base) / h
    return delta

def calculate_greeks(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.1):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_plus_h = monte_carlo_simulation_heston(s0 + h, K, mu, T, simulations, N, rho, kappa, theta, sigma,
                                                             v0)
    delta = (option_price_at_s_plus_h - option_price_base) / h
    option_price_at_s_minus_h = monte_carlo_simulation_heston(s0 - h, K, mu, T, simulations, N, rho, kappa, theta,
                                                              sigma, v0)
    gamma = (option_price_at_s_plus_h - 2 * option_price_base + option_price_at_s_minus_h) / h ** 2
    option_price_at_sigma_plus_h = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta,
                                                                 sigma + h, v0)
    vega = (option_price_at_sigma_plus_h - option_price_base) / h
    return delta, gamma, vega

def calculate_option_combinations(monte_carlo_simulation_func, mu_v, rho_v, kappa_v, theta_v, sigma_v,
                                  s0_values, K_values, T_values, v0_values):
    tab_shape = (len(mu_v), len(rho_v), len(kappa_v), len(theta_v), len(sigma_v),
                 len(s0_values), len(K_values), len(T_values), len(v0_values))
    tab = np.zeros(tab_shape)

    total_combinations = np.prod(tab_shape)
    progress_bar = tqdm(total=total_combinations, desc="Calculating options")

    for i, mu in enumerate(mu_v):
        for j, rho in enumerate(rho_v):
            for k, kappa in enumerate(kappa_v):
                for l, theta in enumerate(theta_v):
                    for m, sigma in enumerate(sigma_v):
                        for n, s0 in enumerate(s0_values):
                            for o, K in enumerate(K_values):
                                for p, T in enumerate(T_values):
                                    for q, v0 in enumerate(v0_values):
                                        tab[i, j, k, l, m, n, o, p, q] = monte_carlo_simulation_func(s0, K, mu, T,
                                                                                                     simulations, N,
                                                                                                     rho,
                                                                                                     kappa, theta,
                                                                                                     sigma,
                                                                                                     v0)
                                        progress_bar.update(1)

    progress_bar.close()
    return tab


def save_to_csv(tab, mu_v, rho_v, kappa_v, theta_v, sigma_v,
                s0_values, K_values, T_values, v0_values):
    with open("data/option_prices.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mu", "rho", "kappa", "theta", "sigma", "s0", "K", "T", "v0", "option_price"])
        for i, mu in enumerate(mu_v):
            for j, rho in enumerate(rho_v):
                for k, kappa in enumerate(kappa_v):
                    for l, theta in enumerate(theta_v):
                        for m, sigma in enumerate(sigma_v):
                            for n, s0 in enumerate(s0_values):
                                for o, K in enumerate(K_values):
                                    for p, T in enumerate(T_values):
                                        for q, v0 in enumerate(v0_values):
                                            option_price = tab[i, j, k, l, m, n, o, p, q]
                                            writer.writerow([mu, rho, kappa, theta, sigma, s0, K, T, v0, option_price])

    print("Data saved to 'data/option_prices.csv'")


# Fonction pour tracer un graphique en 2D
def plot_2d_graph(x_values, y_values, xlabel, ylabel, title, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values)  # Add this line to plot the data
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Fonction pour tracer un graphique en 3D
def plot_3d_graph(x_values, y_values, z_values, xlabel, ylabel, zlabel, title, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_values, y_values, z_values, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


######
# Option price get data
######
# Define ranges for parameters
# mu_v = np.linspace(0.01, 0.1, 10)
# rho_v = np.linspace(-0.9, 0.9, 10)
# kappa_v = np.linspace(0.1, 2.0, 10)
# theta_v = np.linspace(0.02, 0.2, 10)
# sigma_v = np.linspace(0.1, 0.9, 10)
# s0_values = np.linspace(90, 110, 10)  # asset prices
# K_values = np.linspace(95, 105, 10)  # strike prices
# T_values = np.linspace(0.1, 1, 10)  # maturities
# v0_values = np.linspace(0.04, 0.06, 10)  # volatilities
#
# # Set other parameters and option details
# simulations = 1000
# N = 100
#
# # Calculate option prices for each combination of parameters
# tab = calculate_option_combinations(monte_carlo_simulation_heston, mu_v, rho_v, kappa_v, theta_v, sigma_v,
#                                     s0_values, K_values, T_values, v0_values)

#save into a csv
# save_to_csv(tab, mu_v, rho_v, kappa_v, theta_v, sigma_v,
#             s0_values, K_values, T_values, v0_values)
######
# Option price
######
#
# # Parameters
# s0 = 100  # Prix initial de l'actif sous-jacent
# K = 100  # Prix d'exercice de l'option
# mu_v = np.linspace(0.01, 0.1, 10)
# rho_v = np.linspace(-0.9, 0.9, 10)
# T = 1  # Maturité de l'option
# simulations = 10000  # Nombre de simulations Monte Carlo
# N = 100  # Nombre de pas de discrétisation
# kappa = 2  # Paramètre de vitesse de réversion
# theta = 0.02  # Paramètre de niveau de volatilité à long terme
# sigma = 0.9  # Paramètre de volatilité de volatilité
# v0 = 0.04  # Volatilité initiale
#
# result = np.zeros((len(mu_v), len(rho_v)))
# for i, mu in enumerate(mu_v):
#     for j, rho in enumerate(rho_v):
#         result[i, j] = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
# MU, RHO = np.meshgrid(mu_r, rho_v)
# plot_3d_graph(MU, RHO, result, 'Drift ($\mu$)', 'Correlation ($\\rho$)', 'Price of option',
#               f"Option Prices (Heston Model for Asian Options) sigma = {sigma} sigma = {kappa} theta = {theta}",
#               "Graph/option_price_heston.png")


# Parameters
# v0 = 0.04  # Volatilité initiale
# s0 = 100  # Initial price of the underlying asset
# K = 100  # Option strike price
# mu_r = np.linspace(0.01, 0.1, 10)  # Drift parameter for interest rate
# volatility = np.linspace(0.1, 0.9, 10)  # Volatility parameter
# T = 1  # Option maturity
# simulations = 10000  # Number of Monte Carlo simulations
# N = 100  # Number of discretization steps
# rho = 0.03
# kappa = 0.1  # Speed of mean reversion for interest rate
# theta = 0.03  # Long-term mean of interest rate
# sigma = 0.02  # Volatility of interest rate
# 
# result = np.zeros((len(mu_r), len(volatility)))
# for i, mu in enumerate(mu_r):
#     for j, vol in enumerate(volatility):
#         result[i, j] = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, vol, v0)
# MU, VOL = np.meshgrid(mu_r, volatility)
# plot_3d_graph(MU, VOL, result, 'Drift ($\mu_r$)', 'Volatility', 'Price of option',
#               f"Option Prices (Heston) rho = {rho} kappa = {kappa} theta_r = {theta}",
#               "Graph/option_price_heston_vol.png")

###########
# Greeks
###########
# Define the functions calculate_vega, calculate_gamma, calculate_delta, and plot_3d_graph

# Parameters
s0_values = np.linspace(80, 120, 40)  # Range of S values
T_values = np.linspace(0.1, 1.0, 10)  # Range of T values
v0 = 0.05
N = 100
simulations = 1000
kappa = 2  # Speed of mean reversion
theta = 0.02  # Long-term mean of volatility
sigma = 0.9  # Volatility of volatility
v0 = 0.04  # Initial volatility
mu = 0.01
rho = 0.03
K = 100

# Calculate Greeks for each combination of S and T values
vega_values = np.zeros((len(s0_values), len(T_values)))
gamma_values = np.zeros((len(s0_values), len(T_values)))
delta_values = np.zeros((len(s0_values), len(T_values)))
for i, s0 in tqdm(enumerate(s0_values), total=len(s0_values), desc="S0 Progress"):
    for j, T in enumerate(T_values):
        vega_values[i, j] = calculate_vega(s0, K, mu, T, simulations, N, rho, kappa, theta,
                                           sigma, v0)
        gamma_values[i, j] = calculate_gamma(s0, K, mu, T, simulations, N, rho, kappa, theta,
                                             sigma, v0)
        delta_values[i, j] = calculate_delta(s0, K, mu, T, simulations, N, rho, kappa, theta,
                                             sigma, v0)

T, S = np.meshgrid(T_values, s0_values)

# Plot 3D graphs for Vega, Gamma, and Delta
plot_3d_graph(S, T, vega_values, 'Asset Price ($S$)', 'Time to Maturity ($T$)', 'Vega',
              "Vega (Sensitivity to Volatility)", "Graph/vega_plot.png")
plot_3d_graph(S, T, gamma_values, 'Asset Price ($S$)', 'Time to Maturity ($T$)', 'Gamma',
              "Gamma (Second Derivative of Option Price)", "Graph/gamma_plot.png")
plot_3d_graph(S, T, delta_values, 'Asset Price ($S$)', 'Time to Maturity ($T$)', 'Delta',
              "Delta (Sensitivity to Asset Price)", "Graph/delta_plot.png")

# Plot 2D graphs for Vega, Gamma, and Delta varying with spot price (S)
T_index = 5  # Choose a specific index for time to maturity (T)
plot_2d_graph(s0_values, vega_values[:, T_index], 'Asset Price ($S$)', 'Vega',
              f"Vega at T = {T_values[T_index]}", "Graph/plot_2d_vega.png")

plot_2d_graph(s0_values, gamma_values[:, T_index], 'Asset Price ($S$)', 'Gamma',
              f"Gamma at T = {T_values[T_index]}", "Graph/plot_2d_gamma.png")

plot_2d_graph(s0_values, delta_values[:, T_index], 'Asset Price ($S$)', 'Delta',
              f"Delta at T = {T_values[T_index]}", "Graph/plot_2d_delta.png")


######
# Implied volatility smile plot
######
# parameter
mu_v = np.linspace(0.01, 0.1, 100)
sigma_v = np.linspace(0.01, 0.9, 100)
s0 = 100
K = 100
T = 1
simulations = 10000
N = 100
rho = -0.5  # Example value, replace with your desired value
kappa = 2
theta = 0.02
sigma = 0.9
v0 = 0.04

Z = np.zeros((len(mu_v), len(sigma_v)))

for i, sigma2 in enumerate(tqdm(sigma_v, desc="Sigma Progress")):
    for j, mu in enumerate(mu_v):
        Z[i, j] = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)

implied_volatility = np.sqrt(2 * np.pi) / (K * np.exp(-mu_v) * Z)
plot_3d_graph(mu_v, sigma_v, implied_volatility, 'Interest rate ($\mu$)', 'Volatility ($\sigma^2$)',
              'Implied Volatility', "Volatility Smile (Implied Volatility)", "Graph/heston_implied_vol.png")
