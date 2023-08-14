import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

# Define ranges for parameters
mu_v = np.linspace(0.01, 0.1, 10)
rho_v = np.linspace(-0.9, 0.9, 10)
kappa_v = np.linspace(0.1, 2.0, 10)
theta_v = np.linspace(0.02, 0.2, 10)
sigma_v = np.linspace(0.1, 0.9, 10)

# Set other parameters and option details
simulations = 100
s0 = 100
K = 100
T = 1
v0 = 0.05
N = 100

# Calculate option prices for each combination of parameters
tab = np.zeros((len(mu_v), len(rho_v), len(kappa_v), len(theta_v), len(sigma_v)))
result = np.zeros((len(mu_v), len(rho_v)))
for i, mu in enumerate(mu_v):
    for j, rho in enumerate(rho_v):
        for k, kappa in enumerate(kappa_v):
            for l, theta in enumerate(theta_v):
                for m, sigma in enumerate(sigma_v):
                    tab[i, j, k, l, m] = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
                    result[i, j] = tab[i, j, k, l, m]

# Prepare data for 3D plotting
MU, RHO = np.meshgrid(mu_v, rho_v)

# Create a 3D plot to visualize the option prices as a function of parameters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(MU, RHO, result, cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel('Drift ($\mu$)')
ax.set_ylabel('Correlation ($\\rho$)')
ax.set_zlabel('Price of option')
plt.title("Option Prices (Heston Model for Asian Options) sigma = "+str(sigma)+" sigma = "+str(kappa)+" theta = "+str(theta))
plt.show()