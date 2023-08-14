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

def calculate_vega(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.1):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_sigma_plus_h = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma + h, v0)
    vega = (option_price_at_sigma_plus_h - option_price_base) / h
    return vega

def calculate_gamma(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.1):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_plus_h = monte_carlo_simulation_heston(s0 + h, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_minus_h = monte_carlo_simulation_heston(s0 - h, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    gamma = (option_price_at_s_plus_h - 2 * option_price_base + option_price_at_s_minus_h) / h ** 2
    return gamma

def calculate_delta(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0, h=0.1):
    option_price_base = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    option_price_at_s_plus_h = monte_carlo_simulation_heston(s0 + h, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
    delta = (option_price_at_s_plus_h - option_price_base) / h
    return delta

# Define ranges for parameters
mu_v = np.linspace(0.01, 0.1, 10)
rho_v = np.linspace(-0.9, 0.9, 10)
kappa_v = np.linspace(0.1, 2.0, 10)
theta_v = np.linspace(0.02, 0.2, 10)
sigma_v = np.linspace(0.1, 0.9, 10)

# Set other parameters and option details
simulations = 1000
s0 = 100
K = 100
T = 1
v0 = 0.05
N = 100
######
# option graph
######
# # Calculate option prices for each combination of parameters
# tab = np.zeros((len(mu_v), len(rho_v), len(kappa_v), len(theta_v), len(sigma_v)))
# result = np.zeros((len(mu_v), len(rho_v)))
# for i, mu in enumerate(mu_v):
#     for j, rho in enumerate(rho_v):
#         for k, kappa in enumerate(kappa_v):
#             for l, theta in enumerate(theta_v):
#                 for m, sigma in enumerate(sigma_v):
#                     tab[i, j, k, l, m] = monte_carlo_simulation_heston(s0, K, mu, T, simulations, N, rho, kappa, theta, sigma, v0)
#                     result[i, j] = tab[i, j, k, l, m]
#
# # Prepare data for 3D plotting
# MU, RHO = np.meshgrid(mu_v, rho_v)
#
# # Create a 3D plot to visualize the option prices as a function of parameters
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(MU, RHO, result, cmap=cm.coolwarm, antialiased=False)
# ax.set_xlabel('Drift ($\mu$)')
# ax.set_ylabel('Correlation ($\\rho$)')
# ax.set_zlabel('Price of option')
# plt.title("Option Prices (Heston Model for Asian Options) sigma = "+str(sigma)+" sigma = "+str(kappa)+" theta = "+str(theta))
# plt.show()
############
# grecs
###########

s0_values = np.linspace(80, 120, 100)  # Range of S values
T_values = np.linspace(0.1, 1.0, 100)   # Range of T values
v0 = 0.05
N = 1000

# Calculate Vega for each combination of S and T values
vega_values = np.zeros((len(s0_values), len(T_values)))
for i, s0 in enumerate(s0_values):
    for j, T in enumerate(T_values):
        vega_values[i, j] = calculate_vega(s0, K, mu_v[-1], T, simulations, N, rho_v[-1], kappa_v[-1], theta_v[-1], sigma_v[-1], v0)

# Calculate Gamma for each combination of S and T values
gamma_values = np.zeros((len(s0_values), len(T_values)))
for i, s0 in enumerate(s0_values):
    for j, T in enumerate(T_values):
        gamma_values[i, j] = calculate_gamma(s0, K, mu_v[-1], T, simulations, N, rho_v[-1], kappa_v[-1], theta_v[-1], sigma_v[-1], v0)

# Calculate Delta for each combination of S and T values
delta_values = np.zeros((len(s0_values), len(T_values)))
for i, s0 in enumerate(s0_values):
    for j, T in enumerate(T_values):
        delta_values[i, j] = calculate_delta(s0, K, mu_v[-1], T, simulations, N, rho_v[-1], kappa_v[-1], theta_v[-1], sigma_v[-1], v0)

# Prepare data for 3D plotting
S, T = np.meshgrid(s0_values, T_values)

# Create 3D plots for Vega, Gamma, and Delta
fig = plt.figure(figsize=(18, 6))

# Vega plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(S, T, vega_values, cmap=cm.coolwarm)
ax1.set_xlabel('Asset Price ($S$)')
ax1.set_ylabel('Time to Maturity ($T$)')
ax1.set_zlabel('Vega')
ax1.set_title("Vega (Sensitivity to Volatility)")

# Gamma plot
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(S, T, gamma_values, cmap=cm.coolwarm)
ax2.set_xlabel('Asset Price ($S$)')
ax2.set_ylabel('Time to Maturity ($T$)')
ax2.set_zlabel('Gamma')
ax2.set_title("Gamma (Second Derivative of Option Price)")

# Delta plot
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(S, T, delta_values, cmap=cm.coolwarm)
ax3.set_xlabel('Asset Price ($S$)')
ax3.set_ylabel('Time to Maturity ($T$)')
ax3.set_zlabel('Delta')
ax3.set_title("Delta (Sensitivity to Asset Price)")
plt.show()

# Vega plot
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(S, T, vega_values, cmap=cm.coolwarm)
ax1.set_xlabel('Asset Price ($S$)')
ax1.set_ylabel('Time to Maturity ($T$)')
ax1.set_zlabel('Vega')
ax1.set_title("Vega (Sensitivity to Volatility)")
plt.savefig("Graph/vega_plot.png")
plt.show()

# Gamma plot
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(S, T, gamma_values, cmap=cm.coolwarm)
ax2.set_xlabel('Asset Price ($S$)')
ax2.set_ylabel('Time to Maturity ($T$)')
ax2.set_zlabel('Gamma')
ax2.set_title("Gamma (Second Derivative of Option Price)")
plt.savefig("Graph/gamma_plot.png")
plt.show()

# Delta plot
fig = plt.figure(figsize=(12, 8))
ax3 = fig.add_subplot(111, projection='3d')
ax3.plot_surface(S, T, delta_values, cmap=cm.coolwarm)
ax3.set_xlabel('Asset Price ($S$)')
ax3.set_ylabel('Time to Maturity ($T$)')
ax3.set_zlabel('Delta')
ax3.set_title("Delta (Sensitivity to Asset Price)")
plt.savefig("Graph/delta_plot.png")
plt.show()


# Create a 3D plot to visualize the volatility smile
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mu_v = np.linspace(0.01, 0.1, 100)
sigma2_v = np.linspace(0.01, 0.9, 100)
T = 1

Z = np.zeros(len(mu_v), len(sigma2_v))
for i, sigma2 in enumerate(sigma2_v):
    for j, mu in enumerate(mu_v):
        Z[i, j] = calculate_vega(s0, K, mu, T, simulations, N, rho_v[-1], kappa_v[-1], theta_v[-1], sigma2, v0)

implied_volatility = np.sqrt(2 * np.pi) / (K * np.exp(-mu_v) * Z)

# Create a meshgrid for mu_v and sigma2_v
MU_V, SIGMA2_V = np.meshgrid(mu_v, sigma2_v)

# Plot the volatility smile surface
surf = ax.plot_surface(MU_V, SIGMA2_V, implied_volatility, cmap=cm.coolwarm, antialiased=False)

# Add labels and title
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Implied Volatility')
plt.title("Volatility Smile (Implied Volatility)")
plt.savefig('Graph/heston_implied_vol.png')
plt.colorbar(surf)

plt.show()
