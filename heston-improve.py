import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Function to perform Monte Carlo simulation for option pricing
def monte_carlo_simulation(s0, K, drift, vol, simulations):
    random_vector = np.random.normal(0, 1, size=(simulations, 3))
    s_paths = s0 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector.cumsum(axis=1))
    P = np.mean(s_paths, axis=1)
    sample_payoff = np.maximum(P - K, 0)
    return np.mean(sample_payoff) * np.exp(-drift)


# Function to calculate option prices for specific mu and sigma2 values
def option_price(mu, sigma2, s0, K, simulations):
    drift = mu
    vol = np.sqrt(sigma2)
    return monte_carlo_simulation(s0, K, drift, vol, simulations)


# Function to calculate Vega (sensitivity to volatility) for specific mu and sigma2 values
def vega(mu, sigma2, s0, K, simulations, h=0.01):
    vega_values = np.zeros((len(mu), len(sigma2)))
    for i in range(len(mu)):
        for j in range(len(sigma2)):
            option_price_at_sigma2 = option_price(mu[i], sigma2[j], s0, K, simulations)
            option_price_at_sigma2_plus_h = option_price(mu[i], sigma2[j] + np.sqrt(h), s0, K, simulations)
            vega_values[i, j] = (option_price_at_sigma2_plus_h - option_price_at_sigma2) / h
    return vega_values


# Function to calculate Delta (sensitivity to asset price) for specific mu and sigma2 values
def delta(mu, sigma2, s0, K, simulations, h=0.01):
    delta_values = np.zeros((len(mu), len(sigma2)))
    for i in range(len(mu)):
        for j in range(len(sigma2)):
            option_price_at_s0 = option_price(mu[i], sigma2[j], s0, K, simulations)
            option_price_at_s0_plus_h = option_price(mu[i], sigma2[j], s0 + h, K, simulations)
            delta_values[i, j] = (option_price_at_s0_plus_h - option_price_at_s0) / h
    return delta_values



# Define ranges for drift (mu) and volatility squared (sigma2)
mu_v = np.linspace(0.01, 0.1, 100)  # Increase the number of points for smoother plot
sigma2_v = np.linspace(0.01, 0.9, 100)  # Increase the number of points for smoother plot

# Set the number of Monte Carlo simulations and initial values for the option
simulations = 5000  # Increase the number of simulations for smoother results
s0 = 10
K = 10

# Calculate option prices for each combination of mu and sigma2
result = np.zeros((len(mu_v), len(sigma2_v)))
for i, mu in enumerate(mu_v):
    for j, sigma2 in enumerate(sigma2_v):
        result[i, j] = option_price(mu, sigma2, s0, K, simulations)

# Prepare data for 3D plotting
MU, SIGMA2 = np.meshgrid(mu_v, sigma2_v)

# Create a 3D plot to visualize the option prices as a function of mu and sigma2
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(MU, SIGMA2, result, cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Price of option')
plt.title("Option Prices (Heston Model)")
plt.show()

# Calculate and plot Vega as a function of mu and sigma2
vega_values = vega(mu_v, sigma2_v, s0, K, simulations)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(MU, SIGMA2, vega_values, cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Vega')
plt.title("Vega (Sensitivity to Volatility)")

plt.savefig('Graph/heston_vega.png')
plt.show()

# Calculate and plot Delta as a function of mu and sigma2
delta_values = delta(mu_v, sigma2_v, s0, K, simulations)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(MU, SIGMA2, delta_values, cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Delta')
plt.title("Delta (Sensitivity to Asset Price)")
plt.savefig('Graph/heston_delta.png')
plt.show()
