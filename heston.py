# Import necessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define ranges for drift (mu) and volatility squared (sigma2)
mu_v = np.linspace(0.01, 0.1, 25)
sigma2_v = np.linspace(0.01, 0.9, 25)

# Set the number of Monte Carlo simulations and initial values for the option
simulations = 500
s0 = 10
K = 10

# Create an empty array to store the results of option pricing
result = np.zeros([len(mu_v), len(sigma2_v)])

# Create an empty array to store random Gaussian numbers for simulations
random_vector = np.zeros(simulations * 3)

# Generate random Gaussian numbers and store them in random_vector
for n in range(len(random_vector)):
    random_vector[n] = random.gauss(mu=0, sigma=1)

# Loop over each combination of drift (mu) and volatility squared (sigma2)
for x in range(len(mu_v)):
    drift = mu_v[x]
    for y in range(len(sigma2_v)):
        vol = sigma2_v[y]

        # Create an array to store the payoffs for each simulation
        sample_payoff = np.zeros(simulations)

        # Perform Monte Carlo simulation for option pricing
        for n in range(simulations):
            s1 = s0 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n])
            s2 = s1 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n + 1])
            s3 = s2 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n + 2])
            P = np.mean(np.array([s1, s2, s3]))
            sample_payoff[n] = max(P - K, 0)

        # Calculate the average payoff and discount it using the risk-free rate
        result[x, y] = np.mean(sample_payoff) * np.exp(-drift)


# Define a function to find the option price for specific mu and sigma2 values
def f(mu, sigma2):
    for a in range(len(mu_v)):
        if mu_v[a] == mu:
            x = a
            break
    for b in range(len(sigma2_v)):
        if sigma2_v[b] == sigma2:
            y = b
            break
    return result[x, y]


# Prepare data for 3D plotting
mu = mu_v
sigma2 = sigma2_v
MU, SIGMA2 = np.meshgrid(mu, sigma2)
zs = np.array([f(mu, sigma2) for mu, sigma2 in zip(np.ravel(MU), np.ravel(SIGMA2))])
Z = zs.reshape(MU.shape)

# Create a 3D plot to visualize the option prices as a function of mu and sigma2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MU, SIGMA2, Z, cmap=cm.coolwarm, antialiased=False)
cbar = plt.colorbar(surf)
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Price of option')
plt.savefig('Graph/heston.png')
plt.show()
