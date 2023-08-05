import numpy as np
import matplotlib.pyplot as plt

def calculate_Y(S0, r, sigma, T):
    drift = (r - 0.5 * sigma**2) * T
    volatility = sigma * np.sqrt(T)
    return np.random.normal(drift, volatility)

def calculate_Z(S0, K, r, sigma, T, Y):
    d = np.sqrt(3) * (np.log(S0/K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * (S0 * np.exp((r - 0.5 * sigma**2) * T + sigma**2 * T / 6 * np.random.normal(d + np.sqrt(3), size=len(Y))) - K * np.random.normal(d, size=len(Y)))

# Paramètres
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
N_values = [100, 500, 1000, 5000, 10000]  # Différentes valeurs de N

# Estimations de prix pour chaque valeur de N
option_prices = []
option_pricesY = []

for N in N_values:
    Y_values = np.array([calculate_Y(S0, r, sigma, T) for _ in range(N)])
    Z_values = np.array([calculate_Z(S0, K, r, sigma, T, Y_values) for _ in range(N)])
    E_Z = np.mean(Z_values)
    option_prices.append(E_Z)
    option_pricesY.append(np.mean(Y_values))

plt.figure()
plt.plot(N_values, option_prices)
# plt.plot(N_values, option_pricesY)

plt.title("Prix de l'option en fonction du nombre de simulations (N)")
plt.xlabel("Nombre de simulations (N)")
plt.ylabel("Prix de l'option")
plt.grid(True)
plt.show()