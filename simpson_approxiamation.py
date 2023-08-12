import numpy as np
import matplotlib.pyplot as plt

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

    for i in range(Nmc):
        W = np.random.randn(1) * np.sqrt(dt)
        integral_approximation = simpson_rule(lambda t: geometric_brownian_motion(t, S0, r, sigma, W), 0, T,
                                              N)
        S_avg[i] = integral_approximation
        payoffs[i] = np.maximum(S_avg[i] - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price

if __name__ == '__main__':
    # Paramètres
    S0 = 100
    r = 0.2
    K = 100
    T = 1
    N = 100
    sigmas = np.linspace(0.1, 0.5, 5)  # Variations de la volatilité
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
    plt.savefig("Graph/simpsonHomer.png")
    plt.show()

    S_values = np.linspace(50, 150, 10)  # Variations du prix de l'actif sous-jacent
    sigmas = np.linspace(0.1, 0.5, 10)  # Variations de la volatilité
    Nmc = 1000  # Nombre d'itérations de Monte Carlo constant

    # Calcul des prix des options asiatiques pour chaque combinaison de S et sigma
    option_prices = np.zeros((len(S_values), len(sigmas)))

    for i, S in enumerate(S_values):
        for j, sigma in enumerate(sigmas):
            option_prices[i, j] = asian_option_price_call(S, r, K, T, N, sigma, Nmc)

    # Création du graphique 3D
    sigma_grid, S_grid = np.meshgrid(sigmas, S_values)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sigma_grid, S_grid, option_prices, cmap='viridis')
    ax.set_xlabel('Volatilité')
    ax.set_ylabel('Prix de l\'actif sous-jacent (S)')
    ax.set_zlabel('Prix de l\'option')
    ax.set_title('Prix de l\'option asiatique en fonction de S et de la volatilité')
    plt.savefig("Graph/simpsonMarge.png")
    plt.show()