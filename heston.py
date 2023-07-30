import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mu_v = np.linspace(0.01, 0.1, 25)
sigma2_v = np.linspace(0.01, 0.9, 25)
simulations = 500
s0 = 10
K = 10
result = np.zeros([len(mu_v), len(sigma2_v)])
random_vector = np.zeros(simulations * 3)

for n in range(len(random_vector)):
    random_vector[n] = random.gauss(mu=0, sigma=1)

for x in range(len(mu_v)):
    drift = mu_v[x]
    for y in range(len(sigma2_v)):
        vol = sigma2_v[y]
        sample_payoff = np.zeros(simulations)

        for n in range(simulations):
            s1 = s0 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n])
            s2 = s1 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n + 1])
            s3 = s2 * np.exp((drift - vol / 2) * 1 / 3 + (vol * 1 / 3) ** 0.5 * random_vector[n + 2])
            P = np.mean(np.array([s1, s2, s3]))
            sample_payoff[n] = max(P - K, 0)

        result[x, y] = np.mean(sample_payoff) * np.exp(-drift)


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


mu = mu_v
sigma2 = sigma2_v
MU, SIGMA2 = np.meshgrid(mu, sigma2)
zs = np.array([f(mu, sigma2) for mu, sigma2 in zip(np.ravel(MU), np.ravel(SIGMA2))])
Z = zs.reshape(MU.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MU, SIGMA2, Z, cmap=cm.coolwarm, antialiased=False)
cbar = plt.colorbar(surf)
ax.set_xlabel('Interest rate ($\mu$)')
ax.set_ylabel('Volatility ($\sigma^2$)')
ax.set_zlabel('Price of option')
plt.show()
