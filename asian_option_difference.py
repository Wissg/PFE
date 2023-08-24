import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------
#------------------------------- Cas d'un Call --------------------------------
#------------------------------------------------------------------------------

def euler(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    delta_t = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 1, N + 1))

    # Set initial and boundary conditions
    F[:, 0] = 1 / (r * T) * (1 - np.exp(-r * (T - t))) - x_min*np.exp(-r*(T - t))
    F[:, N] = 0
    F[M, :] = np.maximum(-x, 0)

    for n in range(M, 0, -1):
        for i in range(1, N):
            F[n - 1, i] = F[n, i] - delta_t * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * delta_x) + \
                          sigma ** 2 * x[i] ** 2 * delta_t / (2 * delta_x ** 2) * (
                                      F[n, i + 1] - 2 * F[n, i] + F[n, i - 1])

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis', edgecolor='none')
    ax.view_init(elev=10, azim=1)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(t, x)')
    ax.set_title('Numerical Solution using Explicit Euler Method')
    #plt.savefig('Graph/euler_explicit.png')
    plt.show()
    
    return(F)


if __name__ == '__main__':
    x_max = 1
    x_min = 0
    S0 = 10
    r = 0.4
    sigma = 0.3
    T = 1
    K = 10
    N = 99
    M = 999
    F = euler(S0, r, sigma, T, K, N, M, x_max, x_min)
    
S0 = np.linspace(1,20,20)
V = np.zeros(len(S0))
delta_x = (x_max - x_min) / (N + 1)
for i in range(len(S0)):
    a = int(np.floor(K/S0[i]*1/delta_x) + 1)
    if a - 1 >= N + 1:
        V[i] = 0
    else:
        V[i] = S0[i] * euler(S0[i], r, sigma, T, K, N, M, x_max, x_min)[0, a]
        
plt.plot(S0, V)
plt.xlabel("Valeur initial de l'actif")
plt.ylabel("Prix de l'option asiatique (t=0)")
plt.show()

S = np.linspace(1,20,20)
A = np.linspace(1,20,20)
delta_t = T / (M + 1)
V = np.zeros(shape=(len(S),len(A)))
for k in range(len(S)):
    for j in range(len(A)):
        a = int(np.floor((K - A[j]/2)/S[k]*1/delta_x) + 1)
        b = int(np.floor(T/2*(1/delta_t)) + 1)
        if a - 1 >= N + 1:
            V[k, j] = 0
        else:
            V[k, j] = S[k] * euler(S[k], r, sigma, T, K, N, M, x_max, x_min)[b, a]
    
Ss, Aa = np.meshgrid(S, A)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Ss, Aa, V, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=200)
ax.set_xlabel('S')
ax.set_ylabel('A')
ax.set_zlabel('V')
ax.set_title('Numerical Solution using Explicit Euler Method')
plt.show()

#------------------------------------------------------------------------------
#------------------------------- Cas d'un Put ---------------------------------
#------------------------------------------------------------------------------

def euler_put(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    delta_t = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 1, N + 1))

    # Set initial and boundary conditions
    
    F[:, N] = 1 / (r * T) * (np.exp(-r * (T - t)) - 1) + x_max*np.exp(-r*(T - t))
    F[:, 0] = 0
    F[M, :] = np.maximum(x, 0)

    for n in range(M, 0, -1):
        for i in range(1, N):
            F[n - 1, i] = F[n, i] - delta_t * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * delta_x) + \
                          sigma ** 2 * x[i] ** 2 * delta_t / (2 * delta_x ** 2) * (
                                      F[n, i + 1] - 2 * F[n, i] + F[n, i - 1])

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis', edgecolor='none')
    ax.view_init(elev=30, azim=200)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(t, x)')
    ax.set_title('Numerical Solution using Explicit Euler Method')
    plt.show()
    
    return(F)

if __name__ == '__main__':
    x_max = 2
    x_min = 0
    S0 = 100
    r = 0.02
    sigma = 0.3
    T = 1
    K = 10
    N = 100
    M = 999
    F = euler_put(S0, r, sigma, T, K, N, M, x_max, x_min)
    
S0 = np.linspace(1,100,20)
V = np.zeros(len(S0))
delta_x = (x_max - x_min) / (N + 1)
for i in range(len(S0)):
    a = int(np.floor(K/(S0[i]*delta_x)) + 1)
    if a - 1 >= N + 1:
        V[i] = V[i + 1] + 1
    else:
        V[i] = S0[i] * euler_put(S0[i], r, sigma, T, K, N, M, x_max, x_min)[0, a]
        
plt.plot(S0, V)
plt.xlabel("Valeur initial de l'actif")
plt.ylabel("Prix de l'option asiatique (t=0)")
plt.show()

S = np.linspace(1,20,20)
A = np.linspace(1,20,20)
delta_t = T / (M + 1)
V = np.zeros(shape=(len(S),len(A)))
for k in range(len(S)):
    for j in range(len(A)):
        a = int(np.floor((K - A[j]/2)/S[k]*1/delta_x) + 1)
        b = int(np.floor(T/2*(1/delta_t)) + 1)
        if a - 1 >= N + 1:
                V[k, j] = V[k + 1, j] + 1
        else:
                V[k, j] = S[k] * euler_put(S[k], r, sigma, T, K, N, M, x_max, x_min)[b, a]
    
Ss, Aa = np.meshgrid(S, A)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Ss, Aa, V, cmap='viridis', edgecolor='none')
ax.view_init(elev=50, azim=30)
ax.set_xlabel('S')
ax.set_ylabel('A')
ax.set_zlabel('V')
ax.set_title('Numerical Solution using Explicit Euler Method')
plt.show()

#------------------------------------------------------------------------------
#------------------------------- Cas d'un Put flottant ------------------------
#------------------------------------------------------------------------------

def euler_flottant_put(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    delta_t = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 1, N + 1))

    # Set initial and boundary conditions
    F[:, 0] = 1 / (r * T) * (1 - np.exp(-r * (T - t))) - x_min*np.exp(-r*(T - t)) - 1
    F[:, N] = 0
    F[M, :] = np.maximum(-(x+1), 0)

    for n in range(M, 0, -1):
        for i in range(1, N):
            F[n - 1, i] = F[n, i] - delta_t * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * delta_x) + \
                          sigma ** 2 * x[i] ** 2 * delta_t / (2 * delta_x ** 2) * (
                                      F[n, i + 1] - 2 * F[n, i] + F[n, i - 1])

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis', edgecolor='none')
    ax.view_init(elev=10, azim=10)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(t, x)')
    ax.set_title('Numerical Solution using Explicit Euler Method')
    #plt.savefig('Graph/euler_explicit.png')
    plt.show()
    
    return(F)


if __name__ == '__main__':
    x_max = 2
    x_min = -2
    S0 = 10
    r = 0.4
    sigma = 0.3
    T = 1
    K = 10
    N = 99
    M = 999
    F = euler_flottant_put(S0, r, sigma, T, K, N, M, x_max, x_min)
    
S0 = np.linspace(1,20,20)
V = np.zeros(len(S0))
delta_x = (x_max - x_min) / (N + 1)
for i in range(len(S0)):
    a = int(np.floor(K/S0[i]*1/delta_x) + 1)
    if a - 1 >= N + 1:
        V[i] = 0
    else:
        V[i] = S0[i] * euler_flottant_put(S0[i], r, sigma, T, K, N, M, x_max, x_min)[0, a]
        
plt.plot(S0, V)
plt.xlabel("Valeur initial de l'actif")
plt.ylabel("Prix de l'option asiatique (t=0)")
plt.show()

S = np.linspace(1,20,20)
A = np.linspace(1,20,20)
delta_t = T / (M + 1)
V = np.zeros(shape=(len(S),len(A)))
for k in range(len(S)):
    for j in range(len(A)):
        a = int(np.floor((K - A[j]/2)/S[k]*1/delta_x) + 1)
        b = int(np.floor(T/2*(1/delta_t)) + 1)
        if a - 1 >= N + 1:
            V[k, j] = 0
        else:
            V[k, j] = S[k] * euler_flottant_put(S[k], r, sigma, T, K, N, M, x_max, x_min)[b, a]
    
Ss, Aa = np.meshgrid(S, A)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Ss, Aa, V, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=200)
ax.set_xlabel('S')
ax.set_ylabel('A')
ax.set_zlabel('V')
ax.set_title('Numerical Solution using Explicit Euler Method')
plt.show()

#------------------------------------------------------------------------------
#------------------------------- Cas d'un Call flottant -----------------------
#------------------------------------------------------------------------------

def euler_call_flottant(S0, r, sigma, T, K, N, M, x_max, x_min):
    x = np.linspace(x_min, x_max, N + 1)
    t = np.linspace(0, T, M + 1)
    delta_t = T / (M + 1)
    delta_x = (x_max - x_min) / (N + 1)

    F = np.zeros(shape=(M + 1, N + 1))

    # Set initial and boundary conditions
    F[:, N] = 1 / (r * T) * (np.exp(-r * (T - t)) - 1) + x_max*np.exp(-r*(T - t)) + 1
    F[:, 0] = 0
    F[M, :] = np.maximum(x + 1, 0)

    for n in range(M, 0, -1):
        for i in range(1, N):
            F[n - 1, i] = F[n, i] - delta_t * (1 / T + r * x[i]) * (F[n, i + 1] - F[n, i - 1]) / (2 * delta_x) + \
                          sigma ** 2 * x[i] ** 2 * delta_t / (2 * delta_x ** 2) * (
                                      F[n, i + 1] - 2 * F[n, i] + F[n, i - 1])

    # Create a meshgrid to plot the surface
    X, T = np.meshgrid(x, t)

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, F, cmap='viridis', edgecolor='none')
    ax.view_init(elev=30, azim=200)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(t, x)')
    ax.set_title('Numerical Solution using Explicit Euler Method')
    plt.show()
    
    return(F)

if __name__ == '__main__':
    x_max = 2
    x_min = -2
    S0 = 10
    r = 0.4
    sigma = 0.3
    T = 1
    K = 10
    N = 99
    M = 999
    F = euler_call_flottant(S0, r, sigma, T, K, N, M, x_max, x_min)
    
S0 = np.linspace(1,20,20)
V = np.zeros(len(S0))
delta_x = (x_max - x_min) / (N + 1)
for i in range(len(S0)):
    a = int(np.floor(K/S0[i]*1/delta_x) + 1)
    if a - 1 >= N + 1:
        V[i] = V[i + 1] + 1
    else:
        V[i] = S0[i] * euler_call_flottant(S0[i], r, sigma, T, K, N, M, x_max, x_min)[0, a]
        
plt.plot(S0, V)
plt.xlabel("Valeur initial de l'actif")
plt.ylabel("Prix de l'option asiatique (t=0)")
plt.show()

S = np.linspace(1,20,20)
A = np.linspace(1,20,20)
delta_t = T / (M + 1)
V = np.zeros(shape=(len(S),len(A)))
for k in range(len(S)):
    for j in range(len(A)):
        a = int(np.floor((K - A[j]/2)/S[k]*1/delta_x) + 1)
        b = int(np.floor(T/2*(1/delta_t)) + 1)
        if a - 1 >= N + 1:
            V[k, j] =  V[k + 1, j] + 1
        else:
            V[k, j] = S[k] * euler_call_flottant(S[k], r, sigma, T, K, N, M, x_max, x_min)[b, a]
    
Ss, Aa = np.meshgrid(S, A)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Ss, Aa, V, cmap='viridis', edgecolor='none')
ax.view_init(elev=10, azim=1)
ax.set_xlabel('S')
ax.set_ylabel('A')
ax.set_zlabel('V')
ax.set_title('Numerical Solution using Explicit Euler Method')
plt.show()