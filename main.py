import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

if __name__ == '__main__':

    K = 50
    p = 5

    esp = np.linspace(0,100,1000)
    x = np.zeros(len(esp))
    x1 = np.zeros(len(esp))
    for i in range(len(esp)):
        x[i] = np.maximum(esp[i]-K,0) - p
        x1[i] = np.maximum(K - esp[i], 0) - p


    plt.plot(esp, x,label="Payoff Call")
    plt.plot(esp, x1,label="Payoff Put")

    plt.axhline(y=0, color='black')
    plt.axvline(x=K, color='r', linestyle='--',label='K = '+str(K))
    plt.xlabel('Esperance[S]')
    plt.ylabel('payoff')
    plt.title("Payoff")
    plt.legend()
    plt.savefig('Graph\payoff.png')



    plt.show()
