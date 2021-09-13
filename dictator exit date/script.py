import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exit_date_mc(age, r=0.035, n=1000):
    f = pd.read_csv("actuarial.csv")
    res = []
    for k in range(n):
        i = age
        while True:
            if np.random.rand() > (1-f["DeathProbM"][i])*(1-r):
                break
            i += 1
        res.append(i)
    return res

def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def plot_ecdf(a):
    x, y = ecdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post')
    plt.grid(True)
    plt.show()
