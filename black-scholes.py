import numpy as np
from scipy.stats import norm

def BS(spot, strike, vol, time, r = 0):
    d1 = (np.log(spot/strike) + (r + vol**2/2) * time)/(vol * np.sqrt(time))
    d2 = d1 - vol * np.sqrt(time)

    return norm.cdf(d1) * spot - norm.cdf(d2) * strike * np.exp(-r * time)

print(BS(200, 320, 0.2, 14 * 24))
