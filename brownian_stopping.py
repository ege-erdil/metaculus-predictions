import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def brownian_stopping_pdf(time, threshold, drift, vol=1):
    a = threshold/vol
    c = drift/vol
    t = time

    if t == 0:
        return 0
    else:
        return (a/np.sqrt(2*np.pi*t**3)) * np.exp(-(a-c*t)**2/(2*t))

def brownian_stopping_cdf(time, threshold, drift, vol=1):
    if time <= 0:
        return (0, 0)
    else:
        return quad(lambda x: brownian_stopping_pdf(x, threshold, drift, vol), 0, time)[0]

def brownian_stopping_quantile(q, threshold, drift, vol=1):
    r = minimize(lambda x: (q - brownian_stopping_cdf(x, threshold, drift, vol))**2, x0=(threshold/vol)**2, method="Nelder-Mead")
    return r.x[0]

print(brownian_stopping_cdf(136/365, np.log(10000) - np.log(47600), -1.3**2/2, 1.3) + brownian_stopping_cdf(136/365, np.log(10000) - np.log(47600), -1.1**2/2, 1.1))
#print(brownian_stopping_quantile(0.5, 0.22, -0.07 + 0.2**2/2, 0.2))
