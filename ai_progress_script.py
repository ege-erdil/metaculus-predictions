import numpy as np
from scipy.stats import expon, gamma

alpha = 2
beta = np.log(84.15)

N = 30000
L = []

for _ in range(N):
    lambda_val = gamma.rvs(a=alpha, scale=1/beta)
    L.append(np.exp(expon.rvs(scale=1/lambda_val)) + 1)

X = [t-10 for t in L if t > 10]

for k in range(1, 16):
    print("%dth percentile: %.2f" % (5*k, np.percentile(X, 5*k)))

print("\n")
alpha = 2
beta = 16.4

N = 30000
L = []

for _ in range(N):
    lambda_val = gamma.rvs(a=alpha, scale=1/beta)
    L.append(expon.rvs(scale=1/lambda_val))

X = [t-10 for t in L if t > 10]

for k in range(1, 20):
    print("%dth percentile: %.2f" % (5*k, np.percentile(X, 5*k)))
