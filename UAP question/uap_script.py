import numpy as np
from scipy.stats import poisson, lognorm
import matplotlib.pyplot as plt
from logistic_fit import estimate

probs = [0.85, 0.03, 0.01, 0.2, 0.2, 0.05, 0.2, 0.02, 0.01]
alpha = 0.5

results = []

for _ in range(10**5):
    r = 0
    q = np.random.rand()
    for p in probs:
        if q < p**alpha and np.random.rand() < p**(1 - alpha):
            r += 1

    s = 0.5
    f = lognorm.rvs(s) * np.exp(-s**2/2)
    r += min(poisson.rvs(mu=14 * f * 7/14)/25, 1)
    results.append(r)

print(np.mean(results), np.std(results))

for k in range(5):
    r_list = [x for x in results if x >= k and x < k+1]
    print(k, len(r_list)/10**5, np.mean(r_list), np.percentile(r_list, 25), np.percentile(r_list, 75))

k_list = [k/100 for k in range(1000)]
plt.plot(k_list, [len([x for x in results if x < k])/10**5 for k in k_list])
plt.show()
    
