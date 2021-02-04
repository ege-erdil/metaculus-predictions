import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

def sample_path(prob, steps):
    t = norm.isf(prob, loc=0, scale=np.sqrt(steps))
    path = [0]
    prob_path = [prob]

    for k in range(steps):
        path.append(path[-1] + float(norm.rvs(loc=0, scale=1, size=1)))
        if k < steps-1:
            prob_path.append(norm.sf(t, loc=path[-1], scale=np.sqrt(steps - k - 1)))
        else:
            prob_path.append(0 if path[-1] < t else 1)

    return prob_path

def sample_path_u(prob, steps):
    t = norm.isf(prob/2, loc=0, scale=np.sqrt(steps))
    path = [0]
    prob_path = [prob]

    for k in range(steps):
        if path[-1] < t:
            path.append(path[-1] + float(norm.rvs(loc=0, scale=1, size=1)))
            if k < steps-1:
                prob_path.append(float(np.minimum(2 * norm.sf(t, loc=path[-1], scale=np.sqrt(steps - k - 1)), 1)))
            else:
                prob_path.append(0 if path[-1] < t else 1)
        else:
            prob_path.append(1)

    return prob_path

i = 0
while True:
    path = sample_path_u(0.05, 500)
    if path[-1] == 1:
        break

# print(i)

# path = sample_path(0.5, 250)

eps = np.array([path[i+1] - path[i] for i in range(len(path)-1)])
# print(np.corrcoef(eps[1:], eps[:-1]))

mod = sm.OLS(eps[1:], sm.add_constant(eps[:-1]))
res = mod.fit()
print(res.summary())

fig, axes = plt.subplots(2, 1, sharex=False)
axes[0].plot(path); axes[0].set_title('Original Series')
plot_acf(eps, ax=axes[1])
plt.show()
