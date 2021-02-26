import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from logistic_fit import estimate

h = 10
start = 1946

f = pd.read_csv("data.csv")
g = pd.read_csv("BAA.csv")
prices_str = list(f["Price"])[:1928-start]
prices = [float(x.replace(",", "")) for x in prices_str]
prices = np.array(prices)
prices = np.flip(prices)

yields_str = list(f["Yield"])[:1928-start]
yields = [float(x.replace("%", "")) for x in yields_str]
yields = np.array(yields)
yields = np.flip(yields)

baa_yield = list(g["BAA"])
baa_yield = baa_yield[len(baa_yield) - len(prices):]
baa_yield = np.array(baa_yield)
baa_yield = np.reshape(baa_yield, (len(baa_yield), 1))
baa_yield = baa_yield[:-h]

aaa_yield = list(g["AAA"])
aaa_yield = aaa_yield[len(aaa_yield) - len(prices):]
aaa_yield = np.array(aaa_yield)
aaa_yield = np.reshape(aaa_yield, (len(aaa_yield), 1))
aaa_yield = aaa_yield[:-h]

pce = list(g["PCE"])
pce = pce[len(pce) - len(prices):]
pce = np.array(pce)

yields_fwd = [np.prod(1 + yields[i:i+h]/100) for i in range(len(yields) - h)]

yields = yields[:-h]
yields = np.reshape(yields, (len(yields), 1))

returns = 100 * ( ((prices[h:] * pce[h:] * yields_fwd)/(prices[:-h] * pce[:-h]))**(1/h) - 1)

pce_ex = 100 * ((pce[:-h]/pce[h:])**(1/h) - 1)
pce_ex = np.reshape(pce_ex, (len(pce_ex), 1))
ex = sm.add_constant(np.concatenate((baa_yield, aaa_yield, yields), axis=1))

model = sm.OLS(endog=returns, exog=ex)
res = model.fit()

print(res.summary())

p = res.get_prediction(exog=[1, 3.6025, 2.47, 1.58])
print(p.predicted_mean + np.percentile(res.resid, 5), p.predicted_mean + np.percentile(res.resid, 95))
print(p.predicted_mean)

q = res.get_prediction(exog=ex)
print(np.std(q.predicted_mean))

print(estimate(p.predicted_mean[0] + res.resid, num=1, tol=0.001))

plt.plot(range(start, 2021-h), q.predicted_mean, color="r")
plt.plot(range(start, 2021-h), returns, color="b")
plt.show()
