import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
v = list(df["Barrels"])
v.reverse()
v = v[-52*5:]

y = np.array(v)
# y = y[1:] - y[:-1]

model = SARIMAX(y, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
result = model.fit()

print(result.summary())

r = result.get_forecast(52)

# print(r.predicted_mean)

c = r.conf_int(alpha = .5)
w = r.predicted_mean

c_lower = [x[0] for x in c]
c_upper = [x[1] for x in c]

q = []

for _ in range(1000):
    s = result.simulate(nsimulations=3, anchor="end")
    # print(v[-1], s, (v[-1] + s[0] + s[1] + s[2])/4)
    q.append((v[-1] + s[0] + s[1] + s[2])/4)

print(np.percentile(q, 25))
print(np.percentile(q, 50))
print(np.percentile(q, 75))

plt.plot(w)
plt.fill_between(range(52), c_lower, c_upper, alpha=.1)
plt.show()
