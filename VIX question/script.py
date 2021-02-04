import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import random
from logistic_fit import estimate

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

f = pd.read_csv("VIXCLS.csv")
vix = list(f["VIXCLS"])
errors = []

# for k in range(3):
    # for j in range(3):
        # model = ARIMA(vix, order = (k, 0, j))
        # model_fit = model.fit()

        # print(k, j, model_fit.aic)

model = ARIMA(vix, order = (2, 0, 0))
model_fit = model.fit()

print(model_fit.summary())
params = model_fit.params
print(params)

result = adfuller(vix)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

for k in range(2, len(vix)):
    pr = params[0] + params[1] * (vix[k-1] - params[0]) + params[2] * (vix[k-2] - params[0])
    errors.append(vix[k] - pr)

s = []
for k in range(1000):
    v1 = vix[-1]
    v2 = vix[-2]
    # print (v1, v2)
    i = 0
    for _ in range(2000):
        i += 1
        t = v1
        v1 = params[0] + params[1] * (v1 - params[0]) + params[2] * (v2 - params[0]) + random.choice(errors)
        v2 = t
        if v1 > 40:
            break
    s.append(i)

# print(np.average(s), np.std(s))
# print(len([x for x in s if x < 25]))
# print(len([x for x in s if x < 50]))
# print(len([x for x in s if x < 75]))
# print(len([x for x in s if x < 110]))

print(np.percentile(s, 5))
print(np.percentile(s, 25))
print(np.percentile(s, 50))
print(np.percentile(s, 75))
print(np.percentile(s, 95))

# print(estimate(s, 2, 0.01, 50))

#print(np.average(errors))
#print(len([x for x in errors if abs(x) > 10]))

# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(vix); axes[0, 0].set_title('Original Series')
# plot_acf(np.array(errors), ax=axes[0, 1])
# plt.show()
    
