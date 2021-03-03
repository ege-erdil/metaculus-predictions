import numpy as np
import statsmodels.api as sm
import pandas as pd

f = pd.read_csv("data.csv")
x = list(f["x"])
y = list(f["y"])

i_list = [h for h in range(len(x)) if x[h] > 80]
x = [x[i] for i in i_list]
y = [y[i] for i in i_list]

x = np.array(x)
y = np.array(y)

ppq = y/x

print(np.corrcoef(ppq, x)[0, 1])

m = sm.OLS(endog=ppq, exog=sm.add_constant(x))
r = m.fit()
print(r.summary())
