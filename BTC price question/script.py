import numpy as np
import pandas as pd

vol = 0.045
num_of_days = 4 * 365 - 33
num_of_tries = 1000
num_of_successes = 0

samples = pd.read_csv("bootstrap.csv")

orig_val = list(samples["Values"])[-365:]
inv_val = [np.log(2 - np.exp(x)) for x in orig_val]

complete_list = orig_val + inv_val
length_samples = len(complete_list)

for x in range(num_of_tries):
    init_price = np.log(36169)
    for k in range(num_of_days):
        # init_price += np.random.normal(loc = 0, scale = vol) - (vol**2)/2
        init_price += complete_list[np.random.randint(0, length_samples)]
        if init_price > np.log(100000):
            num_of_successes += 1
            break

print(num_of_successes)
