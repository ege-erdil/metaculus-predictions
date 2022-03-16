import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

cpi_first_a = [0.027338744, -0.018295218, 0.05802626, 0.059647718, 0.009066868, 0.005990266, -0.003721623, 0.003735525, 0.028284332, 0.030401737, 0.017562346, 0.015188126, 0.013600816, 0.006709158, 0.012329224, 0.016458196, 0.011981865, 0.0192, 0.033594976, 0.032806804, 0.047058824, 0.058988764, 0.055702918, 0.032663317, 0.03406326, 0.089411765, 0.120950324, 0.071290944, 0.050359712, 0.066780822, 0.08988764, 0.132547865, 0.123537061, 0.08912037, 0.038257173, 0.037871034, 0.040433925, 0.037914692, 0.011872146, 0.0433213, 0.044117647, 0.046396023, 0.062549485, 0.029806259, 0.029667149, 0.028109628, 0.025974026, 0.025316456, 0.033788174, 0.016970459, 0.016069221, 0.02676399, 0.03436019, 0.016036655, 0.024802706, 0.020352035, 0.033423181, 0.033385498, 0.025239778, 0.041088134, -0.00022228, 0.028141231, 0.01437793, 0.030620668, 0.01759505, 0.015128384, 0.006531214, 0.006387248, 0.020507989, 0.021014932, 0.019201892, 0.022614488]
cpi_second_a = [-0.045633962, 0.076321478, 0.001621458, -0.05058085, -0.003076602, -0.009711888, 0.007457147, 0.024548807, 0.002117405, -0.012839391, -0.002374221, -0.00158731, -0.006891658, 0.005620066, 0.004128973, -0.004476331, 0.007218135, 0.014394976, -0.000788172, 0.014252019, 0.011929941, -0.003285846, -0.023039601, 0.001399944, 0.055348504, 0.031538559, -0.04965938, -0.020931232, 0.01642111, 0.023106819, 0.042660224, -0.009010803, -0.034416691, -0.050863197, -0.000386139, 0.002562891, -0.002519233, -0.026042546, 0.031449154, 0.000796347, 0.002278376, 0.016153462, -0.032743226, -0.00013911, -0.001557522, -0.002135602, -0.00065757, 0.008471718, -0.016817715, -0.000901238, 0.010694769, 0.007596199, -0.018323534, 0.008766051, -0.004450671, 0.013071145, -3.76824E-05, -0.00814572, 0.015848356, -0.041310414, 0.028363511, -0.013763301, 0.016242738, -0.013025619, -0.002466666, -0.00859717, -0.000143966, 0.014120742, 0.000506943, -0.001813039, 0.003412596, -0.00961309]
unrate_a = [4.0, 6.6, 4.3, 3.1, 2.7, 4.5, 5.0, 4.2, 4.2, 5.2, 6.2, 5.3, 6.6, 6.0, 5.5, 5.5, 5.0, 4.0, 3.8, 3.8, 3.4, 3.5, 6.1, 6.0, 5.2, 4.9, 7.2, 8.2, 7.8, 6.4, 6.0, 6.0, 7.2, 8.5, 10.8, 8.3, 7.3, 7.0, 6.6, 5.7, 5.3, 5.4, 6.3, 7.3, 7.4, 6.5, 5.5, 5.6, 5.4, 4.7, 4.4, 4.0, 3.9, 5.7, 6.0, 5.7, 5.4, 4.9, 4.4, 5.0, 7.3, 9.9, 9.3, 8.5, 7.9, 6.7, 5.6, 5.0, 4.7, 4.1, 3.9, 3.6]

cpi_first_a = 100 * np.array(cpi_first_a)
cpi_second_a = 100 * np.array(cpi_second_a)
unrate_a = np.array(unrate_a)

N = 4000
L = []

for k in range(N):
    indices = np.random.choice(range(len(cpi_first_a) - 1), size=10)
    #unrate_indices = np.random.choice(range(len(cpi_first_a) - 1), size=10)

    cpi_first = [cpi_first_a[i+1] for i in indices]
    cpi_first_lag = [cpi_first_a[i] for i in indices]
    unrate = [unrate_a[i+1] for i in indices]

    #unrate = [np.random.rand() for i in unrate_indices]
    
    m1 = sm.OLS(endog=cpi_first, exog=sm.add_constant(np.transpose(np.stack((cpi_first_lag, unrate)))))
    r1 = m1.fit()

    m2 = sm.OLS(endog=cpi_first, exog=sm.add_constant(cpi_first_lag))
    r2 = m2.fit()

    L.append(r2.rsquared_adj - r1.rsquared_adj)
    #L.append(r2.bic - r1.bic)

print(np.mean(L), np.std(L))
print(len([x for x in L if x < 0])/N)