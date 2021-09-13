import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

a = 0.037831
b = 0.733653

v = np.sqrt(0.4181)
R = 0.733087

s = 0.44

N = 10000
L = []

errors = [-0.269563454, 0.015118552, 0.304207122, -0.096845612, 0.359936835, -0.256747894, -0.218297226, 0.418270372, 0.102902891, 0.300859195, -0.106648726, 0.059771266, -0.239248461, -0.239760963, 0.263311697, 0.033286836, 0.109548997, 0.013885029, -0.486411151, 0.34784951, 0.326633711, 0.034663824, -0.223003193, 0.022784663, -0.215393475, -0.497444425, 0.047494485, 0.151750914, 0.207621283, 0.026374545, -0.408533115, -0.000112999, 0.189085978, -0.068253825, -1.049145532, 0.306926678, 0.016547789, -0.022925404, -0.071101739, -0.462867075, 0.035096347, -0.832362319, -0.644769077, -0.072887772, 0.059608783, 0.201881516, -0.664834142, 0.240683618, -0.025052172, -0.18588767, 0.072717986, 0.196723363, -0.032411759, -0.38396616, -0.330002213, 0.08150024, -0.234679792, 0.288650811, -0.110338433, -0.219165646, -0.243471809, 0.019322588, -0.038073765, -0.231428555, 0.023012023, -0.252706105, -0.288407742, 0.144639498, 0.117172132, 0.168051854, -0.343249554, -0.220432624, -0.142893275, -0.00192293, -0.049047423, 0.415438498, 0.023468241, -0.310094867, -0.124958833, -0.181301106, 0.186941565, 0.28872229, -0.107890296, -0.222031241, -0.194552587, 0.055091889, 0.151786633, -0.177660705, -0.011430959, -0.224149894, -0.076108337, 0.201067384, -0.153488825, 0.16775557, -0.349479954, -0.053739273, -0.181000612, -0.180680382, 0.053204301, 0.038304933, 0.309327717, -0.261728753, -0.141273015, -0.021338447, 0.23562296, 2.37834E-05, -0.538511264, 0.029660445, -0.136091571, -0.115496342, 0.111447585, 0.16042529, 0.047849841, -0.068874836, -0.510096112, 0.125720306, -0.001182182, 0.133826345, -0.052433596, -0.026178511, 0.037339321, -0.060312255, 0.255684893, -0.034221205, 0.133486317, -0.212985691, -0.228202768, 0.206610533, 0.477097797, 0.292468806, -0.01836604, -0.034706392, 0.07303552, -0.120767063, 0.078455065, 0.28652964, 0.085660903, 0.045666665, 0.015767947, -0.014465051, 0.258774971, 0.196729831, -0.148568248, 0.206361619, -0.073222645, -0.557209519, 0.756901193, 0.315531331, -0.223669117, 0.011795364, -0.244405079, 0.358243797, 0.228126828, 0.23270495, 0.03236534, -0.297804821, -0.298954215, 0.781984423, 0.047432578, -0.31205615, 0.316442739, 0.409320848, -0.740790183, 0.77882971, 0.160651881, 0.316636263, -0.701740929, 0.266844015, 0.232805204, -0.441095963, -0.017339264, 0.476299444, 0.800176417, -0.151277734, -0.748201449, 1.091905988, 0.039850796, 0.526192135, -0.423387392, 0.033374961, -0.22375957, -0.056342073, 0.840678095, 0.180721783, 0.182665702, -0.151796537, -0.014926167, 0.248343212, 0.214631487, 0.564721805, -0.542582384, -0.549507507, 0.03970721, 0.970716326, 0.290941557, -0.191051599, 0.445424647, -0.274551797, 0.05681373, 0.85760087, 0.34866331, 0.149199583, -0.325960721, 0.035241125, 0.413424795, 0.099306293, -0.265504213, 0.529739014, 0.715757803, -0.101329742, -0.030954474, -0.238414992, 0.60819039, 1.032128346, -0.058370778, -0.221431974, 0.226620937, -0.636605374, 1.1347244, 0.252348596, -0.334677686, -0.273309546, -1.077277061, 1.4872251, 1.389847777, -0.160866544, -1.123661353, 0.492608207, -0.276670762, -1.177810339, 0.025165731, 0.030811896, -0.098772094, -0.21140654, 1.087599295, 0.601002457, -0.442293667, -0.317022507, -0.009327225, -0.548116607, -1.086487076, -0.447667614, -0.629431319, 0.354096336, 0.174422736, -0.008762027, -0.625116071, -0.231795312, -1.223667583, 0.912066439, -0.923348224, -0.377068101, -1.078384285, -0.535587547, -0.854449544, -0.666501414, 0.155991187, 0.629733854, -0.62813618, 0.882593288, 1.176684945, -1.228394766, -1.703087803, -0.336714252, 0.017105291, -0.012014083, -0.069527843, -0.498099703, 1.098186078, -0.815490294, -0.125322103, -0.004637005, 0.00100531, 0.130426832, -0.921940577, -0.642447573, 0.423668345, 0.394621819, -0.226735698, -0.206369563, -0.349622568, -0.635705671, -0.056062616, -0.008162002, 0.035740563, -0.106036496, -0.809346176, -0.780254702, -0.560156681, 0.243601881, -0.089101342, 0.725606196, -0.307723592, -0.604668534, -0.235399524, 0.312115242, -0.358662182, 0.355078597, -0.660129518, -0.228276137, 0.423929012, 0.488535005, 0.529375832, 0.184570125, 0.054855517, -0.909481137, 0.049666977, 0.704707909, 0.905667559, 0.00874135, -0.220712869, -0.913947013, 1.474372321, 0.332562963, -0.222510468, 0.06176441, 0.446823551, 0.063666034, 0.062559227, 0.545138799, 0.093494837, -0.030652398, -0.348799146, -0.245771377, -0.407906002, 0.057392354, -0.080205494, 0.169282912, 0.429464397, 0.318258428, -0.18118794, 0.600301506, 0.583764084, 0.352017631, 0.220803865, -1.167248593, -0.192108018, 0.290816943, -0.424353985, 0.166872415, 0.556385067, 0.31314531, 0.275193311, 0.059266734, 0.352629738, 0.329508089, -0.448113862, -0.054172316, -0.432590314, -0.131366536, 0.10284648, 0.061946219, 0.700671318, -0.053133188, -0.135674686, 0.206716061, 0.476372104, -0.102456051, -0.582370246, 0.015970688, 0.368926808, -0.232366629, 0.390882622, 0.094887435, 0.284883072, -0.143243103, -0.464228766, 0.827985845, 0.000534151, 0.334896077, -0.718629608, -0.040574359, 0.331541009, -0.309213945, -0.04231371, 0.23169482, 0.226144384, 0.073384912, -0.191023203, -0.025300796, 0.657623629, 0.145846935, -0.536417689, -0.30877309, -0.215688875, -0.059783483, 0.308552816, 0.223908462, -0.170183819, 0.308228422, -0.032122267, -0.311985796, 0.523864306, 0.212025559, -0.417972741, 0.168577214, 0.726579789, 0.642160684, -0.826450792, -0.691254262, 0.381880791, 0.79183245, 0.57078044, 0.54842031, 0.37705784, 0.004811671, -0.189490696, 0.027858731, 0.270281335, 0.12617504, 0.22441015, 0.088713533]

for _ in range(N):
    f = 0.6961
    q = 1.67984
    for k in range(6):
        #f = a + b*f + np.random.normal(loc=0, scale=s)
        f = a + b*f + np.random.choice(errors)
        q *= np.exp(f/100)
    L.append(q)

print(np.percentile(L, 25), np.percentile(L, 50), np.percentile(L, 75))

f = 0.6961
q = 1.67984
NL = [q]

for k in range(240):
    f = a + b*f + np.random.normal(loc=0, scale=s)
    q *= np.exp(f/100)
    NL.append(q)

plt.plot(NL)
plt.show()