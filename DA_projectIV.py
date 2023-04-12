import numpy as np
import matplotlib.pyplot as plt

tau_muon = 2.1969811e-6 # [s]
tau_muon_uncert = 0.0000022e-6 # [s]

tau_pion = 8.43e-17 # [s]
tau_pion_uncert = 0.13e-17 # [s]

N_0 = 1 # normalisation
def N(t): # exp. decay function (eq. (1) )
    return N_0/(tau_muon-tau_pion) * (np.exp(-t/tau_muon)-np.exp(-t/tau_pion))

np.random.seed(42) # setting random seed for reprocucibility

# plotting the function N(t) to understand what it looks like, will delete later
x_test = np.random.uniform(0,3e-5,1000)
plt.plot(x_test,N(x_test),".",markersize=1)
plt.xlabel("t")
plt.ylabel("Counts")
plt.show()
plt.clf()

# generate 10k decay times according to N(t) using accept-reject method
accepted_t_vals = []
n_max = N(t=0) # exponential decay => starting value greatest
i=0
while i < 10000:
    t_i = np.random.uniform(0,3e-5)
    y_i = np.random.uniform(0,45000)
    if N(t_i) > y_i:
        accepted_t_vals.append(t_i)
        i += 1

plt.hist(accepted_t_vals,bins=30,label="Generated decay times",alpha=0.5)
plt.xlabel("t [s]")
plt.ylabel("Number of entries")
plt.title("3(a) Histogram of 10'000 simulated decay times")
plt.legend()
plt.show()
plt.clf()