import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

tau_muon = 2.1969811e-6 # [s]
tau_muon_uncert = 0.0000022e-6 # [s]

tau_pion = 8.43e-17 # [s]
tau_pion_uncert = 0.13e-17 # [s]

N_0 = 1 # normalisation
def N(t): # exp. decay function (eq. (1) )
    return (N_0/(tau_muon-tau_pion) * (np.exp(-t/tau_muon)-np.exp(-t/tau_pion)))

def one_over_N(t): # need this to find the max. of N(t) for the acc-reject method as N(t=0) != max(N(t))
    return 1/N(t)

np.random.seed(42) # setting random seed for reprocucibility

# plotting the function N(t) to understand what it looks like, will delete later
x = np.linspace(0,1e-5,100000)
plt.plot(x,N(x),".",markersize=1)
plt.xlabel("t")
plt.ylabel("Counts")
plt.show()
plt.clf()

# generate 10k decay times according to N(t) using accept-reject method
accepted_t_vals = []
t_0 = 1e-30
t_bound = [(1e-30,3e-5)]
n_max = 1/(optimize.minimize(one_over_N,x0=t_0,bounds=t_bound)["fun"]) # minimising 1/N = maximising N
i=0
while i < 10000:
    t_i = np.random.uniform(0,3e-5)
    y_i = np.random.uniform(0,n_max)
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