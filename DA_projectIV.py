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

np.random.seed(42) # setting random seed for reproducibility

# generate 10k decay times according to N(t) using accept-reject method
accepted_t_vals = []
t_0 = 1e-10
t_bound = [(1e-10,3e-5)]
n_max = 1/(optimize.minimize(one_over_N,x0=t_0,bounds=t_bound)["fun"]) # minimising 1/N = maximising N
i=0
t_start = 0
t_end = 3e-5
while i < 10000:
    t_i = np.random.uniform(t_start,t_end)
    y_i = np.random.uniform(0,n_max)
    if N(t_i) > y_i:
        accepted_t_vals.append(t_i)
        i += 1

# plotting the histogram of the decay times
num_of_bins = 30
counts, edges = np.histogram(accepted_t_vals,bins=num_of_bins,range=(t_start,t_end))
binwidth = (t_end-t_start)/num_of_bins
bin_centres = edges[:-1]+binwidth/2
plt.bar(bin_centres,counts,width=binwidth,label="Generated decay times",alpha=0.5)
#plt.hist(accepted_t_vals,bins=num_of_bins,label="Generated decay times",alpha=0.5)
plt.xlabel("t [s]")
plt.ylabel("Number of entries")
plt.title("3(a) Histogram of 10'000 simulated decay times")
plt.legend()
plt.savefig("projectIV 3a histogram.png")
plt.clf()

#--------
# 3(b)
def N(t,tau_mu,tau_pi):
    return (N_0/(tau_mu-tau_pi) * (np.exp(-t/tau_mu)-np.exp(-t/tau_pi)))

data = np.array(accepted_t_vals)

def binned_nll(params):
    tau_mu,tau_pi = params
    pdf = N(bin_centres,tau_mu,tau_pi)
    prediction = len(data)*pdf*((t_end-t_start)/num_of_bins) # = f_i | last factor = delta x = binwidth
    summands = counts*np.log(prediction)-prediction
    return -summands.sum()

# minimising binned nll, bounds and starting values arbitrarily chosen...
bounds = [(tau_muon-10*tau_muon_uncert,tau_muon+10*tau_muon_uncert),(tau_pion-10*tau_pion_uncert,tau_pion+10*tau_pion_uncert)]
tau_0 = [(tau_muon-10*tau_muon_uncert,tau_pion-10*tau_pion_uncert)]
result = optimize.minimize(binned_nll,x0=tau_0,method="SLSQP",bounds=bounds)
tau_muon_3b = result["x"][0]
tau_pion_3b = result["x"][1]
print("Estimated tau_muon =",tau_muon_3b,"s")
print("Estimated tau_pion =", tau_pion_3b,"s")
# need uncertainty on estimates and comparison w/ true vals
