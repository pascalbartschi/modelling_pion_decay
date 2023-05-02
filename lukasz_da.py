import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

# particle lifetimes and uncertainties (muon, pion)
tau = (2.1969811e-6, 2.6033e-8)
tau_unc = (0.0000022e-6, 0.0005e-8)

# variables
tBound = (1e-10, 3e-5)

# decay function
def N(t, param, N0=1) :
    return (N0/(param[0]-param[1]))*(np.exp(-t/param[0]) - np.exp(-t/param[1]))

# accept reject uniform
def accept_uni(pdf, param, lowhi) :
    lx, hx, ly, hy = lowhi
    while True :
        x = np.random.uniform(low=lx, high=hx)
        y = np.random.uniform(low=ly, high=hy)
        if pdf(x, param) > y :
            break
    return x
        