import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

# particle lifetimes and uncertainties (muon, pion)
tau = (2.1969811e-6, 2.6033e-8)
tau_unc = (0.0000022e-6, 0.0005e-8)

# variables
tBound = (1e-10, 3e-5)

