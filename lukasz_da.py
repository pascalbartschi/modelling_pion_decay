import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

# particle lifetimes and uncertainties (muon, pion)
tau = (2.1969811e-6, 2.6033e-8)
tau_unc = (0.0000022e-6, 0.0005e-8)

# variables
tBound = (1e-8, 3e-5)

# decay function
def N(t, param, N0=1) :
    return (N0/(param[0]-param[1]))*(np.exp(-t/param[0]) - np.exp(-t/param[1]))

# accept reject unifogirm
def accept_uni(pdf, param, lowhi) :
    lx, hx, ly, hy = lowhi
    while True :
        x = np.random.uniform(low=lx, high=hx)
        y = np.random.uniform(low=ly, high=hy)
        if pdf(x, param) > y :
            break
    return x
        
# random numbers in an interval given a pdf
def randVals(pdf, samples, params) :
    minVal = opt.minimize(lambda t : 1/pdf(t, params), 0, bounds=[tBound])['fun']
    maxVal = 1/minVal

    # accept-reject
    return [accept_uni(pdf, params, (0, tBound[1], 0, maxVal)) for i in range(samples)]

def wBin(mx, mn, n) :
    return (mx-mn)/n

def threeA() :
    tVals = randVals(N, 10000, tau)
    nBins = 60
    counts, edges = np.histogram(tVals, bins=nBins, range=tBound)
    wBins = wBin(max(edges), min(edges), len(edges)-1)
    cBins = edges[:-1] + wBins/2
    plt.bar(cBins, counts, width=wBins, label="Generated Decay Times", alpha=0.5)
    plt.xlabel("t [s]")
    plt.ylabel("Number of entries")
    plt.title("3(a) Histogram of 10'000 simulated decay times")
    plt.legend()
    plt.savefig("Exercise 3a.png")
    plt.clf()

def nllBinned(params, pdf, x, l, counts, w) :
    pred = l*pdf(x, params)*w
    return -sum(counts*np.log(pred)-pred)

def tauEst(tVals=randVals(N,10000,tau)) :
    # generate random data according to the distribution
    #tVals = randVals(N, 10000, tau) // take as argument for modularity

    # compute length of these values once
    l = len(tVals)

    # define number of bins
    nBins = 50
    counts, edges = np.histogram(tVals, bins=nBins, range=tBound)
    wBins = wBin(max(edges), min(edges), len(edges)-1)
    cBins = edges[:-1] + wBins/2

    tau_range = (0.5e-6, 0.5e-8)
    tauBounds = [(tau[0]-tau_range[0], tau[0]+tau_range[0]), (tau[1]-tau_range[1], tau[1]+tau_range[1])]
    #tauBounds = [(1.5e-6, 3e-6), (1.5e-8, 3e-8)]

    result = opt.minimize(nllBinned, (2e-6, 2.2e-8), args=(N, cBins, l, counts, wBins), bounds=tauBounds, method='Powell')
    return result["x"][0], result["x"][1], result['success']

def threeB() :
    print(tauEst())

def threeC(reps) :
    tauVals=[[],[]]
    suc = True
    for i in range(reps) :
        out = tauEst()
        tauVals[0].append(out[0])
        tauVals[1].append(out[1])
        if out[2] == False : suc = False

    if suc == False :
        print("Optimisation failed!")
    else :
        mu_mean = np.mean(tauVals[0])
        pi_mean = np.mean(tauVals[1])
        mu_std = np.std(tauVals[0])
        pi_std = np.std(tauVals[1])
        print("muon avg: decay time [s] and stdev [s]: " + str(mu_mean) + ", " + str(mu_std))
        print("pion avg: decay time [s] and stdev [s]: " + str(pi_mean) + ", " + str(pi_std))

def four(sigmaTvals=[1/100, 1/10, 1*tau[1]]):
    for sigmaT in sigmaTvals:
        tValsFour = randVals(N,10000,tau) + np.random.normal(loc=0,scale=sigmaT,size=10000)
        out = tauEst(tVals=tValsFour)
        print("For sigma_T = "+str(sigmaT))
        print("muon decay time [s]: " + str(out[0]))
        print("pion decay time [s]: " + str(out[1]))

four()