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

# accept reject uniform
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
    minVal = opt.minimize(lambda t : -pdf(t, params), x0=tBound[0])['fun']
    maxVal = -minVal

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
    plt.savefig("Exercise_3a.png")
    plt.clf()

def nllBinned(params, pdf, x, l, counts, w) :
    pred = l*pdf(x, params)*w
    return -sum(counts*np.log(pred)-pred)

def tauEst(tVals) :
    # compute length of these values once
    l = len(tVals)

    # define number of bins
    nBins = 50
    counts, edges = np.histogram(tVals, bins=nBins, range=tBound)
    wBins = wBin(max(edges), min(edges), len(edges)-1)
    cBins = edges[:-1] + wBins/2

    tau_range = (0.5e-6, 0.5e-8)
    tauBounds = [(tau[0]-tau_range[0], tau[0]+tau_range[0]), (tau[1]-tau_range[1], tau[1]+tau_range[1])]

    result = opt.minimize(nllBinned, (2e-6, 2.2e-8), args=(N, cBins, l, counts, wBins), bounds=tauBounds, method='Powell')
    mparam = (result["x"][0], result["x"][1])
    mval = result["fun"]

    unc_est = 0
    return mparam, unc_est, result['success']

def threeB() :
    # generate random data according to the distribution
    tVals = randVals(N, 10000, tau)
    print(tauEst(tVals))

def threeC(reps) :
    tauVals=[[],[]]
    suc = True
    for i in range(reps) :
        tVals = randVals(N, 10000, tau)
        out = tauEst(tVals)
        tauVals[0].append(out[0][0])
        tauVals[1].append(out[0][1])
        if out[2] == False : suc = False

    if suc == False :
        print("Optimisation failed!")
    else :
        mu_mean = np.mean(tauVals[0])
        pi_mean = np.mean(tauVals[1])
        mu_std = np.std(tauVals[0])
        pi_std = np.std(tauVals[1])
        print("muon avg: decay time and stdev: " + str(mu_mean) + ", " + str(mu_std))
        print("pion avg: decay time and stdev: " + str(pi_mean) + ", " + str(pi_std))

def randValSmear(pdf, samples, params, mu, sigma) :
    rands = randVals(pdf, samples, params)
    smear = np.random.normal(loc=mu, scale=sigma, size=samples)
    randsmear = rands + smear
    # crop to zero where values are negative
    return np.where(randsmear >=0, randsmear, 0)

def four() :
    tauVals = [[], []]
    fig, ax = plt.subplots(3)
    sigmaf = [1/100, 1/10, 1]
    for i in range(len(sigmaf)):
        tVals = randValSmear(N, 10000, tau, 0, sigmaf[i]*tau[1])
        tauEst(tVals)
        out = tauEst(tVals)
        print(out)
        tauVals[0].append(out[0][0])
        tauVals[1].append(out[0][1])

        nBins = 60
        counts, edges = np.histogram(tVals, bins=nBins)
        wBins = wBin(max(edges), min(edges), len(edges)-1)
        cBins = edges[:-1] + wBins/2
        ax[i].bar(cBins, counts, width=wBins, label="Generated Decay Times", alpha=0.5)
        ax[i].plot(tVals, N(tVals, out[0])/210,".",markersize=1,label="Fitted Decay Function")
        ax[i].set_xlabel("t [s]")
        ax[i].set_ylabel("Number of entries")
        ax[i].set_title("$\sigma_t = $"+str(sigmaf[i])+r"$\cdot \tau_\pi$")
    plt.savefig("Exercise_4.png")
