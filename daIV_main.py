import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy.stats import norm

# particle lifetimes and uncertainties (muon, pion)
tau = (2.1969811e-6, 2.6033e-8)
tau_unc = (0.0000022e-6, 0.0005e-8)

# variables
tBound = (1e-8, 3e-5)

# decay function
def N(t, param, N0=1) :
    return (N0/(param[0]-param[1]))*(np.exp(-t/param[0]) - np.exp(-t/param[1]))

def N_gauss(t, param, N0=1) :
    return N(t, param, N0) + norm.pdf(t)

# accept reject uniform
def accept_uni(pdf, param, lowhi) :
    lx, hx, ly, hy = lowhi
    while True :
        x = np.random.uniform(low=lx, high=hx)
        y = np.random.uniform(low=ly, high=hy)
        if pdf(x, param) > y :
            break
    return x

def pull(rec_quant, gen_quant, rec_quant_unc):
    return (np.array(rec_quant)-gen_quant)/np.array(rec_quant_unc)

def pull_dist(pull_vals1, pull_vals2):
    pull_vals = (pull_vals1,pull_vals2)
    fig, ax = plt.subplots(2)
    xlabels = [r"$\hat{\tau}_{\mu}$",r"$\hat{\tau}_{\mu}$"]
    nBins = 30

    # define a local gaussian
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev) ** 2)
    # initial guess for params
    p0 = [1.0, 0.0, 1.0]

    for i in range(2):
        counts, edges = np.histogram(pull_vals[i], bins=nBins)
        wBins = wBin(max(edges), min(edges), len(edges)-1)
        cBins = edges[:-1] + wBins/2
        # fit the data
        # Fit the data using the curve_fit function
        coeff, _ = opt.curve_fit(gaussian, cBins, counts, p0=p0)
        print(f"Histogram fitted to N({coeff[1]}, {coeff[2]**2})")
        ax[i].plot(cBins, gaussian(cBins[:-1], *coeff), 'r--')
        ax[i].bar(cBins, counts, width=wBins, label="Generated Decay Times", alpha=0.5)
        ax[i].set_xlabel("pull of "+str(xlabels[i]))
        ax[i].set_ylabel("Number of entries")
        ax[i].set_title("Pull Distribution of "+str(xlabels[i]))
        ax[i].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("Exercise_3d.png")
        
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
    plt.savefig("Exercise 3a.png")
    plt.clf()

def nllBinned(params, pdf, x, l, counts, w) :
    pred = l*pdf(x, params)*w
    return -sum(counts*np.log(pred)-pred)

def nllBinnedUnc(mparams, pdf, x, l, counts, w, uncBounds, paramBounds) :
    lpar = len(mparams)
    unc_est = []
    for i in range(lpar) :
        mpars = lambda p, param : np.insert(param, i, p)
        inp = lambda p : np.insert(opt.minimize(lambda ptwo : nllBinned(list(mpars(p, ptwo)), N, x, l, counts, w), x0=np.delete(mparams, i), bounds=[paramBounds[i]])['x'], i, p)
        nll = lambda p : nllBinned(list(inp(p)), pdf, x, l, counts, w)
        res = opt.minimize(nll, x0=mparams[i], bounds=[paramBounds[i]], method='Nelder-Mead')
        minp = res['x']
        mval = res['fun']
        # find where we cross the 2.3/2 mark to find confidence interval
        func = lambda p : nll(minp+p) - mval - 2.3/2
        uncm = uncp = 0
        uncmval = uncpval = -2.3/2
        
        for j in np.linspace(uncBounds[i][0], uncBounds[i][1], 1000) :
            if abs(func(-j)) < abs(uncmval) :
                uncm = -j
                uncmval = func(uncm)
            if abs(func(j)) < abs(uncpval) :
                uncp = j
                uncpval = func(uncp)
        print(func(uncm))
        print(func(uncp))
        unc_est.append([abs(uncm), abs(uncp)])

    return unc_est

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

    uncBounds = [[1e-13, 1e-6], [1e-13, 1e-8]]
    paramBounds = [[1e-10, 5e-6], [1e-10, 5e-6]]
    unc_est = ([0, 0], [0, 0])
    unc_est = nllBinnedUnc(mparam, N, cBins, l, counts, wBins, uncBounds, paramBounds)
    return mparam, unc_est, result['success']

def threeB() :
    # generate random data according to the distribution
    tVals = randVals(N, 10000, tau)
    print(tauEst(tVals))

def threeC(reps) :
    tauVals=[[],[]]
    tauUncs=[[],[]]
    suc = True
    for i in range(reps) :
        tVals = randVals(N, 10000, tau)
        out = tauEst(tVals)
        tauVals[0].append(out[0][0])
        tauVals[1].append(out[0][1])
        tauUncs[0].append(out[1][0][0])
        tauUncs[1].append(out[1][1][0])
        if out[2] == False : suc = False

    if suc == False :
        print("Optimisation failed!")
    else :
        mu_mean = np.mean(tauVals[0])
        pi_mean = np.mean(tauVals[1])
        mu_std = np.std(tauVals[0])
        pi_std = np.std(tauVals[1])
        mu_pull = pull(tauVals[0],tau[0],tauUncs[0])
        pi_pull = pull(tauVals[1],tau[1],tauUncs[1])
        print("Mean and std of the pull dist. for muon",np.mean(mu_pull),np.std(mu_pull))
        print("Mean and std of the pull dist. for pion muon",np.mean(pi_pull),np.std(pi_pull))
        pull_dist(mu_pull,pi_pull)
        print("muon avg: decay time and stdev: " + str(mu_mean) + ", " + str(mu_std))
        print("pion avg: decay time and stdev: " + str(pi_mean) + ", " + str(pi_std))

def randValSmear(pdf, samples, params, mu, sigma) :
    rands = randVals(pdf, samples, params)
    smear = np.random.normal(loc=mu, scale=sigma, size=samples)
    randsmear = rands + smear
    # crop to zero where values are negative
    return np.where(randsmear >=0, randsmear, 0)

def four(pdf) :
    tauVals = [[], []]
    fig, ax = plt.subplots(3)
    sigmaf = [1/100, 1/10, 1]
    for i in range(len(sigmaf)):
        tVals = randValSmear(pdf, 10000, tau, 0, sigmaf[i]*tau[1])
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
        ax[i].plot(tVals, pdf(tVals, out[0])/210,".",markersize=1,label="Fitted Decay Function")
        ax[i].set_xlabel("t [s]")
        ax[i].set_ylabel("Number of entries")
        ax[i].set_title("$\sigma_t = $"+str(sigmaf[i])+r"$\cdot \tau_\pi$")
    plt.show()
