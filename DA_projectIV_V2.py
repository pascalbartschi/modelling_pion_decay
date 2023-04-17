import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

tMuon = 2.1969811e-6 # [s]
tMuonUncert = 0.0000022e-6 # [s]

tPion = 8.43e-17 # [s]
tPionUncert = 0.13e-17 # [s]

def N(t, t_muon, t_pion, N0=1): # exp. decay function (eq. (1) )
    return (N0/(t_muon-t_pion) * (np.exp(-t/t_muon)-np.exp(-t/t_pion)))

def inverseN(t, t_muon, t_pion): # need this to find the max. of N(t) for the acc-reject method as N(t=0) != max(N(t))
    return 1/N(t, t_muon, t_pion)

tStart = 0 # need this as global variable.
tEnd = 3e-5  

def randVals(t_muon, t_pion):
    tVals = []
    t0 = 1e-10
    tBounds = [(1e-10,3e-5)]
    minVal = opt.minimize(inverseN, x0=t0, args=(t_muon, t_pion), bounds=tBounds)["fun"]
    nMax = 1/minVal # max of decay function, needed for accept-reject method
    
    i = 0
    while i < 10000: # accept-reject method to get 10k t values according to the decay function
        t_i = np.random.uniform(tStart,tEnd)
        y_i = np.random.uniform(0,nMax)
        if N(t_i, t_muon, t_pion) > y_i:
            tVals.append(t_i)
            i += 1    
    return tVals

tVals_3a = randVals(tMuon,tPion)

def plotHist(numBins=30): # plotting the accepted decay times (= t values) in a histogram to then perform the binned neg. log. likelih.
    tVals = randVals(tMuon, tPion)
    counts, edges = np.histogram(tVals, bins=numBins, range=(tStart, tEnd))
    binWidth = (tEnd-tStart)/numBins
    binCentre = edges[:-1] + binWidth/2
    plt.bar(binCentre, counts, width=binWidth, label="Generated Decay Times", alpha=0.5)
    plt.xlabel("t [s]")
    plt.ylabel("Number of entries")
    plt.title("3(a) Histogram of 10'000 simulated decay times")
    plt.legend()
    plt.savefig("Exercise 3a.png")
    plt.clf()
    #plt.show()
    

def binnedNLL(parms, binCentre, tStart, tEnd, numBins, lenData, counts):
    """binned negative log likelihood to find estimates of tau_mu & tau_pi from the 10'000 t values we created before and plotted in a histogram. We get the estimates by minimising the nll."""
    t_muon, t_pion = parms
    pdf = N(binCentre, t_muon, t_pion)
    prediction = lenData*pdf*((tEnd-tStart)/numBins) # = f_i | last factor = delta x = binwidth
    summands = counts*np.log(prediction)-prediction
    return -summands.sum()

def tauEstimates(t_muon, t_muon_uncert, t_pion, t_pion_uncert):
    tVals = randVals(t_muon, t_pion)
    lenData = len(tVals)
    numBins = 30
    counts, edges = np.histogram(tVals, bins=numBins, range=(tStart, tEnd))
    binWidth = (tEnd-tStart)/numBins
    binCentre = edges[:-1] + binWidth/2
    
    bounds = [(t_muon-10*t_muon_uncert,t_muon+10*t_muon_uncert),(t_pion-10*t_pion_uncert,t_pion+10*t_pion_uncert)]
    tau0 = [(t_muon-10*t_muon_uncert,t_pion-10*t_pion_uncert)]
    result = opt.minimize(binnedNLL,x0=tau0, args=(binCentre, tStart, tEnd, numBins, lenData, counts), method="SLSQP",bounds=bounds)
    tau_muon_est = result["x"][0]
    tau_pion_est = result["x"][1]
    return tau_muon_est, tau_pion_est


print("Exercise 3a)")
plotHist()
print("Graph generated and saved.")
print("")

tMuon3b, tPion3b = tauEstimates(tMuon, tMuonUncert, tPion, tPionUncert)
print("Exercise 3b)")
print(f"Estimated lifetime of muon: {tMuon3b}s")
print(f"Estimated lifetime of pion: {tPion3b}s")
print("")


print("Exercise 3c)")
#np.random.seed()

# many measurements..?
tMuonVals = []
tPionVals = []
for i in range(10):
    tMuonEst, tPionEst = tauEstimates(tMuon, tMuonUncert, tPion, tPionUncert)
    tMuonVals.append(tMuonEst)
    tPionVals.append(tPionEst)
    print(i)

print(tMuonVals)
