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

tVals3a = randVals(tMuon,tPion) # need this as global var
numBins3a = 30
counts3a, edges3a = np.histogram(tVals3a, bins=numBins3a, range=(tStart, tEnd))
binWidth3a = (tEnd-tStart)/numBins3a
binCentre3a = edges3a[:-1] + binWidth3a/2

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

def shiftedBinnedNLL(parms, binCentre, tStart, tEnd, numBins, lenData, counts, shift_for_uncert, fun_min):
    return np.abs(binnedNLL(parms, binCentre, tStart, tEnd, numBins, lenData, counts) - fun_min - shift_for_uncert)

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
    #print(shiftedBinnedNLL([(tau_muon_est,tau_pion_est)],binCentre, tStart, tEnd, numBins, lenData, counts, 0.5,result["fun"]))
    tau0 = [(1e-6,1e-17)]
    zeros = opt.minimize(shiftedBinnedNLL,x0=tau0, args=(binCentre, tStart, tEnd, numBins, lenData, counts, 0.5, result["fun"]), method="SLSQP", bounds=bounds)
    print(zeros)
    #tau_muon_uncert = z
    #tau_pion_uncert = np.sqrt(cov[1,1])
    return tau_muon_est, tau_pion_est #, tau_muon_uncert, tau_pion_uncert


print("Exercise 3a)")
plotHist()
print("Graph generated and saved.")
print("")

tMuon3b, tPion3b = tauEstimates(tMuon, tMuonUncert, tPion, tPionUncert)
print("Exercise 3b)")
print(f"Estimated lifetime of muon: {tMuon3b}s")
print(f"Estimated lifetime of pion: {tPion3b}s")
print("")

def tauUncerts():
    #length = len(tVals3a)
    length = 1000
    tau_m = np.linspace(tMuon3b-1e-6,tMuon3b+1e-6,length)
    tau_p = np.linspace(tPion3b-2.5e-17,tPion3b+2.5e-17,length)
    nll_evaluated = []
    for i in range(length): # NLL is now a 2d function of tau_m & tau_p, 
        tau_p_val = np.ones(length)*tau_p[i]
        nll_evaluated.append([binnedNLL((tau_m[j],tau_p_val[j]),binCentre3a,tStart,tEnd,numBins3a,len(tVals3a),counts3a) for j in range(length)])
    print(np.shape(nll_evaluated))
    plt.plot(tau_m,nll_evaluated[499])
    #plt.plot(tau_m,np.ones(10000)*0.5)
    plt.show()
    #plt.clf()
    #print(tau_muon_3b-2.144e-6)

#tauUncerts()

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
