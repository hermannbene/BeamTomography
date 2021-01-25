# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
from loadData import gauss, loadAndPrepareInput, loadData4d
from scipy import signal
from scipy.optimize import curve_fit
from generateTestData import loadTestData
from scipy.linalg import det
from plot import propagateX, plotProjections4D, twiss, plotPhaseSpace4D, plotTwissEvolution
import pickle


def calcGrowthRate(X, x, bins, y, angle, convFunction = None, microCharge=None):
    """
    X: particle coordinates [[xi], [yi], [xi'], [yi']]
    x: projection axis (displacements of wire)
    bins: bin edges corresponding to x
    y: measurement values from BLM
    angle: projection angles in deg
    """

    cos = np.cos(np.pi*angle/180)
    sin = -np.sin(np.pi*angle/180)
    
    Xrot = sin*X[0]+cos*X[1]
#    Xrot = X
    if microCharge==None:
        hist, b = np.histogram(Xrot, bins=bins)
        hist = hist/np.sum(hist)/(bins[1]-bins[0])
        
    else:
        hist, b = np.histogram(Xrot, bins=bins, density=False)
        hist = np.array(hist, dtype = float)*microCharge/np.average(np.diff(bins))
    if not(convFunction is None) and sum(convFunction)>0:
        hist = signal.convolve(hist, convFunction, mode = "same")
#        plt.figure()
#        plt.plot(bins[:-1], hist)
#        plt.plot(bins[:-1], histConv)
#        plt.plot(bins[:-1], convFunction)
#    hist=histConv
#    plt.plot(x, y)
    
    hist = np.interp(x, bins[:-1]+0.5*(bins[1]-bins[0]), hist)
#    plt.plot(x, hist)
    growthRate = (y-hist)
#    growthRate/=np.max(hist)
    growthRateX = np.interp(Xrot, x, growthRate)
    
    return growthRateX

def redistribute(X, growthRateX, smoothingX, smoothingXP):
    """
    redistributes ensemble X according to growthRateX
    """
    randomX = np.random.random(len(X[0]))
    keepIdx = np.argwhere(-growthRateX<randomX).flatten()
    addIdx = np.argwhere(growthRateX>randomX).flatten()
        
    Xnew = np.hstack([X[:,keepIdx], X[:,addIdx]])
#    selectIdx = np.random.randint(0, len(Xnew[1]), len(X[1]))
#    Xnew = Xnew[:,selectIdx]
#    randomizeAmplitude = np.std(Xnew)/len(Xnew)
    Xnew[:2]+=np.random.normal(0, smoothingX, Xnew[:2].shape)
    Xnew[2:]+=np.random.normal(0, smoothingXP, Xnew[2:].shape)
    return Xnew

class model4D:
    def __init__(self):
        self.Z, self.Angles, self.D, self.s = loadTestData()
#        self.Z, self.Angles, self.D, self.s = loadData4d(normalize=False)
#        self.nZ, self.nAngles, self.nD = self.Z.shape
        self.nZ = len(self.Z)
        self.nAngles = len(self.Angles)
        self.nD = 0
        self.nPart = int(1e5)
        self.X = np.random.rand(4, self.nPart)  # x, y, x", y"
        
#        self.X[0] = self.X[0]*(np.max(self.D)-np.min(self.D))+np.min(self.D)
#        self.X[1] = self.X[1]*(np.max(self.D)-np.min(self.D))+np.min(self.D)
        self.minD = -20
        self.maxD = 20
        self.binwidth = 0.1
        self.bins = np.arange(self.minD, self.maxD+self.binwidth, self.binwidth)
                
        self.X[0] = self.X[0]*(self.maxD-self.minD)+self.minD
        self.X[1] = self.X[1]*(self.maxD-self.minD)+self.minD
        
        self.X[2] = (self.X[2]-0.5)*1e2*2.5
        self.X[3] = (self.X[3]-0.5)*1e2*2.5
        
#        self.Bins = np.zeros((self.nZ, self.nAngles, 6*self.nD+1))
#        self.Bins = []
        offsets=[]
        self.convergenceHistory = []
        self.uncertaintyConvergenceHistory = []
        self.r=1
        for i in range(self.nZ):
#            self.Bins.append([])
            for j in range(self.nAngles):
                self.nD = len(self.D[i][j])
#                self.Bins[i].append(np.linspace(self.D[i][j][0], self.D[i][j][-1]+(self.D[i][j][-1]-self.D[i][j][-2]), 6*self.nD+1))
                s = self.s[i][j]
                   
                initGuess = (np.max(s)-np.min(s), self.D[i][j][np.argmax(s)],  0.1*(np.max(self.D[i][j])-np.min(self.D[i][j])), np.min(s))
                fit = curve_fit(gauss, self.D[i][j], s, p0 = initGuess)[0]
                offsets.append(fit[-1])
#                self.s[i][j] = self.s[i][j]/np.sum(self.s[i][j])/(self.D[i][j][1]-self.D[i][j][0])
        
        integrals = []
        
        for i in range(self.nZ):
            for j in range(self.nAngles):                
                self.s[i][j] -= np.average(offsets)
                integrals.append(np.sum(self.s[i][j])*np.average(np.diff(self.D[i][j])))
        maxSList = []
        for i in range(self.nZ):
            for j in range(self.nAngles):      
                self.s[i][j] = self.s[i][j]/np.average(integrals)
                maxSList.append(np.max(self.s[i][j]))
        
        self.maxS = np.max(maxSList)
        self.wireHalfWidth = 0.5
        wireHalfWidthBins = int(self.wireHalfWidth/(self.bins[1]-self.bins[0]))
        
        convFunction = np.squeeze(np.zeros(len(self.bins)-1))
        m = int(len(convFunction)*0.5)
        convFunction[m-wireHalfWidthBins:m+wireHalfWidthBins]=1.
        self.convFunction = convFunction/np.sum(convFunction)
        
        self.it = 0
        self.historyLength = 10
        self.history = []

    def iterate(self):
        
        print(self.it)
        growthRatesX = []
        for i in range(self.nZ):
    
            Xprop = propagateX(self.X, self.Z[i])
            for j in range(self.nAngles):
                growthRatesX.append(calcGrowthRate(Xprop, self.D[i][j], self.bins, self.s[i][j], self.Angles[j], convFunction=self.convFunction, microCharge=1/self.nPart))
        
        growthRateX = np.average(growthRatesX, axis=0)/self.maxS
        self.convergenceHistory.append(np.average(np.abs(growthRateX)))
        self.X = redistribute(self.X, growthRateX, 0.08, 0.08/(np.max(self.Z)))
        
        self.addToHistory()
        if self.it>5:
            self.r = np.average(np.abs(np.diff(self.convergenceHistory)/self.convergenceHistory[1:])[-5:])
        self.it+=1
        

    def uncertainty(self):
        print('uncertainty sampling')
        self.samples=[]
        for i in range(self.nZ):
            Xprop = propagateX(self.X, self.Z[i])
            for j in range(self.nAngles):
                convergenceHistory = []
                r = 1
                growthRate = calcGrowthRate(Xprop, self.D[i][j], self.bins, self.s[i][j], self.Angles[j], convFunction=self.convFunction, microCharge=1/self.nPart)
                Xnew = redistribute(Xprop, growthRate/self.maxS, 0.08, 0.08/(np.max(self.Z)))
                convergenceHistory.append(np.average(np.abs(growthRate)))
                while r>5e-2:
                    growthRate = calcGrowthRate(Xnew, self.D[i][j], self.bins, self.s[i][j], self.Angles[j], convFunction=self.convFunction, microCharge=1/self.nPart)
                    Xnew = redistribute(Xnew, growthRate/self.maxS, 0.08, 0.08/(np.max(self.Z)))
                    convergenceHistory.append(np.average(np.abs(growthRate)))
                    r=np.abs(convergenceHistory[-1]-convergenceHistory[-2])/np.abs(convergenceHistory[-1])
                self.samples.append(propagateX(Xnew, -self.Z[i]))
                self.uncertaintyConvergenceHistory.append(convergenceHistory)
                
    def addToHistory(self):
        self.history.append(self.X)
        self.history = self.history[-min(self.historyLength, len(self.history)):]

#if __name__ == "__main__":
#    rm = model4D() 
#    rm=pickle.load(open('storedClass0.08_20200918.p', 'rb'))
#    for it in range(20):
#    while rm.r>5e-3:
#        rm.iterate()
#        print(rm.r)
    #    print(twiss(rm.X))
#    rm.uncertainty()
#    plotProjections4D(rm.samples, rm.Z, rm.Angles, rm.D, rm.s, rm.bins, convFunction=rm.convFunction, fileName=0, microCharge=1./rm.nPart)
#    plotPhaseSpace4D(rm.X, np.array([0, 20, 40, 60, 80])*1e-3)
#    plotTwissEvolution(rm.samples, np.linspace(0, 1e-1, 30), rm.Z)
    #twiss(rm.X)
    #hist, bins = np.histogram(X, bins=bins)
    #hist = hist/np.sum(hist)
    #pickle.dump(rm, open('storedClass0.08_20200918.p', 'wb'))
#    plt.figure("projections")
#    plt.savefig('plot20200826/projections.png', dpi=300)
#    plt.figure("phaseSpace")
#    plt.savefig('plot20200826/phaseSpace.png', dpi=300)
    #plt.figure("twissEvolution")
    #plt.savefig('plot20200826/twissEvolution.png', dpi=600)
    #plt.figure("twissEvolution2")
    #plt.savefig('plot20200826/twissEvolution2.png', dpi=600)
        
        
    #def SNR(s):
    #    noise = np.std(s[:,0:5])
    #    avg = np.average(s[:,0:5])
    #    signal = np.max(s)
    #    print((signal-avg)/noise)
    #    
    #for i in range(6):
    #    SNR(rm.s[i])
        
        