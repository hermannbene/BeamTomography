# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from loadData import gauss, loadAndPrepareInput
from scipy import signal
from generateTestData import loadTestData
from plot import plotProjections2D, plotError2D
from scipy.optimize import curve_fit

def growthRate(X, x, bins, y, angle, convFunction = None, microCharge = None):
    """
    X: particle coordinates [[xi], [yi]]
    x: projection axis
    bins: bin edges corresponding to x
    y: normalized measurement values
    angle: projection angle in deg
    """

    """ Rotation """
    cos = np.cos(np.pi*angle/180)
    sin = -np.sin(np.pi*angle/180)
    
    Xrot = sin*X[0]+cos*X[1]

    """ Binning """
#    hist, b = np.histogram(Xrot, bins=bins, density=False)
#    hist = hist/np.sum(hist)
    
    if microCharge==None:
        hist, b = np.histogram(Xrot, bins=bins)
        hist = hist/np.sum(hist)/(bins[1]-bins[0])
        
    else:
        hist, b = np.histogram(Xrot, bins=bins, density=False)
        hist = np.array(hist, dtype = float)*microCharge/np.average(np.diff(bins))
    if not(convFunction is None):
        if sum(convFunction)>0:
            hist = signal.convolve(hist, convFunction, mode = "same")
    #        plt.figure()
    #        plt.plot(bins[:-1], hist)
    #        plt.plot(bins[:-1], histConv)
    #        plt.plot(bins[:-1], convFunction)
    #    hist=histConv
    """ growth rate calculation """
    hist = np.interp(x, bins[:-1]+0.5*(bins[1]-bins[0]), hist)
#    hist = hist/np.sum(hist)
    birthRate = (y-hist)
#    birthRate/=np.max(hist)
#    birthRate*= np.sum(np.abs(y-hist)*(bins[1]-bins[0]))

    birthRateX = np.interp(Xrot, x, birthRate)

    return birthRateX

def grow(X, birthRateX, smoothing):
    randomX = np.random.random(len(X[0]))
    keepIdx = np.argwhere(-birthRateX<randomX).flatten()
    addIdx = np.argwhere(birthRateX>randomX).flatten()
    Xnew = np.hstack([X[:,keepIdx], X[:,addIdx]])
#    Xnew = np.hstack([X.copy()[:,keepIdx], X.copy()[:,addIdx]])
#    selectIdx = np.random.randint(0, len(Xnew[1]), len(X[1]))
#    Xnew = Xnew[:,selectIdx]
#    randomizeAmplitude = np.std(Xnew)/len(Xnew)
    Xnew+=np.random.normal(0, smoothing, Xnew.shape)
    return Xnew

class model2D:
    def __init__(self, fileName):
        self.fileName = fileName
        self.nPart = 2e5
        self.y, self.x, self.projectionAngles = loadAndPrepareInput(fileName, manualSave=False)
        self.bins = [np.linspace(self.x[i][0]-(self.x[i][-1]-self.x[i][-2])*0.5, self.x[i][-1]+(self.x[i][-1]-self.x[i][-2])*0.5, 6*len(self.x[i])+1) for i in range(len(self.x))]
        self.projectionAngles=self.projectionAngles[:,0]
        self.X = np.random.rand(2,int(self.nPart))*(np.max(self.x)-np.min(self.x))+np.min(self.x)
        
        self.nAngles = len(self.projectionAngles)
        offsets=[]
        
        for i in range(self.nAngles):
            self.nD = len(self.x[i])
            s = self.y[i]
               
            initGuess = (np.max(s)-np.min(s), self.x[i][np.argmax(s)],  0.1*(np.max(self.x[i])-np.min(self.x[i])), np.min(s))
            fit = curve_fit(gauss, self.x[i], s, p0 = initGuess)[0]
            offsets.append(fit[-1])
            
        integrals = []
        
        for i in range(self.nAngles):                
            self.y[i] -= np.average(offsets)
            integrals.append(np.sum(self.y[i])*np.average(np.diff(self.x[i])))
        maxSList = []

        for i in range(self.nAngles):      
            self.y[i] = self.y[i]/np.average(integrals)
            maxSList.append(np.max(self.y[i]))
        
        self.maxS = np.max(maxSList)
        
#        self.y = [self.y[i,:]/np.average(integrals) for i in range(len(self.y))]
        
        self.wireHalfWidth = 0.5
        self.wireHalfWidthBins = int(self.wireHalfWidth/(self.bins[0][1]-self.bins[0][0]))
        convFunction = np.squeeze(np.zeros(len(self.bins[0])-1))
        m = int(len(convFunction)*0.5)
        convFunction[m-self.wireHalfWidthBins:m+self.wireHalfWidthBins]=1.
        self.convFunction = convFunction/np.sum(convFunction)
    
        self.i = 0
        self.historyLength=10
        self.history = []
        
    def iterate(self):
        print(self.i)
        self.birthRatesX = []
        for j, angle in enumerate(self.projectionAngles):
            self.birthRatesX.append(growthRate(self.X, self.x[j], self.bins[j], self.y[j], angle, convFunction=self.convFunction, microCharge=1/self.nPart))
            
        birthRateX = np.average(self.birthRatesX, axis=0)/self.maxS
        self.X = grow(self.X, birthRateX, 0.08)
        self.addToHistory()
        self.i+=1

    def uncertainty(self):
#        self.birthRatesX = []
#        for j, angle in enumerate(self.projectionAngles):
#            self.birthRatesX.append(growthRate(self.X, self.x[j], self.bins[j], self.y[j], angle, convFunction=self.convFunction))
        self.samples=[]
        for j, angle in enumerate(self.projectionAngles):
            birthRateX = growthRate(self.X, self.x[j], self.bins[j], self.y[j], angle, convFunction=self.convFunction, microCharge=1/self.nPart)
            Xnew = grow(self.X, birthRateX/self.maxS, 0.08)
            for i in range(10):
                birthRateX = growthRate(Xnew, self.x[j], self.bins[j], self.y[j], angle, convFunction=self.convFunction, microCharge=1/self.nPart)
                Xnew = grow(Xnew, birthRateX/self.maxS, 0.08)
            self.samples.append(Xnew)
#            plotProjections2D([self.samples[-1]], rm.projectionAngles,rm.x, rm.y, rm.bins, convFunction=rm.convFunction, fileName=rm.i)
            
    def addToHistory(self):
        self.history.append(self.X)
        self.history = self.history[-min(self.historyLength, len(self.history)):]

    def saveDistribution(self):
        saveFileName = '/'.join(self.fileName.split('/')[:-1])+'/reconstructedDistribution.npy'
        np.save(saveFileName, self.X)
        
        
if __name__ == "__main__":
    

    path = 'E:/Measurement Data/ATHOS/20210313/Hexapod/'
    fileNames = ['ws_20210313_162151']
    fileNames = [path+f+'/RAWScanData.h5' for f in fileNames]

    for fileName in fileNames:
        rm = model2D(fileName) # reconstruction model
        for i in range(150):
            rm.iterate()
        rm.uncertainty()
        plotProjections2D(rm.samples, rm.projectionAngles,rm.x, rm.y, rm.bins, convFunction=rm.convFunction, fileName=rm.i, microCharge=1./rm.nPart)
#        plotError2D(rm.X, rm.samples)
#        rm.saveDistribution()
#        plt.savefig(fileName+"Tomo.png", dpi=600)
        
