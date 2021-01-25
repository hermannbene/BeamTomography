# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

emitX = 200e-9/6400
emitY = 200e-9/6400

alpha0X = 0
beta0X = 2.0e-2
gamma0X = (1+alpha0X**2)/beta0X

alpha0Y = 0
beta0Y = 3.0e-2
gamma0Y = (1+alpha0Y**2)/beta0Y

offsetX = 0.01
offsetY = -0.0

def betaX(s):
    return beta0X - 2*(s+offsetX)*alpha0X + (s+offsetX)**2*gamma0X

def alphaX(s):
    return alpha0X - (s+offsetX)*gamma0X

def betaY(s):
    return beta0Y - 2*(s+offsetY)*alpha0Y + (s+offsetY)**2*gamma0Y

def alphaY(s):
    return alpha0Y - (s+offsetY)*gamma0Y

def calcSig(alpha, beta, emit):
    gamma = (1+alpha**2)/beta
    sig = emit*np.array([[beta, -alpha], [-alpha, gamma]])
    return sig

def generateDistribution(s, n=int(1e6)):
    sigX = calcSig(alphaX(s), betaX(s), emitX)
    sigY = calcSig(alphaY(s), betaY(s), emitY)
    x = np.random.multivariate_normal([0,0], sigX, n)
    y = np.random.multivariate_normal([0,0], sigY, n)
#    plt.hist2d(x[:,0]*1e6, y[:,0]*1e6, 128)
    
    return x[:,0]*1e6, x[:,1]*1e6, y[:,0]*1e6, y[:,1]*1e6

def generateDistributionBallistic(s, n=int(1e6)):
    
    return 
    
def generateProjection(s, theta, d, doConv=True):
    bins = np.arange(min(d), max(d), 0.05)
    bins = np.append(bins,max(d))
    x, xp, y, yp = generateDistribution(s)
    cos = np.cos(np.pi*theta/180)
    sin = np.sin(np.pi*theta/180)
    xRot = sin*x+cos*y
    
    hist, b = np.histogram(xRot, bins=bins)
    hist = hist/np.sum(hist)
    
    wireHalfWidth = 0.5
    wireHalfWidthBins = int(wireHalfWidth/(bins[1]-bins[0]))
    
    convFunction = np.squeeze(np.zeros(len(bins)-1))
    m = int(len(convFunction)*0.5)
    convFunction[m-wireHalfWidthBins:m+wireHalfWidthBins]=1.
    convFunction = convFunction/np.sum(convFunction)
    if doConv:
        hist = signal.convolve(hist, convFunction, mode = "same")
#    print(hist)
        hist = np.interp(d, bins[:-1]-0.*(bins[1]-bins[0]), hist)
    else:
        hist = np.interp(d, bins[:-1]+0.5*(bins[1]-bins[0]), hist)
    return hist
    
    
def generateTestSignal(Z, Angles, D):
    signal = np.zeros(D.shape)
    for i,s in enumerate(Z[:,0,0]):
        for j,theta in enumerate(Angles[i,:,0]):
            signal[i,j,:] = generateProjection(s, theta, D[i,j,:])
    snr = 30
    noise = np.random.normal(0, np.max(signal)/snr, signal.shape)           
    return signal+noise

def loadTestData():
#    z = np.array([0.])
    z = np.array([-0.06, -0.04, -0.02, -0.01, 0., 0.01, 0.02, 0.04, 0.06]) # in m
#    z = (np.array([-54741, -24741, -9741, -5241, -241, 14759, 44759])+5241)*1e-6
    angles = np.linspace(0,320,9)
    d = np.linspace(-5, 5, 41)
    
    nZ = len(z)
    nAngles = len(angles)
    nD = len(d)
        
    D = np.array([[d]*nAngles]*nZ)
    Angles = np.array([np.array([angles]*nD).T]*nZ)
    Z = np.moveaxis(np.array([np.array([z]*nAngles).T]*nD), [0,1,2], [2,0,1])
    
    

    signal = generateTestSignal(Z, Angles, D)
    print(d.shape, signal.shape, angles.shape )
    return z, angles, D, signal


#Z, D, Angles, signal = loadTestData()