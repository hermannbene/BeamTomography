# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.optimize import curve_fit
import scipy.constants as sc
import os
from scipy.integrate import simps

def gauss(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+offset

def getData(fileName, sensorName):
    """
    Read data from .h5 file, remove nans

    Parameters
    ----------
    fileName    :   str
                    name of file to be read
    sensorName  :   str
                    name of sensor whos data is loaded
    Returns
    -------
    x   :   array
            Array of x values visited
    y   :   array
            Array of y values visited
    s   :   array
            Array containing Beam Loss measured at (x,y)
    """

    f = h5.File(fileName, 'r')
    x = np.array(f['PositionHEX']["x"])
    y = np.array(f['PositionHEX']["y"])
    
    s = np.array(f[sensorName])
#    x, y, s = pwt.removeNan([x,y,s])
    return x,y,s

def removeNan(dataList):
    isNan = np.logical_or.reduce([np.isnan(d) for d in dataList])
    idx = np.argwhere(np.logical_not(isNan)).flatten()
    return [d[idx] for d in dataList]

def getDataManual(fileName, sensorName):
    f = h5.File(fileName, 'r')
    x = np.array(f['PositionHEX']["x"])
    y = np.array(f['PositionHEX']["y"])
    
    s = np.array(f['SATSY03-DLAC080-DHXP:BLM1_comp'])
    x, y, s = removeNan([x,y,s])
    nShots=40
    setupFilename = 'D:/Measurement Data/ATHOS/20201002/ws/ws_20201003_051658/scanPreparation.h5'
    fSetup = h5.File(setupFilename, 'r')
    edges = [fSetup['Points/P'+str(i)].value for i in range(18)]
    
    xFit = []
    yFit = []
    sFit = []
    for i in range(9):
        xStart = edges[2*i][0]
        xEnd = edges[2*i+1][0]
        yStart = edges[2*i][1]
        yEnd = edges[2*i+1][1]
        xFit.extend(np.linspace(xStart, xEnd, nShots))
        yFit.extend(np.linspace(yStart, yEnd, nShots))
        
    xFit = np.array(xFit)
    yFit = np.array(yFit)
#    fFit = interp2d(x, y, s, fill_value=0, kind = 'cubic')
#    sFit = np.array([fFit(xFit[i], yFit[i]) for i in range(len(xFit))])
    for i in range(len(xFit)):
        distance = np.sqrt((x-xFit[i])**2+(y-yFit[i])**2)
        sFit.append(s[np.argmin(distance)])
    sFit = np.array(sFit)
    return xFit,yFit,sFit

def getDataRaw(fileName, sensorName):
    """
    Read data from .h5 file, remove nans

    Parameters
    ----------
    fileName    :   str
                    name of file to be read
    sensorName  :   str
                    name of sensor whos data is loaded
    Returns
    -------
    x   :   array
            Array of x values visited
    y   :   array
            Array of y values visited
    s   :   array
            Array containing Beam Loss measured at (x,y)
    """

    f = h5.File(fileName, 'r')
    x = np.array(f['PositionHEX']["x"])
    y = np.array(f['PositionHEX']["y"])
    
    s = np.array(f[sensorName])
    start = 503
    stop = 523
    offset = np.average(s[:, 0:40], axis=1)
    sB2 = [s[i, start:stop]-offset[i] for i in range(len(offset))]
    sB2 = np.array(sB2)
#    signal = -np.sum(sB2, axis=1)
    signal = -simps(sB2, axis=1)
    
#    plt.plot(signal)
    
    return x,y,signal*1e-3


def loadAndPrepareInput(fileName, sensor='SensorValue/SATCL01-DBLM135:B2_LOSS', nWiresInFile=9, wiresUsed=[0,1,2,3,4,5,6,7,8], map180=False,nPerPoint = 1, normalize = True, manualSave = False):
    """
    load data from file, transform it into displacement and angle formulation

    Parameters
    ----------
    fileName    :   str
                    name of file to be read
    sensor      :   str
                    name of sensor whos data is loaded
    nWiresInFile:   int
                    number of wires contained in the loaded measurement
    wiresUsed   :   list/array
                    list or array selecting which wires are included in the dataset returned
    map180      :   boolean
                    switch to turn on and of projectin of the 0-360 interval to a 0-180 interval
                    When True, mapping to 180 deg will take place
    Returns
    -------
    s       :   array
                 Array containing Beam Loss measured at (d,angle)
    d       :   array
                Array of displacements d visited
    angle   :   array
                Array of angles visited
    """
#    x, y, s = getData(fileName, sensor)
    if manualSave: 
        x, y, s = getDataManual(fileName, sensor)
    else:
        x, y, s = getData(fileName, sensor)
#        x, y, s = getDataRaw(fileName, 'SensorValue/SATCL01-DBLM135:LOSS_SIGNAL_RAW')
    x = x.reshape((nWiresInFile, -1))
    y = y.reshape((nWiresInFile, -1))
    s = s.reshape((nWiresInFile, -1))
    x, y, s = x[wiresUsed], y[wiresUsed], s[wiresUsed]
#    s=s/max(s)
    #get displacements and angles
    [d, angle] = transformRep(x,y, map180)
    for i in range(d.shape[0]):
        initGuess = (np.max(s[i])-np.min(s[i]), d[i][np.argmax(s[i])],  0.1*(np.max(d[i])-np.min(d[i])), np.min(s[i]))
        fit = curve_fit(gauss, d[i], s[i], p0 = initGuess)[0]
        d[i] -= fit[1]
        if normalize:
            s[i] -= fit[3]
        
#        com = np.sum(d[i]*s[i])/np.sum(s[i])
#        d[i]-=com
        
#        d[i] -= d[i][np.argmax(s[i])]
#        print(d[i][np.argmax(s[i])])
        
    if nPerPoint != 1:
        s = s.reshape(-1,nPerPoint).mean(axis = 1).reshape(len(wiresUsed),-1)
        d = d.reshape(-1,nPerPoint).mean(axis = 1).reshape(len(wiresUsed),-1)
        angle = angle.reshape(-1,nPerPoint).mean(axis = 1).reshape(len(wiresUsed),-1)
    if normalize:
        return s/np.max(s),d,angle
    else:
        return s,d,angle
        

def transformRep(x,y, spacingWraped180):
    d = np.zeros_like(x)
    angle = np.zeros_like(x)
    for line in np.arange(0,d.shape[0]):
        d[line] = np.sqrt((x[line]-x[line,0])**2+(y[line]-y[line,0])**2)
        d[line]-= np.average(d[line])
        if spacingWraped180 is True:
            modVal = np.pi
        else:
            modVal = 2*np.pi
        
        angle[line] = np.repeat(np.round(np.rad2deg((np.arctan2((y[line,-1]-y[line,0]),(x[line,-1]-x[line,0]))-np.pi/2)%(modVal))), x.shape[1])
    return [d, angle]


path = 'D:/Measurement Data/ATHOS/20200609/'

fileNames = ['S2scanZ-54741_20200609_061105',
             'S2scanZ-24741_20200609_055922',
             'S2scanZ-9741_20200609_053940', 
             'S2scanZ-5241_20200609_050402',
#             'S2scanZ-5241_20200609_052022',
             'S2scanZ-241_20200609_062337',
             'S2scanZ14758_20200609_063304',
             'S2scanZ44758_20200609_064353',
             ]

fileNames20200609 = [path+f+'/RAWScanData.h5' for f in fileNames]
z = (np.array([-54741, -24741, -9741, -5241, -241, 14759, 44759])+5241)*1e-6

path = 'D:/Measurement Data/ATHOS/20200826/ACHIP/'
    
fileNames = [
             'ws_20200826_052518',
             'ws_z4758_20200826_054008',
             'ws_z9758_20200826_055125',
             'ws_z19758_20200826_060107',
             'ws_z39758_20200826_062837',
             'ws_z79758_20200826_063944',
                     ]

fileNames20200826 = [path+f+'/RAWScanData.h5' for f in fileNames]
z = (np.array([0, 5000, 10000, 20000, 40000, 80000]))*1e-6

def loadData4d(fileNames=fileNames20200826, z=z, normalize = True):
    nZ = len(z)
    nAngles = 9
    nD = 31
    angles = np.linspace(0,320,nAngles)
    
#    D = np.array([[np.zeros(nD)]*nAngles]*nZ)
#    Signal = np.array([[np.zeros(nD)]*nAngles]*nZ)
    D = []
    Signal = []
    Angles = np.array([np.array([angles]*nD).T]*nZ)
    Z = np.moveaxis(np.array([np.array([z]*nAngles).T]*nD), [0,1,2], [2,0,1])
    
    for i, fileName in enumerate(fileNames):
        s,d,angle = loadAndPrepareInput(fileName, sensor='SensorValue/SATCL01-DBLM135:B2_LOSS', nWiresInFile=9, wiresUsed=[0,1,2,3,4,5,6,7,8], map180=False,nPerPoint = 1, normalize = normalize)
#        D[i] = d
#        Signal[i] = s
        D.append(d)
        Signal.append(s)
        

#    print(d.shape, Signal.shape, angles.shape )
    return z, angles, D, Signal

